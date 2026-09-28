"""
simulation.py
─────────────
Pure-computation module: no Streamlit, no Plotly, no UI of any kind.

Imported by:
  campaign_simulator.py  — Streamlit visualisation app
  evaluate.py            — parameter calibration / batch evaluation scripts

Public API
──────────
  DAYS                   int
  THRESHOLD              float
  RESCUE_MIN             float
  TEMPLATES              dict
  STRATEGY_NAMES         list[str]
  STRATEGY_DESCRIPTIONS  dict[str, str]
  SPEND_STRATEGIES       list[str]

  get_template(template, n, rng)  → (coords, names, shares)
  get_daily_spend(strategy, total_budget) → np.ndarray[DAYS]
  simulate(...)           → (history, pos_history)
"""

import math

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

THRESHOLD  = 0.05   # parliamentary viability threshold
RESCUE_MIN = 0.03   # rescue only parties still somewhat alive
DAYS       = 180    # simulation horizon (days)


# ─────────────────────────────────────────────
# Templates
#
# Compass axes:
#   x : Left (−1) → Right (+1)
#   y : Authoritarian (−1) → Liberal (+1)
#
# Real-world positions are informed by the Chapel Hill Expert Survey (CHES)
# lrecon and galtan scales, rescaled from [0, 10] to [−1, 1].
# Shares are approximate results / polling averages at the time of the
# relevant election cycle — use as starting points, not ground truth.
# ─────────────────────────────────────────────

TEMPLATES: dict = {
    # ── Generic procedural (n configurable) ──────────────────────────────────
    "Two large parties": None,
    "Multiparty":        None,
    "Dominant party":    None,
    "Fragmented":        None,
}

df = pd.read_csv("templates.csv")

for system, group in df.groupby("system"):
    TEMPLATES[system] = {
        "names": group["party"].tolist(),
        "x": group["x"].tolist(),
        "y": group["y"].tolist(),
        "shares": group["share"].tolist(),
    }


def get_template(
    template: str,
    n: int | None,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """
    Parameters
    ----------
    template : key from TEMPLATES
    n        : number of parties (only used for generic procedural templates)
    rng      : numpy Generator for reproducible randomness

    Returns
    -------
    coords : (n_parties, 2) float array of compass positions
    names  : list of party name strings
    shares : (n_parties,) float array of initial vote shares, sums to 1
    """
    if template in TEMPLATES and TEMPLATES[template] is not None:
        t      = TEMPLATES[template]
        coords = np.column_stack([t["x"], t["y"]]).astype(float)
        names  = list(t["names"])
        shares = np.array(t["shares"], dtype=float)
        shares /= shares.sum()
        return coords, names, shares

    # Generic procedural
    assert n is not None, "n must be provided for generic templates"
    names = [chr(65 + i) for i in range(n)]

    if template == "Two large parties":
        coords = np.vstack([
            np.array([[-0.7, 0.0], [0.7, 0.0]]),
            rng.uniform(-0.5, 0.5, (n - 2, 2)),
        ])
        shares = np.array([0.35, 0.35] + [0.30 / (n - 2)] * (n - 2))

    elif template == "Multiparty":
        coords = np.column_stack([np.linspace(-1, 1, n), rng.normal(0, 0.1, n)])
        shares = np.full(n, 1.0 / n)

    elif template == "Dominant party":
        coords = np.vstack([[[0, 0]], rng.uniform(-1, 1, (n - 1, 2))])
        shares = np.array([0.50] + [0.50 / (n - 1)] * (n - 1))

    elif template == "Fragmented":
        coords = rng.uniform(-1.0, 1.0, (n, 2))
        shares = rng.dirichlet(np.ones(n))

    else:
        coords = rng.uniform(-1.0, 1.0, (n, 2))
        shares = np.full(n, 1.0 / n)

    shares = shares / shares.sum()
    return coords, names, shares


# ─────────────────────────────────────────────
# Private Poll
# ─────────────────────────────────────────────

def private_poll(
    true_shares: np.ndarray,
    poll_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Simulate a private poll as a multinomial draw from the true distribution.
    Larger poll_size → less sampling noise → better-informed movement decisions.
    """
    p = np.clip(true_shares, 0, 1)
    p = p / p.sum()
    counts = rng.multinomial(int(poll_size), p)
    s = counts.sum()
    return counts / s if s > 0 else p.copy()


# ─────────────────────────────────────────────
# Movement Strategies
# ─────────────────────────────────────────────

STRATEGY_NAMES: list[str] = [
    "None (hold position)",
    "Drift to Centre",
    "Oppose Leader",
    "Mirror Leader",
    "Siphon Small Parties",
    "Middle Ground",
]

STRATEGY_DESCRIPTIONS: dict[str, str] = {
    "None (hold position)":
        "Hold your platform. Let advertising do the work.",
    "Drift to Centre":
        "Glide toward (0, 0) over the campaign — median-voter strategy, "
        "arriving at the centre by election day.",
    "Oppose Leader":
        "Steadily move to the ideological mirror of the leading party, "
        "arriving at maximum differentiation by election day.",
    "Mirror Leader":
        "Converge on the leader's current position by election day, "
        "contesting their voter base directly.",
    "Siphon Small Parties":
        "Move toward the centroid of sub-threshold parties, weighted by "
        "poll share, arriving there by election day.",
    "Middle Ground":
        "Glide to a position between the leader and the challenger, "
        "blocking the challenger's growth by election day.",
}


def _target(
    name: str,
    coords: np.ndarray,
    noisy_poll: np.ndarray,
    campaign_idx: int,
    initial_leader: int | None = None,
) -> np.ndarray:
    """
    Return the *desired endpoint* for the campaign party under this strategy,
    given the current private poll snapshot.  The caller is responsible for
    deciding how far to move this cycle.

    Parameters
    ----------
    name         : one of STRATEGY_NAMES
    coords       : (n, 2) current compass positions
    noisy_poll   : (n,) private poll result
    campaign_idx : index of the party being campaigned for

    Returns
    -------
    target : (2,) desired compass position (un-stepped, un-clamped)
    """
    pos    = coords[campaign_idx].copy()
    n      = len(noisy_poll)
    leader = int(np.argmax(noisy_poll))
    others = [i for i in range(n) if i != campaign_idx]

    if name == "None (hold position)":
        return pos

    elif name == "Drift to Centre":
        return np.zeros(2)

    elif name == "Oppose Leader":
        leader = leader if initial_leader is None else initial_leader
        if leader == campaign_idx:
            return pos
        return -coords[leader]

    elif name == "Mirror Leader":
        if leader == campaign_idx:
            return pos
        return coords[leader].copy()

    elif name == "Siphon Small Parties":
        targets = [i for i in others if noisy_poll[i] < 0.05]
        if not targets:
            targets = others
        if not targets:
            return pos
        w = noisy_poll[targets]
        if w.sum() < 1e-9:
            return pos
        return np.average(coords[targets], axis=0, weights=w)

    elif name == "Middle Ground":
        non_leader = [i for i in others if i != leader]
        if not non_leader:
            return pos
        challenger = non_leader[int(np.argmax(noisy_poll[non_leader]))]
        if leader != campaign_idx:
            return (coords[leader] + coords[challenger]) / 2.0
        else:
            return coords[challenger].copy()

    return pos


# ─────────────────────────────────────────────
# Campaign Spending Schedule
# ─────────────────────────────────────────────

SPEND_STRATEGIES: list[str] = ["Uniform", "Front-loaded", "Back-loaded"]


def get_daily_spend(strategy: str, total_budget: float) -> np.ndarray:
    """
    Return a (DAYS,) array of per-day spend amounts that sum to total_budget.

    Strategies
    ----------
    Uniform      : flat spend across all days
    Front-loaded : exponential decay — heavy early, tapers off
    Back-loaded  : exponential ramp — builds toward election day
    """
    arr = np.zeros(DAYS)
    if total_budget <= 0:
        return arr

    if strategy == "Uniform":
        arr[:] = total_budget / DAYS

    elif strategy == "Front-loaded":
        w   = np.exp(-np.linspace(0, 3, DAYS))
        arr = w / w.sum() * total_budget

    elif strategy == "Back-loaded":
        w   = np.exp(np.linspace(-3, 0, DAYS))
        arr = w / w.sum() * total_budget

    return arr


# ─────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────

def simulate(
    shares:            np.ndarray,
    coords_in:         np.ndarray,
    names:             list[str],
    poll_interval:     int,
    tactical:          float,
    bandwagon:         float,
    anti_leader:       float,
    loyalty:           float,
    campaign_idx:      int,
    daily_spend:       np.ndarray,
    effectiveness:     float,
    movement_strategy: str,
    poll_size:         int,
    seed:              int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the political dynamics simulation for DAYS days.

    Movement model
    ──────────────
    Each poll cycle the strategy is re-evaluated against the private poll to
    produce a *desired target* position.  The party then moves a fraction of
    the distance to that target equal to 1 / remaining_poll_cycles, so that
    it arrives exactly at the target on the final poll day of the simulation.
    This produces smooth, purposeful trajectories with no hard step cap.

    Parameters
    ----------
    shares            : (n,) initial vote shares, must sum to 1
    coords_in         : (n, 2) initial compass positions
    names             : party name strings (length n)
    poll_interval     : how many days between public poll cycles
    tactical          : strength of tactical voting + threshold rescue [0, 1]
    bandwagon         : strength of bandwagon drift toward leader [0, 1]
    anti_leader       : strength of anti-leader rally behind challenger [0, 1]
    loyalty           : voter loyalty — inhibits all flows [0, 1]
    campaign_idx      : index of the party being actively campaigned for
    daily_spend       : (DAYS,) per-day advertising budget array
    effectiveness     : voter conversion per unit of daily spend
    movement_strategy : one of STRATEGY_NAMES
    poll_size         : number of respondents in the private poll
    seed              : RNG seed for reproducibility

    Returns
    -------
    history     : (DAYS, n)  vote share of each party on each day
    pos_history : (DAYS, 2)  compass position of campaign_idx on each day
    """
    rng     = np.random.default_rng(seed)
    coords  = coords_in.copy().astype(float)
    dists   = cdist(coords, coords)
    n       = len(names)
    current = shares.copy().astype(float)
    current = current / current.sum()
    initial_leader = int(np.argmax(current))
    original_shares = shares.copy()
    arrival_fraction = 2 / 3

    # Pre-compute the total number of poll cycles so we know how far to move
    # each cycle to arrive at the target by the last poll day.
    total_poll_cycles = math.ceil(DAYS / poll_interval)

    history  : list[np.ndarray] = []
    pos_hist : list[np.ndarray] = []
    poll_cycle_count = 0  # how many poll cycles have elapsed (including this one)

    for day in range(DAYS):
        history.append(current.copy())
        pos_hist.append(coords[campaign_idx].copy())

        mob = 1.0 - loyalty   # fraction of voters that are mobile

        # ── Advertising ────────────────────────────────────────────────────
        spend = daily_spend[day] if day < len(daily_spend) else 0.0
        if spend > 0 and effectiveness > 0:
            oth      = [i for i in range(n) if i != campaign_idx]
            weights  = current[oth] * np.exp(-dists[campaign_idx, oth])
            total_w  = weights.sum()
            boost    = spend * effectiveness * mob * total_w   # scales with reachable voter mass — more effective when surrounded by close rivals
            for idx, i in enumerate(oth):
                current[i] -= boost * (weights[idx] / total_w)
            current[campaign_idx] += boost
            current = np.clip(current, 0, 1)
            current /= current.sum()

        # ── Poll-cycle dynamics ────────────────────────────────────────────
        if day % poll_interval == 0:
            poll_cycle_count += 1
            pub    = np.clip(current, 0, 1)
            pub   /= pub.sum()
            leader = int(np.argmax(pub))
            viable = pub >= THRESHOLD

            # Tactical voting: sub-threshold voters flee to nearest viable party
            if any(~viable) and any(viable):
                for i in np.where(~viable)[0]:
                    v    = np.where(viable)[0]
                    t    = v[int(np.argmin(dists[i, v]))]
                    flow = current[i] * tactical * mob
                    current[i] -= flow
                    current[t] += flow

            # Bandwagon: proximity-weighted drift toward the leader
            for i in range(n):
                if i != leader:
                    flow = current[i] * bandwagon * np.exp(-dists[i, leader]) * mob
                    current[i]      -= flow
                    current[leader] += flow

            # Anti-leader: rally behind the challenger (2nd-largest party)
            oth_l = [x for x in range(n) if x != leader]
            if oth_l:
                challenger = oth_l[int(np.argmax(pub[oth_l]))]
                for i in oth_l:
                    if i != challenger:
                        flow = current[i] * anti_leader * np.exp(-dists[i, challenger]) * mob
                        current[i]          -= flow
                        current[challenger] += flow

            # Threshold rescue: leader voters lend support to near-threshold allies
            for small in range(n):
                if small == leader:
                    continue
                if not (RESCUE_MIN <= pub[small] < THRESHOLD):
                    continue
                proximity = np.exp(-dists[leader, small])
                urgency   = 1.0 - (pub[small] - RESCUE_MIN) / (THRESHOLD - RESCUE_MIN)
                flow      = current[leader] * tactical * proximity * urgency * mob
                flow      = min(flow, current[leader] * 0.05)
                current[leader] -= flow
                current[small]  += flow

            # ── Movement ──────────────────────────────────────────────────
            # The strategy returns the *desired endpoint*.  We move a fraction
            # 1 / remaining_cycles toward it so we arrive exactly on the last
            # poll day — no hard step cap needed.
            if movement_strategy != "None (hold position)":
                priv    = private_poll(current, poll_size, rng)
                desired = _target(
                    movement_strategy,
                    coords,
                    priv,
                    campaign_idx,
                    initial_leader=initial_leader,
                )
                desired = np.clip(desired, -1.5, 1.5)

                arrival_cycle    = max(1, round(total_poll_cycles * arrival_fraction))
                remaining_cycles = max(1, arrival_cycle - poll_cycle_count + 1)
                delta            = (desired - coords[campaign_idx]) / remaining_cycles * (poll_cycle_count <= arrival_cycle)
                new_pos          = np.clip(coords[campaign_idx] + delta, -1.5, 1.5)

                old_pos = coords[campaign_idx].copy()
                moved   = np.linalg.norm(new_pos - old_pos)
                if moved > 1e-6:
                    coords[campaign_idx] = new_pos
                    dists = cdist(coords, coords)

                    # Alienation: voters near the party's origin defect to
                    # rivals who are now closer to the abandoned ground.
                    home        = coords_in[campaign_idx]
                    d_party     = np.linalg.norm(new_pos - home)          # cumulative drift from origin — voters care about total displacement, not step size
                    oth_idx     = [j for j in range(n) if j != campaign_idx]
                    d_rivals    = np.linalg.norm(coords[oth_idx] - home, axis=1)

                    pull_rivals = (current[oth_idx]          # size: large rivals are more credible alternatives
                                   * np.exp(-d_rivals)       # proximity to home ground: standard spatial voting decay
                                   * (d_rivals < d_party))   # eligibility: only rivals closer to home than the party itself
                    pull_party  = np.exp(-d_party)            # party's own decaying claim on its home voters
                    party_share = pull_party / (pull_party + pull_rivals.sum() + 1e-12)  # multinomial logit share among all options

                    pool        = original_shares[campaign_idx] * (1.0 - party_share) * mob                 # voters lost = complement of logit share, damped by loyalty
                    w_sum       = pull_rivals.sum() + 1e-12
                    for i, j in enumerate(oth_idx):
                        flow                   = pool * (pull_rivals[i] / w_sum)  # distribute defectors proportionally to each rival's pull
                        original_shares[campaign_idx] -= flow
                        current[campaign_idx] -= flow
                        current[j]            += flow

                    current = np.clip(current, 0, 1)
                    current /= current.sum()

        current  = np.clip(current + rng.normal(0, 0.002, n), 0, 1)
        current /= current.sum()

    return np.array(history), np.array(pos_hist)
