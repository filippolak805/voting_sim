from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy.spatial.distance import cdist

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
THRESHOLD = 0.05
DAYS      = 180
ANIM_STEP = 3
FRAME_MS  = 80

st.set_page_config(layout="wide")
st.title("Macro-Political Dynamics — Baseline vs Fake Poll")

# ─────────────────────────────────────────────
# Poll Pipeline
# ─────────────────────────────────────────────

@dataclass
class PollModifier:
    name:   str
    active: bool
    apply:  Callable[[np.ndarray, int, List[str]], np.ndarray]


def make_fake_boost(target_parties, boost_factor, start_day, end_day) -> PollModifier:
    def _apply(poll, day, party_names):
        if not (start_day <= day <= end_day):
            return poll
        out = poll.copy()
        for name in target_parties:
            if name in party_names:
                out[party_names.index(name)] *= 1.0 + boost_factor
        return out / out.sum()
    return PollModifier(
        name=f"Boost {target_parties} x{1+boost_factor:.2f} (days {start_day}-{end_day})",
        active=True, apply=_apply,
    )


def apply_pipeline(raw_poll, day, party_names, modifiers):
    poll = raw_poll.copy()
    for mod in modifiers:
        if mod.active:
            poll = mod.apply(poll, day, party_names)
    s = poll.sum()
    return poll / s if s > 0 else poll


# ─────────────────────────────────────────────
# Spatial Templates
# ─────────────────────────────────────────────

def get_template(template, n, rng):
    names = [chr(65 + i) for i in range(n)]
    if template == "Two large parties":
        pos = np.vstack([np.array([[-0.7, 0.0], [0.7, 0.0]]),
                         rng.uniform(-0.5, 0.5, (n - 2, 2))])
        w   = np.array([0.35, 0.35] + [0.30 / (n - 2)] * (n - 2))
    elif template == "Multiparty":
        pos = np.column_stack([np.linspace(-1, 1, n), rng.normal(0, 0.1, n)])
        w   = np.full(n, 1.0 / n)
    elif template == "Dominant party":
        pos = np.vstack([[[0, 0]], rng.uniform(-1, 1, (n - 1, 2))])
        w   = np.array([0.50] + [0.50 / (n - 1)] * (n - 1))
    elif template == "Fragmented":
        pos = rng.uniform(-1.0, 1.0, (n, 2))
        w   = rng.dirichlet(np.ones(n))
    else:
        pos = rng.uniform(-1.0, 1.0, (n, 2))
        w   = np.full(n, 1.0 / n)
    return pos, names, w


# ─────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────

def hex_colors(n):
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap("tab20")
    return [
        "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        for r, g, b, _ in [cmap(i / max(n - 1, 1)) for i in range(n)]
    ]


# ─────────────────────────────────────────────
# Ideological Drift
# ─────────────────────────────────────────────

def apply_ideology_drift(coords, mode, strength, moving_mask=None):
    if mode == "Static" or strength <= 0:
        return coords

    updated = coords.copy()
    if moving_mask is None:
        moving_mask = np.ones(len(coords), dtype=bool)
    if not np.any(moving_mask):
        return coords

    if mode == "Toward center":
        target = np.zeros_like(coords)
        updated[moving_mask] += strength * (target[moving_mask] - coords[moving_mask])
    elif mode == "Away from center":
        norms = np.linalg.norm(coords, axis=1, keepdims=True)
        outward = np.divide(
            coords,
            np.where(norms > 1e-9, norms, 1.0),
            out=np.zeros_like(coords),
            where=np.ones_like(coords, dtype=bool),
        )
        outward[norms[:, 0] <= 1e-9] = np.array([1.0, 0.0])
        updated[moving_mask] += strength * outward[moving_mask]
    elif mode == "Polarize left-right":
        target_x = np.where(coords[:, 0] >= 0, 1.5, -1.5)
        updated[moving_mask, 0] += strength * (target_x[moving_mask] - coords[moving_mask, 0])
    elif mode == "Polarize auth-lib":
        target_y = np.where(coords[:, 1] >= 0, 1.5, -1.5)
        updated[moving_mask, 1] += strength * (target_y[moving_mask] - coords[moving_mask, 1])

    return np.clip(updated, -1.5, 1.5)


# ─────────────────────────────────────────────
# Simulation — pure computation
# ─────────────────────────────────────────────

def simulate(shares, coords, names, poll_interval,
             tactical, bandwagon, anti_leader, loyalty,
             modifiers, seed, ideology_mode="Static", ideology_strength=0.0,
             moving_parties=None):
    rng   = np.random.default_rng(seed)
    n     = len(names)
    current = shares.copy()
    current_coords = coords.copy()
    moving_mask = np.isin(names, moving_parties) if moving_parties else np.zeros(n, dtype=bool)
    share_history = []
    coord_history = []

    for day in range(DAYS):
        if day > 0 and day % poll_interval == 0:
            current_coords = apply_ideology_drift(
                current_coords, ideology_mode, ideology_strength, moving_mask
            )

        share_history.append(current.copy())
        coord_history.append(current_coords.copy())
        mobility = (1.0 - loyalty) * 0.2

        if day % poll_interval == 0:
            dists = cdist(current_coords, current_coords)
            raw = np.clip(current + rng.normal(0, 0.02, n), 0, 1)
            raw /= raw.sum()
            poll = apply_pipeline(raw, day, names, modifiers)
            leader = np.argmax(poll)
            viable = poll >= THRESHOLD

            if any(~viable) and any(viable):
                for i in np.where(~viable)[0]:
                    v = np.where(viable)[0]
                    t = v[np.argmin(dists[i, v])]
                    flow = current[i] * tactical * mobility
                    current[i] -= flow;  current[t] += flow

            for i in range(n):
                if i != leader:
                    flow = current[i] * bandwagon * np.exp(-dists[i, leader]) * mobility
                    current[i] -= flow;  current[leader] += flow

            others = [x for x in range(n) if x != leader]
            if others:
                t    = others[np.argmin(dists[leader, others])]
                loss = current[leader] * anti_leader * mobility
                current[leader] -= loss;  current[t] += loss

        current = np.clip(current + rng.normal(0, 0.002, n), 1e-4, 1)
        current /= current.sum()

    return {
        "shares": np.array(share_history),
        "coords": np.array(coord_history),
    }


# ─────────────────────────────────────────────
# Combined Animated Figure
# Trace index layout (4n + 2 total):
#   0   … n-1  : baseline trend lines
#   n   … 2n-1 : fake-poll trend lines      (hidden if no fake run)
#   2n  … 3n-1 : compass baseline bubbles
#   3n  … 4n-1 : compass fake bubbles       (hidden if no fake run)
#   4n          : bar — baseline
#   4n+1        : bar — fake poll            (hidden if no fake run)
# ─────────────────────────────────────────────

def build_figure(hb, hf, names, colors):
    n        = len(names)
    has_fake = hf is not None
    hb_shares = hb["shares"]
    hb_coords = hb["coords"]
    hf_shares = hf["shares"] if has_fake else None
    hf_coords = hf["coords"] if has_fake else None
    n_days   = len(hb_shares)
    all_idx  = list(range(4 * n + 2))

    frame_days = list(range(ANIM_STEP, n_days + 1, ANIM_STEP))
    if frame_days[-1] != n_days:
        frame_days.append(n_days)

    all_coords = hb_coords if not has_fake else np.concatenate([hb_coords, hf_coords], axis=0)
    x0, x1 = all_coords[:, :, 0].min(), all_coords[:, :, 0].max()
    y0, y1 = all_coords[:, :, 1].min(), all_coords[:, :, 1].max()
    px = max((x1 - x0) * 0.25, 0.4)
    py = max((y1 - y0) * 0.25, 0.4)
    max_share = max(hb_shares.max(), hf_shares.max() if has_fake else 0)

    fig = make_subplots(
        rows=2, cols=2,
        specs=[[{"colspan": 2, "type": "scatter"}, None],
               [{"type": "scatter"}, {"type": "bar"}]],
        subplot_titles=[
            "Support Over Time",
            "Political Compass (bubble = vote share)",
            "Vote Shares",
        ],
        row_heights=[0.52, 0.48],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    # ── Initial traces (full history on load) ──────────────

    # Baseline trend lines
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=list(range(n_days)), y=hb_shares[:, i],
            mode="lines", name=f"{name} Baseline",
            line=dict(color=colors[i], width=2),
            legendgroup=f"base_{i}", showlegend=True,
            visible=True,
            hovertemplate=f"<b>{name} Baseline</b>: %{{y:.1%}}<extra></extra>",
        ), row=1, col=1)

    # Fake trend lines
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=list(range(n_days)) if has_fake else [],
            y=hf_shares[:, i] if has_fake else [],
            mode="lines", name=f"{name} Fake Poll",
            line=dict(color=colors[i], width=2, dash="dash"),
            legendgroup=f"fake_{i}", showlegend=True,
            visible=True if has_fake else "legendonly",
            hovertemplate=f"<b>{name} Fake Poll</b>: %{{y:.1%}}<extra></extra>",
        ), row=1, col=1)

    # Compass — baseline bubbles (faded when fake exists so fake is prominent)
    for i, name in enumerate(names):
        s = hb_shares[-1, i]
        fig.add_trace(go.Scatter(
            x=[hb_coords[-1, i, 0]], y=[hb_coords[-1, i, 1]],
            mode="markers+text", name=name,
            text=[name], textposition="middle center",
            marker=dict(size=max(s * 110 + 14, 6), color=colors[i],
                        opacity=0.4 if has_fake else 1.0,
                        line=dict(color="black", width=1)),
            legendgroup=f"p{i}", showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Baseline: {s:.1%}<extra></extra>",
        ), row=2, col=1)

    # Compass — fake bubbles
    for i, name in enumerate(names):
        s = hf_shares[-1, i] if has_fake else 0.0
        fig.add_trace(go.Scatter(
            x=[hf_coords[-1, i, 0]] if has_fake else [hb_coords[-1, i, 0]],
            y=[hf_coords[-1, i, 1]] if has_fake else [hb_coords[-1, i, 1]],
            mode="markers+text" if has_fake else "markers",
            name=f"{name} (Fake)", text=[name] if has_fake else [""],
            textposition="middle center",
            marker=dict(size=max(s * 110 + 14, 3) if has_fake else 3,
                        color=colors[i], line=dict(color="black", width=2)),
            legendgroup=f"p{i}", showlegend=False,
            visible=has_fake,
            hovertemplate=f"<b>{name}</b><br>Fake Poll: {s:.1%}<extra></extra>",
        ), row=2, col=1)

    # Bar — baseline
    fig.add_trace(go.Bar(
        x=names, y=hb_shares[-1], name="Baseline",
        marker_color=colors, opacity=0.9, showlegend=False,
        hovertemplate="<b>%{x}</b> Baseline: %{y:.1%}<extra></extra>",
    ), row=2, col=2)

    # Bar — fake
    fig.add_trace(go.Bar(
        x=names, y=hf_shares[-1] if has_fake else [0] * n,
        name="Fake Poll", marker_color=colors, opacity=0.45,
        marker_line=dict(color="black", width=1),
        showlegend=False, visible=True if has_fake else "legendonly",
        hovertemplate="<b>%{x}</b> Fake Poll: %{y:.1%}<extra></extra>",
    ), row=2, col=2)

    # ── Animation frames ────────────────────────────────────

    frames = []
    for d in frame_days:
        fd = []
        for i in range(n):   # baseline trend
            fd.append(go.Scatter(x=list(range(d)), y=hb_shares[:d, i]))
        for i in range(n):   # fake trend
            fd.append(go.Scatter(
                x=list(range(d)) if has_fake else [],
                y=hf_shares[:d, i] if has_fake else [],
            ))
        for i in range(n):   # compass baseline
            s = hb_shares[d - 1, i]
            fd.append(go.Scatter(
                x=[hb_coords[d - 1, i, 0]], y=[hb_coords[d - 1, i, 1]],
                text=[names[i]], textposition="middle center",
                marker=dict(size=max(s * 110 + 14, 6)),
            ))
        for i in range(n):   # compass fake
            if has_fake:
                s = hf_shares[d - 1, i]
                fd.append(go.Scatter(
                    x=[hf_coords[d - 1, i, 0]], y=[hf_coords[d - 1, i, 1]],
                    text=[names[i]], textposition="middle center",
                    marker=dict(size=max(s * 110 + 14, 3)),
                ))
            else:
                fd.append(go.Scatter(x=[hb_coords[d - 1, i, 0]], y=[hb_coords[d - 1, i, 1]],
                                     marker=dict(size=3)))
        fd.append(go.Bar(y=hb_shares[d - 1]))                               # bar baseline
        fd.append(go.Bar(y=hf_shares[d - 1] if has_fake else [0] * n))      # bar fake
        frames.append(go.Frame(data=fd, name=str(d), traces=all_idx))

    fig.frames = frames

    # Viability threshold line (shape — unaffected by frames)
    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=0, x1=n_days, y0=THRESHOLD, y1=THRESHOLD,
        line=dict(color="red", dash="dot", width=1),
        row=1, col=1,
    )

    fig.update_layout(
        height=780, barmode="group", hovermode="closest",
        margin=dict(l=10, r=10, t=90, b=80),
        legend=dict(orientation="h", y=1.06, x=0, xanchor="left"),
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.13, x=1.0, xanchor="right", yanchor="top",
            buttons=[
                dict(label="▶  Play All", method="animate",
                     args=[None, {"frame": {"duration": FRAME_MS, "redraw": True},
                                  "fromcurrent": False,
                                  "transition": {"duration": 0}}]),
                dict(label="⏸  Pause", method="animate",
                     args=[[None], {"frame": {"duration": 0}, "mode": "immediate",
                                    "transition": {"duration": 0}}]),
            ],
        )],
        sliders=[dict(
            steps=[dict(
                args=[[str(d)], {"frame": {"duration": 0}, "mode": "immediate",
                                  "transition": {"duration": 0}}],
                label=str(d), method="animate",
            ) for d in frame_days],
            x=0, y=-0.02, len=1.0, pad={"t": 10},
            currentvalue=dict(prefix="Day: ", visible=True, xanchor="center"),
            transition=dict(duration=0),
        )],
    )

    fig.update_xaxes(range=[0, n_days], title_text="Day", row=1, col=1)
    fig.update_yaxes(range=[0, min(max_share * 1.15, 1.0)], tickformat=".0%", row=1, col=1)
    fig.update_xaxes(range=[x0 - px, x1 + px], title_text="Left / Right", row=2, col=1)
    fig.update_yaxes(range=[y0 - py, y1 + py], title_text="Auth / Lib",   row=2, col=1)
    fig.update_yaxes(range=[0, min(max_share * 1.15, 1.0)], tickformat=".0%", row=2, col=2)

    return fig


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

for k, v in {
    "history_base": None,
    "history_fake": None,
    "custom_pos":   None,
    "custom_w":     None,
    "rand_n":       None,
    "rand_ver":     0,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.title("Simulation Controls")
poll_interval = st.sidebar.slider("Poll frequency (days)", 3, 14, 7)

st.sidebar.subheader("Voter Behaviour")
tactical    = st.sidebar.slider("Tactical voting strength", 0.0, 1.0, 0.40)
bandwagon   = st.sidebar.slider("Bandwagon effect",         0.0, 1.0, 0.15)
anti_leader = st.sidebar.slider("Anti-leader effect",       0.0, 1.0, 0.05)
loyalty     = st.sidebar.slider("Voter loyalty",            0.0, 1.0, 0.85)

st.sidebar.subheader("Party Configuration")
template  = st.sidebar.selectbox(
    "Template",
    ["Two large parties", "Multiparty", "Dominant party", "Fragmented", "Custom"],
)
n_parties = st.sidebar.slider("Number of parties", 3, 15, 8)

rng0 = np.random.default_rng(42)
default_pos, names, default_w = get_template(template, n_parties, rng0)
colors = hex_colors(n_parties)

rcol1, rcol2 = st.sidebar.columns(2)
if rcol1.button("🎲 Positions", use_container_width=True):
    st.session_state.custom_pos = np.random.default_rng().uniform(-1.0, 1.0, (n_parties, 2))
    st.session_state.rand_n   = n_parties
    st.session_state.rand_ver += 1
if rcol2.button("🎲 Shares", use_container_width=True):
    st.session_state.custom_w = np.random.default_rng().dirichlet(np.ones(n_parties)) * 100
    st.session_state.rand_n   = n_parties
    st.session_state.rand_ver += 1

same_n  = st.session_state.rand_n == n_parties
use_pos = same_n and st.session_state.custom_pos is not None
use_w   = same_n and st.session_state.custom_w   is not None
init_pos = st.session_state.custom_pos if use_pos else default_pos
init_w   = st.session_state.custom_w   if use_w   else default_w * 100

df0 = pd.DataFrame({
    "Party":      names,
    "Weight (%)": np.round(init_w, 1),
    "X-Pos":      np.round(init_pos[:, 0], 2),
    "Y-Pos":      np.round(init_pos[:, 1], 2),
})
edf = st.sidebar.data_editor(
    df0, hide_index=True, use_container_width=True,
    key=f"party_editor_{st.session_state.rand_ver}",
    column_config={
        "X-Pos": st.column_config.NumberColumn(min_value=-1.5, max_value=1.5, step=0.05),
        "Y-Pos": st.column_config.NumberColumn(min_value=-1.5, max_value=1.5, step=0.05),
    },
)
shares = edf["Weight (%)"].values.astype(float)
shares /= shares.sum()
coords  = edf[["X-Pos", "Y-Pos"]].values.astype(float)

st.sidebar.subheader("Ideological Movement")
ideology_mode = st.sidebar.selectbox(
    "Movement pattern",
    ["Static", "Toward center", "Away from center", "Polarize left-right", "Polarize auth-lib"],
    help="Applied once every poll interval to gradually move party positions.",
)
ideology_strength = st.sidebar.slider(
    "Movement per poll",
    0.0, 0.35, 0.08, 0.01,
    help="Higher values make parties shift faster on the political compass.",
)
moving_parties = st.sidebar.multiselect(
    "Parties that move",
    names,
    default=names,
    help="Only these parties will shift ideologically. Others stay fixed.",
)

st.sidebar.subheader("Fake Poll")
fake_enabled = st.sidebar.toggle("Enable fake poll manipulation", value=False)
fake_parties = st.sidebar.multiselect("Target parties", names, default=[names[0]])
fake_boost   = st.sidebar.slider("Boost factor", 0.05, 1.0, 0.50,
    help="+50 % means the party appears 50 % stronger in published polls")
fake_start   = st.sidebar.slider("Active from day", 0, DAYS - 1, 20)
fake_end     = st.sidebar.slider("Active until day", 1, DAYS, 120)

modifiers = []
if fake_enabled and fake_parties:
    modifiers.append(make_fake_boost(fake_parties, fake_boost, fake_start, fake_end))


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────

btn_base, btn_fake, btn_exp = st.columns([2, 2, 1])
run_base = btn_base.button("▶ Run Baseline",        use_container_width=True)
run_fake = btn_fake.button("▶ Run with Fake Polls", use_container_width=True,
                            disabled=not fake_enabled)
prog          = st.empty()
chart_area    = st.empty()

# ── Baseline only ────────────────────────────
if run_base:
    seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    prog.progress(0, text="Computing baseline…")
    hb = simulate(shares, coords, names, poll_interval,
                  tactical, bandwagon, anti_leader, loyalty,
                  modifiers=[], seed=seed,
                  ideology_mode=ideology_mode,
                  ideology_strength=ideology_strength,
                  moving_parties=moving_parties)
    st.session_state.history_base = hb
    st.session_state.history_fake = None   # old fake run is now stale
    prog.empty()

# ── Fake polls — runs BOTH simulations with the same seed ────────
if run_fake:
    seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)

    prog.progress(0, text="Computing baseline…")
    hb = simulate(shares, coords, names, poll_interval,
                  tactical, bandwagon, anti_leader, loyalty,
                  modifiers=[], seed=seed,
                  ideology_mode=ideology_mode,
                  ideology_strength=ideology_strength,
                  moving_parties=moving_parties)
    st.session_state.history_base = hb

    prog.progress(0, text="Computing fake-poll run…")
    hf = simulate(shares, coords, names, poll_interval,
                  tactical, bandwagon, anti_leader, loyalty,
                  modifiers=modifiers, seed=seed,
                  ideology_mode=ideology_mode,
                  ideology_strength=ideology_strength,
                  moving_parties=moving_parties)
    st.session_state.history_fake = hf

    prog.empty()

# ── Draw (once, no flickering) ───────────────
hb = st.session_state.history_base
hf = st.session_state.history_fake

if hb is not None:
    chart_area.plotly_chart(
        build_figure(hb, hf, names, colors),
        use_container_width=True,
    )

# ── Export ────────────────────────────────────
if hb is not None or hf is not None:
    dfs = []
    if hb is not None:
        df = pd.DataFrame(hb["shares"], columns=names)
        df_x = pd.DataFrame(hb["coords"][:, :, 0], columns=[f"{name}_X" for name in names])
        df_y = pd.DataFrame(hb["coords"][:, :, 1], columns=[f"{name}_Y" for name in names])
        df = pd.concat([df, df_x, df_y], axis=1)
        df.insert(0, "Day", range(len(df))); df.insert(0, "Run", "Baseline")
        dfs.append(df)
    if hf is not None:
        df = pd.DataFrame(hf["shares"], columns=names)
        df_x = pd.DataFrame(hf["coords"][:, :, 0], columns=[f"{name}_X" for name in names])
        df_y = pd.DataFrame(hf["coords"][:, :, 1], columns=[f"{name}_Y" for name in names])
        df = pd.concat([df, df_x, df_y], axis=1)
        df.insert(0, "Day", range(len(df))); df.insert(0, "Run", "Fake Poll")
        dfs.append(df)
    csv = pd.concat(dfs, ignore_index=True).to_csv(index=False)
    btn_exp.download_button(
        "⬇ Export CSV", csv, "results.csv", "text/csv",
        use_container_width=True,
    )
