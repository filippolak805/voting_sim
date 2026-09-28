"""
campaign_simulator.py
─────────────────────
Streamlit visualisation app.
All simulation logic lives in simulation.py — this file contains only
UI, session state, Plotly figure builders, and the thin glue between them.

Run with:
    streamlit run campaign_simulator.py
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from simulation import (
    DAYS,
    STRATEGY_NAMES,
    STRATEGY_DESCRIPTIONS,
    SPEND_STRATEGIES,
    TEMPLATES,
    get_template,
    get_daily_spend,
    simulate,
)

# ─────────────────────────────────────────────
# UI-only constants
# ─────────────────────────────────────────────

ANIM_STEP = 3
FRAME_MS  = 80
POLL_SIZE = 1000   # fixed (slider removed)

EXCLUDED_TEMPLATES = {"EU", "UK", "USA"}

st.set_page_config(layout="wide")
st.title("Campaign Strategy Simulator — Macro-Political Dynamics")


# ─────────────────────────────────────────────
# Colour helper (UI concern — not needed in simulation)
# ─────────────────────────────────────────────

def hex_colors(n: int) -> list[str]:
    import matplotlib.pyplot as plt
    cmap = plt.colormaps.get_cmap("tab20")
    return [
        "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))
        for r, g, b, _ in [cmap(i / max(n - 1, 1)) for i in range(n)]
    ]


# ─────────────────────────────────────────────
# Plotly figure builders
# ─────────────────────────────────────────────

def build_figure(hb, pb, hc, pc, names, colors, coords_init,
                 daily_spend, campaign_idx, has_campaign):
    """
    Build the main animated three-panel figure.

    Trace layout  (4n + 4 total):
      0          : spend area               [static, secondary-y]
      1 … n      : baseline trend lines
      n+1 … 2n   : campaign trend lines
      2n+1 … 3n  : compass baseline bubbles (fixed positions)
      3n+1 … 4n  : compass campaign bubbles (campaign party tracks pos_history)
      4n+1       : trajectory path
      4n+2       : bar — baseline
      4n+3       : bar — campaign
    animated_indices = 1 … 4n+3
    """
    n        = len(names)
    n_days   = len(hb)
    animated = list(range(1, 4 * n + 4))

    frame_days = list(range(ANIM_STEP, n_days + 1, ANIM_STEP))
    if frame_days[-1] != n_days:
        frame_days.append(n_days)

    max_share = max(hb.max(), hc.max() if has_campaign else 0)
    cp_color  = colors[campaign_idx]

    # ── Legend strategy: collapse to legendgroup toggles when there are many parties
    # Show one legend entry per party (not per baseline/campaign line separately)
    # to keep the legend compact regardless of party count.
    many_parties = n > 6

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"colspan": 2, "secondary_y": True}, None],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=[
            "Support Over Time",
            "Political Compass  (bubble = vote share  ● = campaign party)",
            "Final Vote Shares",
        ],
        row_heights=[0.52, 0.48],
        vertical_spacing=0.13,
        horizontal_spacing=0.08,
    )

    # Trace 0 — spend area (static, not animated)
    fig.add_trace(go.Scatter(
        x=list(range(n_days)), y=daily_spend[:n_days],
        fill="tozeroy", fillcolor="rgba(255,200,0,0.18)",
        line=dict(color="rgba(255,200,0,0.6)", width=1),
        name="Campaign Spend", showlegend=has_campaign, visible=has_campaign,
        hovertemplate="Day %{x} — spend: %{y:.3f}<extra></extra>",
    ), row=1, col=1, secondary_y=True)

    # Traces 1…n — baseline trends
    for i, name in enumerate(names):
        if many_parties:
            show_legend = True
            legend_name = name
        else:
            show_legend = True
            legend_name = f"{name} Baseline"

        fig.add_trace(go.Scatter(
            x=list(range(n_days)), y=hb[:, i],
            mode="lines", name=legend_name,
            line=dict(color=colors[i], width=3 if i == campaign_idx else 2),
            legendgroup=str(i),
            showlegend=show_legend,
            hovertemplate=f"<b>{name} Baseline</b>: %{{y:.1%}}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

    # Traces n+1…2n — campaign trends
    for i, name in enumerate(names):
        fig.add_trace(go.Scatter(
            x=list(range(n_days)) if has_campaign else [],
            y=hc[:, i]            if has_campaign else [],
            mode="lines", name=f"{name} Campaign",
            line=dict(color=colors[i], width=3 if i == campaign_idx else 2, dash="dash"),
            legendgroup=str(i),
            showlegend=False if many_parties else True,
            visible=True if has_campaign else "legendonly",
            hovertemplate=f"<b>{name} Campaign</b>: %{{y:.1%}}<extra></extra>",
        ), row=1, col=1, secondary_y=False)

    # ── Political Compass background quadrants ──────────────────────────────
    _CR = 1.2
    quadrant_colors = [
        ("rgba(255,100,100,0.10)", -_CR,   0,    0, _CR),  # top-left    Auth-Left
        ("rgba(100,100,255,0.10)",    0, _CR,    0, _CR),  # top-right   Auth-Right
        ("rgba(100,200,100,0.10)",  -_CR,   0, -_CR,   0),  # bottom-left Lib-Left
        ("rgba(255,200,80,0.10)",    0, _CR, -_CR,   0),  # bottom-right Lib-Right
    ]
    quadrant_labels = [
        ("Auth-Left",  -0.6,  0.9),
        ("Auth-Right",  0.6,  0.9),
        ("Lib-Left",   -0.6, -0.9),
        ("Lib-Right",   0.6, -0.9),
    ]
    for color, x0, x1, y0, y1 in quadrant_colors:
        fig.add_shape(
            type="rect", xref="x", yref="y",
            x0=x0, x1=x1, y0=y0, y1=y1,
            fillcolor=color, line_width=0, layer="below",
            row=2, col=1,
        )
    for label, lx, ly in quadrant_labels:
        fig.add_annotation(
            x=lx, y=ly, text=label,
            xref="x", yref="y",
            showarrow=False,
            font=dict(size=10, color="rgba(80,80,80,0.55)"),
            row=2, col=1,
        )

    # Traces 2n+1…3n — compass baseline bubbles (fixed)
    for i, name in enumerate(names):
        s           = hb[-1, i]
        is_campaign = (i == campaign_idx)
        fig.add_trace(go.Scatter(
            x=[coords_init[i, 0]], y=[coords_init[i, 1]],
            mode="markers+text", name=name,
            text=[name], textposition="middle center",
            marker=dict(
                size=max(s * 110 + 14, 6),
                color=colors[i],
                opacity=0.4 if has_campaign else 1.0,
                symbol="circle",
                line=dict(
                    color="white" if is_campaign else "rgba(0,0,0,0.4)",
                    width=4 if is_campaign else 1,
                ),
            ),
            legendgroup=f"p{i}", showlegend=False,
            hovertemplate=f"<b>{name}</b><br>Baseline: {s:.1%}<extra></extra>",
        ), row=2, col=1)

    # Traces 3n+1…4n — compass campaign bubbles
    for i, name in enumerate(names):
        final_pos   = pc[-1] if (has_campaign and i == campaign_idx) else coords_init[i]
        s           = hc[-1, i] if has_campaign else 0.0
        is_campaign = (i == campaign_idx)
        fig.add_trace(go.Scatter(
            x=[final_pos[0]], y=[final_pos[1]],
            mode="markers+text" if has_campaign else "markers",
            name=f"{name} (Campaign)",
            text=[name] if has_campaign else [""],
            textposition="middle center",
            marker=dict(
                size=max(s * 110 + 14, 3) if has_campaign else 3,
                color=colors[i],
                symbol="circle",
                line=dict(
                    color="white" if is_campaign else "rgba(0,0,0,0.6)",
                    width=4 if is_campaign else 2,
                ),
            ),
            legendgroup=f"p{i}", showlegend=False,
            visible=has_campaign,
            hovertemplate=f"<b>{name}</b><br>Campaign: {s:.1%}<extra></extra>",
        ), row=2, col=1)

    # Trace 4n+1 — campaign party trajectory path
    fig.add_trace(go.Scatter(
        x=pc[:, 0].tolist() if has_campaign else [],
        y=pc[:, 1].tolist() if has_campaign else [],
        mode="lines+markers",
        name=f"{names[campaign_idx]} path",
        line=dict(color=cp_color, width=2, dash="dot"),
        marker=dict(size=3, color=cp_color),
        showlegend=has_campaign, visible=has_campaign,
        hovertemplate="Day %{pointNumber}<br>(%{x:.2f}, %{y:.2f})<extra></extra>",
    ), row=2, col=1)

    # Trace 4n+2 — bar baseline
    fig.add_trace(go.Bar(
        x=names, y=hb[-1], name="Baseline",
        marker_color=colors, opacity=0.9, showlegend=False,
        hovertemplate="<b>%{x}</b> Baseline: %{y:.1%}<extra></extra>",
    ), row=2, col=2)

    # Trace 4n+3 — bar campaign
    fig.add_trace(go.Bar(
        x=names, y=hc[-1] if has_campaign else [0] * n,
        name="Campaign", marker_color=colors, opacity=0.45,
        marker_line=dict(color="black", width=1), showlegend=False,
        visible=True if has_campaign else "legendonly",
        hovertemplate="<b>%{x}</b> Campaign: %{y:.1%}<extra></extra>",
    ), row=2, col=2)

    # ── Animation frames ───────────────────────────────────────────────────
    frames = []
    for d in frame_days:
        fd = []
        for i in range(n):
            fd.append(go.Scatter(x=list(range(d)), y=hb[:d, i]))
        for i in range(n):
            fd.append(go.Scatter(
                x=list(range(d)) if has_campaign else [],
                y=hc[:d, i]      if has_campaign else [],
            ))
        for i in range(n):
            s           = hb[d - 1, i]
            is_campaign = (i == campaign_idx)
            fd.append(go.Scatter(
                x=[coords_init[i, 0]], y=[coords_init[i, 1]],
                text=[names[i]], textposition="middle center",
                marker=dict(
                    size=max(s * 110 + 14, 6),
                    line=dict(
                        color="white" if is_campaign else "rgba(0,0,0,0.4)",
                        width=4 if is_campaign else 1,
                    ),
                ),
            ))
        for i in range(n):
            is_campaign = (i == campaign_idx)
            if has_campaign:
                s       = hc[d - 1, i]
                cur_pos = pc[d - 1] if i == campaign_idx else coords_init[i]
                fd.append(go.Scatter(
                    x=[cur_pos[0]], y=[cur_pos[1]],
                    text=[names[i]], textposition="middle center",
                    marker=dict(
                        size=max(s * 110 + 14, 3),
                        line=dict(
                            color="white" if is_campaign else "rgba(0,0,0,0.6)",
                            width=4 if is_campaign else 2,
                        ),
                    ),
                ))
            else:
                fd.append(go.Scatter(
                    x=[coords_init[i, 0]], y=[coords_init[i, 1]],
                    marker=dict(size=3),
                ))
        fd.append(go.Scatter(
            x=pc[:d, 0].tolist() if has_campaign else [],
            y=pc[:d, 1].tolist() if has_campaign else [],
        ))
        fd.append(go.Bar(y=hb[d - 1]))
        fd.append(go.Bar(y=hc[d - 1] if has_campaign else [0] * n))
        frames.append(go.Frame(data=fd, name=str(d), traces=animated))

    fig.frames = frames

    # ── Axes / layout ──────────────────────────────────────────────────────
    fig.add_shape(
        type="line", xref="x", yref="y",
        x0=0, x1=n_days, y0=0.05, y1=0.05,
        line=dict(color="red", dash="dot", width=1), row=1, col=1,
    )

    if many_parties:
        legend_cfg = dict(
            orientation="h",
            y=1.10, x=0, xanchor="left",
            font=dict(size=11),
            tracegroupgap=2,
            itemwidth=40,
        )
    else:
        legend_cfg = dict(orientation="h", y=1.06, x=0, xanchor="left")

    fig.update_layout(
        height=780, barmode="group", hovermode="closest",
        margin=dict(l=10, r=10, t=120, b=80),
        legend=legend_cfg,
        updatemenus=[dict(
            type="buttons", showactive=False,
            y=1.13, x=1.0, xanchor="right", yanchor="top",
            buttons=[
                dict(label="▶  Play All", method="animate",
                     args=[None, {"frame": {"duration": FRAME_MS, "redraw": True},
                                  "fromcurrent": False, "transition": {"duration": 0}}]),
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

    # Line-chart axes
    fig.update_xaxes(range=[0, n_days], title_text="Day", row=1, col=1)
    fig.update_yaxes(range=[0, min(max_share * 1.15, 1.0)], tickformat=".0%",
                     title_text="Vote Share", row=1, col=1, secondary_y=False)
    fig.update_yaxes(title_text="Daily Spend", showgrid=False,
                     row=1, col=1, secondary_y=True)

    # Political compass axes
    fig.update_xaxes(
        range=[-1.2, 1.2], title_text="← Left / Right →",
        zeroline=True, zerolinecolor="rgba(0,0,0,0.35)", zerolinewidth=2,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        row=2, col=1,
    )
    fig.update_yaxes(
        range=[-1.2, 1.2], title_text="← Lib / Auth →",
        zeroline=True, zerolinecolor="rgba(0,0,0,0.35)", zerolinewidth=2,
        showgrid=True, gridcolor="rgba(0,0,0,0.08)",
        tickvals=[-1, -0.5, 0, 0.5, 1],
        row=2, col=1,
    )

    # Bar-chart axis
    fig.update_yaxes(range=[0, min(max_share * 1.15, 1.0)], tickformat=".0%", row=2, col=2)
    return fig


# ─────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────

for k, v in {
    "history_base":  None, "pos_base":     None,
    "history_camp":  None, "pos_camp":     None,
    "last_spend":    None,
    "custom_pos":    None, "custom_w":     None,
    "rand_n":        None, "rand_ver":     0,
    "last_template": None,
    "sim_names":     None, "sim_colors":   None,
    "sim_coords":    None, "sim_camp_idx": None,
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────

st.sidebar.title("Simulation Controls")
poll_interval = st.sidebar.slider("Poll frequency (days)", 3, 14, 7)

st.sidebar.subheader("Voter Behaviour")
tactical    = st.sidebar.slider("Tactical voting strength",   0.0, 1.0, 0.40,
    help="Scales sub-threshold fleeing AND leader-voter coalition rescue.")
bandwagon   = st.sidebar.slider("Bandwagon effect",           0.0, 1.0, 0.05)
anti_leader = st.sidebar.slider("Anti-leader / rally effect", 0.0, 1.0, 0.10)
loyalty     = st.sidebar.slider("Voter loyalty",              0.0, 1.0, 0.8,
    help="Higher loyalty = less mobility. Inhibits all voter flows.")

st.sidebar.subheader("Party Configuration")
all_templates = [t for t in TEMPLATES.keys() if t not in EXCLUDED_TEMPLATES]
template      = st.sidebar.selectbox("Template", all_templates, index=min(6, len(all_templates) - 1))

# Clear custom edits when template changes
if template != st.session_state.last_template:
    st.session_state.custom_pos   = None
    st.session_state.custom_w     = None
    st.session_state.rand_n       = None
    st.session_state.last_template = template

rng0    = np.random.default_rng(42)
is_real = TEMPLATES.get(template) is not None

if is_real:
    default_pos, names, default_w = get_template(template, None, rng0)
    n_parties = len(names)
else:
    n_parties   = st.sidebar.slider("Number of parties", 3, 15, 8)
    default_pos, names, default_w = get_template(template, n_parties, rng0)

colors = hex_colors(n_parties)

if not is_real:
    rc1, rc2 = st.sidebar.columns(2)
    if rc1.button("🎲 Positions", use_container_width=True):
        st.session_state.custom_pos = np.random.default_rng().uniform(-1.0, 1.0, (n_parties, 2))
        st.session_state.rand_n    = n_parties
        st.session_state.rand_ver += 1
    if rc2.button("🎲 Shares", use_container_width=True):
        st.session_state.custom_w  = np.random.default_rng().dirichlet(np.ones(n_parties)) * 100
        st.session_state.rand_n    = n_parties
        st.session_state.rand_ver += 1

same_n   = st.session_state.rand_n == n_parties
init_pos = (st.session_state.custom_pos
            if (same_n and st.session_state.custom_pos is not None)
            else default_pos)
init_w   = (st.session_state.custom_w
            if (same_n and st.session_state.custom_w is not None)
            else default_w * 100)

df0 = pd.DataFrame({
    "Party":      names,
    "Weight (%)": np.round(init_w, 1),
    "X-Pos":      np.round(init_pos[:, 0], 2),
    "Y-Pos":      np.round(init_pos[:, 1], 2),
})
edf = st.sidebar.data_editor(
    df0, hide_index=True, use_container_width=True,
    key=f"party_editor_{template}_{st.session_state.rand_ver}",
    column_config={
        "X-Pos": st.column_config.NumberColumn(min_value=-1.5, max_value=1.5, step=0.05),
        "Y-Pos": st.column_config.NumberColumn(min_value=-1.5, max_value=1.5, step=0.05),
    },
)
shares_raw = edf["Weight (%)"].values.astype(float)
shares     = shares_raw / shares_raw.sum()
coords     = edf[["X-Pos", "Y-Pos"]].values.astype(float)

st.sidebar.subheader("Campaign Party")
campaign_party = st.sidebar.selectbox("Your party", names, index=9)
campaign_idx   = names.index(campaign_party)

st.sidebar.subheader("Advertising Budget")
total_budget  = st.sidebar.slider("Total budget (units)", 0, 500, 200)
effectiveness = st.sidebar.slider(
    "Ad effectiveness (conversion / unit·day)",
    0.0001, 0.005, 0.001, step=0.0001, format="%.4f",
)
spend_strat = st.sidebar.radio("Spending strategy", SPEND_STRATEGIES, horizontal=True)
daily_spend = get_daily_spend(spend_strat, total_budget)
if total_budget > 0:
    st.sidebar.caption(
        f"Peak daily: **{daily_spend.max():.2f}** | Avg: **{daily_spend.mean():.2f}**"
    )

st.sidebar.subheader("Movement Strategy")
movement_strategy = st.sidebar.selectbox("Strategy", STRATEGY_NAMES, index=0)
st.sidebar.caption(f"*{STRATEGY_DESCRIPTIONS[movement_strategy]}*")


# ─────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────

col_b, col_s, col_e = st.columns([2, 2, 1])
run_base  = col_b.button("▶ Run Baseline",      use_container_width=True)
run_strat = col_s.button("▶ Run with Strategy", use_container_width=True)

prog       = st.empty()
metrics    = st.empty()
chart_area = st.empty()


# ── Helpers ────────────────────────────────────────────────────────────────────

def _run(seed: int, apply_campaign: bool, strat: str | None = None):
    return simulate(
        shares, coords, names, poll_interval,
        tactical, bandwagon, anti_leader, loyalty,
        campaign_idx=campaign_idx,
        daily_spend=daily_spend if apply_campaign else np.zeros(DAYS),
        effectiveness=effectiveness if apply_campaign else 0.0,
        movement_strategy=(
            (strat if strat is not None else movement_strategy)
            if apply_campaign else "None (hold position)"
        ),
        poll_size=POLL_SIZE,
        seed=seed,
    )


def _store_snapshot():
    st.session_state.sim_names    = list(names)
    st.session_state.sim_colors   = list(colors)
    st.session_state.sim_coords   = coords.copy()
    st.session_state.sim_camp_idx = campaign_idx


# ── Run buttons ────────────────────────────────────────────────────────────────

if run_base:
    seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    prog.progress(0, text="Computing baseline…")
    hb, pb = _run(seed, apply_campaign=False)
    _store_snapshot()
    st.session_state.update(
        history_base=hb, pos_base=pb,
        history_camp=None, pos_camp=None,
        last_spend=np.zeros(DAYS),
    )
    prog.empty()

if run_strat:
    seed = int(np.random.SeedSequence().entropy & 0xFFFFFFFF)
    prog.progress(0,  text="Computing baseline…")
    hb, pb = _run(seed, apply_campaign=False)
    prog.progress(50, text=f"Running: {movement_strategy}…")
    hc, pc = _run(seed, apply_campaign=True)
    _store_snapshot()
    st.session_state.update(
        history_base=hb, pos_base=pb,
        history_camp=hc, pos_camp=pc,
        last_spend=daily_spend.copy(),
    )
    prog.empty()


# ── Retrieve stored results ────────────────────────────────────────────────────

hb    = st.session_state.history_base
pb    = st.session_state.pos_base
hc    = st.session_state.history_camp
pc    = st.session_state.pos_camp
spend = (st.session_state.last_spend
         if st.session_state.last_spend is not None
         else np.zeros(DAYS))

s_names    = st.session_state.sim_names    or names
s_colors   = st.session_state.sim_colors   or colors
s_coords   = (st.session_state.sim_coords
              if st.session_state.sim_coords is not None else coords)
s_camp_idx = (st.session_state.sim_camp_idx
              if st.session_state.sim_camp_idx is not None else campaign_idx)


# ── Metrics ────────────────────────────────────────────────────────────────────

if hb is not None and hc is not None:
    cp   = s_camp_idx
    gain = hc[-1, cp] - hb[-1, cp]
    roi  = (gain / total_budget * 100) if total_budget > 0 else 0.0
    dist = (float(np.sum(np.linalg.norm(np.diff(pc, axis=0), axis=1)))
            if pc is not None else 0.0)
    with metrics.container():
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric(f"{s_names[cp]} Baseline",  f"{hb[-1, cp]:.1%}")
        m2.metric(f"{s_names[cp]} Strategy",  f"{hc[-1, cp]:.1%}", delta=f"{gain:+.1%}")
        m3.metric("Peak advantage at day",    str(int(np.argmax(hc[:, cp] - hb[:, cp]))))
        m4.metric("Ad ROI (pp / 100 budget)", f"{roi:.2f} pp")
        m5.metric("Total distance moved",     f"{dist:.2f}")


# ── Main chart ─────────────────────────────────────────────────────────────────

if hb is not None:
    has_camp = hc is not None
    pb_use   = pb if pb is not None else np.tile(s_coords[s_camp_idx], (DAYS, 1))
    pc_use   = pc if has_camp else pb_use
    chart_area.plotly_chart(
        build_figure(
            hb, pb_use,
            hc if has_camp else hb,
            pc_use,
            s_names, s_colors, s_coords,
            spend, s_camp_idx, has_camp,
        ),
        use_container_width=True,
    )


# ── Export ─────────────────────────────────────────────────────────────────────

if hb is not None or hc is not None:
    dfs = []
    if hb is not None:
        df = pd.DataFrame(hb, columns=s_names)
        df.insert(0, "Day", range(len(df)))
        df.insert(0, "Run", "Baseline")
        dfs.append(df)
    if hc is not None:
        df = pd.DataFrame(hc, columns=s_names)
        df.insert(0, "Day", range(len(df)))
        df.insert(0, "Run", "Campaign")
        dfs.append(df)
    csv = pd.concat(dfs, ignore_index=True).to_csv(index=False)
    col_e.download_button("⬇ CSV", csv, "results.csv", "text/csv",
                          use_container_width=True)