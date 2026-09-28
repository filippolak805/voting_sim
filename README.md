# Campaign Strategy Simulator

## Setup

Python 3.11+ required. All files must be in the same directory, including `templates.csv`.

```bash
python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

---

## Interactive simulator

```bash
streamlit run campaign_simulator.py
```

Opens at `http://localhost:8501`. Use the sidebar to configure a scenario, then click one of the two run buttons in the main area.

**Sidebar controls:**

- **Template** — choose a country or a generic party system structure
- **Party table** — edit starting positions (X/Y on the political compass) and vote shares directly in the table
- **Campaign Party** — the party you're running the campaign for
- **Voter behaviour sliders** — tactical voting, bandwagon, anti-leader effect, loyalty
- **Budget, effectiveness & spending strategy** — total budget, ad conversion rate per unit per day, and whether to spend uniformly, front-loaded, or back-loaded
- **Movement strategy** — where on the compass your party moves during the campaign

**Running a scenario:**

1. Click **▶ Run Baseline** to see the no-campaign trajectory.
2. Click **▶ Run with Strategy** to overlay the campaign result.
3. Use **▶ Play All** or the day slider under the chart to animate.
4. Click **⬇ CSV** to export both trajectories.

**Chart panels:**

- **Support over time** — solid = baseline, dashed = campaign. Red dotted line = 5% threshold. Yellow shading = daily ad spend.
- **Political compass** — bubble size = vote share; faded = baseline, solid = campaign; dotted trail = your party's path.
- **Final vote shares** — side-by-side bar chart of the two runs.

The five metric cards above the chart show baseline share, campaign share (with delta), day of peak advantage, ad ROI, and total compass distance moved.

---

## Evaluation notebook

```bash
code evaluation.ipynb
```

You can edit the constants at the top of **Section 1** to change the scenario, then run **Kernel → Restart & Run All**:
