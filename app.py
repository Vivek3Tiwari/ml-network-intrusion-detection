import html as _html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBClassifier

BASE = Path(__file__).parent
GITHUB_URL = "https://github.com/Vivek3Tiwari/ml-network-intrusion-detection"

st.set_page_config(
    page_title="NIDS | ML Network Intrusion Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PALETTE = {"BENIGN": "#34d399", "BruteForce": "#fbbf24", "DoS": "#fb7185"}
SEVERITY = {"BENIGN": ("None", "ok"), "BruteForce": ("High", "warn"), "DoS": ("Critical", "crit")}


def h(s: str) -> str:
    """Strip indentation so Streamlit's markdown never treats HTML as a code block."""
    return "\n".join(line.strip() for line in s.strip().splitlines())


def show(s: str):
    st.markdown(h(s), unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@property --n { syntax: '<integer>'; initial-value: 0; inherits: false; }

:root{
  --bg:#070b14; --card:rgba(15,23,42,.62); --line:rgba(148,163,184,.16);
  --text:#e2e8f0; --muted:#94a3b8; --cyan:#22d3ee; --violet:#a78bfa;
  --ok:#34d399; --warn:#fbbf24; --crit:#fb7185;
}
html, body, .stApp, [class*="css"]{ font-family:'Inter',system-ui,-apple-system,sans-serif; }
.stApp{ background:var(--bg); color:var(--text); }

/* animated aurora background + faint grid */
.stApp::before{
  content:""; position:fixed; inset:-20%; z-index:0; pointer-events:none;
  background:
    radial-gradient(40% 40% at 18% 22%, rgba(34,211,238,.17), transparent 70%),
    radial-gradient(36% 36% at 82% 18%, rgba(167,139,250,.20), transparent 70%),
    radial-gradient(42% 42% at 62% 88%, rgba(251,113,133,.11), transparent 70%);
  animation:drift 22s ease-in-out infinite alternate;
}
.stApp::after{
  content:""; position:fixed; inset:0; z-index:0; pointer-events:none;
  background-image:linear-gradient(rgba(148,163,184,.05) 1px, transparent 1px),
                   linear-gradient(90deg, rgba(148,163,184,.05) 1px, transparent 1px);
  background-size:46px 46px;
  -webkit-mask-image:radial-gradient(ellipse at 50% 0%, #000 10%, transparent 70%);
          mask-image:radial-gradient(ellipse at 50% 0%, #000 10%, transparent 70%);
}
@keyframes drift{ from{transform:translate3d(-2%,-1%,0) scale(1)} to{transform:translate3d(3%,2%,0) scale(1.08)} }
@keyframes rise{ from{opacity:0; transform:translateY(18px)} to{opacity:1; transform:none} }
@keyframes shine{ to{ background-position:300% 0 } }
@keyframes ring{ 0%{ box-shadow:0 0 0 0 rgba(52,211,153,.6) } 100%{ box-shadow:0 0 0 12px rgba(52,211,153,0) } }
@keyframes count{ from{ --n:0 } }
@keyframes flow{ to{ background-position:200% 0 } }

#MainMenu, footer, [data-testid="stDecoration"]{ visibility:hidden; height:0; }
[data-testid="stHeader"]{ background:transparent; }
.block-container{ position:relative; z-index:1; max-width:1240px; padding-top:2.2rem; padding-bottom:4rem; }

/* hero */
.hero{ padding:1.2rem 0 .6rem; }
.badge{ display:inline-flex; align-items:center; gap:10px; padding:7px 14px; border-radius:999px;
  background:rgba(15,23,42,.7); border:1px solid var(--line); color:var(--muted); font-size:.82rem; font-weight:500;
  backdrop-filter:blur(8px); animation:rise .6s both; }
.pulse{ width:9px; height:9px; border-radius:50%; background:var(--ok); animation:ring 1.8s infinite; }
.title{ font-size:clamp(2.3rem,5.2vw,4rem); font-weight:800; line-height:1.04; letter-spacing:-.035em;
  margin:.9rem 0 .9rem; animation:rise .8s .05s both; }
.grad{ background:linear-gradient(90deg,#22d3ee,#a78bfa,#fb7185,#22d3ee); background-size:300% 100%;
  -webkit-background-clip:text; background-clip:text; color:transparent; animation:shine 9s linear infinite; }
.lead{ max-width:760px; color:var(--muted); font-size:1.06rem; line-height:1.7; animation:rise .9s .12s both; }
.chips{ display:flex; flex-wrap:wrap; gap:8px; margin-top:1.1rem; animation:rise 1s .2s both; }
.chip{ padding:5px 12px; border-radius:999px; font-size:.78rem; font-weight:600; color:#cbd5e1;
  background:rgba(148,163,184,.10); border:1px solid var(--line); transition:.25s; }
.chip:hover{ color:#fff; border-color:rgba(34,211,238,.6); transform:translateY(-2px); }

/* cards */
.kpis{ display:grid; grid-template-columns:repeat(auto-fit,minmax(210px,1fr)); gap:14px; margin:1.6rem 0 1.4rem; }
.kpi{ position:relative; overflow:hidden; padding:18px 20px; border-radius:18px; background:var(--card);
  border:1px solid var(--line); backdrop-filter:blur(10px); transition:.3s; animation:rise .7s both; }
.kpi:nth-child(2){ animation-delay:.08s } .kpi:nth-child(3){ animation-delay:.16s } .kpi:nth-child(4){ animation-delay:.24s }
.kpi::before{ content:""; position:absolute; left:0; top:0; right:0; height:2px;
  background:linear-gradient(90deg,var(--cyan),var(--violet),var(--crit)); background-size:200% 100%; animation:flow 6s linear infinite; }
.kpi:hover{ transform:translateY(-5px); border-color:rgba(34,211,238,.45); box-shadow:0 18px 40px -18px rgba(34,211,238,.35); }
.k-label{ color:var(--muted); font-size:.78rem; font-weight:600; letter-spacing:.08em; text-transform:uppercase; }
.k-value{ font-size:2.1rem; font-weight:800; letter-spacing:-.02em; margin:.25rem 0 .1rem; color:#f8fafc; }
.k-sub{ color:var(--muted); font-size:.8rem; }
.count{ counter-reset:n var(--n); animation:count 1.4s cubic-bezier(.2,.7,.2,1) both; }
.count::after{ content:counter(n); }

.glass{ padding:20px 22px; border-radius:18px; background:var(--card); border:1px solid var(--line);
  backdrop-filter:blur(10px); animation:rise .7s both; transition:.3s; }
.glass:hover{ border-color:rgba(167,139,250,.4); }
.glass h4{ margin:.2rem 0 .5rem; font-size:1.02rem; color:#f1f5f9; }
.glass p, .glass li{ color:var(--muted); font-size:.92rem; line-height:1.65; }
.glass ul{ padding-left:1.1rem; margin:.3rem 0 0; }

.section-title{ display:flex; align-items:center; gap:12px; font-size:1.25rem; font-weight:700; margin:2rem 0 .9rem; color:#f8fafc; }
.section-title span{ font-size:.78rem; font-weight:700; padding:4px 9px; border-radius:8px; color:#06101c;
  background:linear-gradient(135deg,var(--cyan),var(--violet)); }
.note{ color:var(--muted); font-size:.85rem; margin:-.4rem 0 .8rem; }

/* tabs */
[data-baseweb="tab-list"]{ gap:6px; background:rgba(15,23,42,.6); border:1px solid var(--line);
  padding:6px; border-radius:14px; width:fit-content; backdrop-filter:blur(10px); }
button[data-baseweb="tab"]{ background:transparent; border-radius:10px; padding:8px 18px; height:auto; color:var(--muted); font-weight:600; }
button[data-baseweb="tab"][aria-selected="true"]{ background:linear-gradient(135deg,rgba(34,211,238,.20),rgba(167,139,250,.24)); color:#fff; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"]{ display:none; }

/* inputs */
[data-testid="stFileUploader"] section{ background:rgba(15,23,42,.55); border:1.5px dashed rgba(34,211,238,.38);
  border-radius:16px; transition:.3s; }
[data-testid="stFileUploader"] section:hover{ border-color:var(--cyan); background:rgba(34,211,238,.06); }
.stButton>button, .stDownloadButton>button{ background:linear-gradient(135deg,#22d3ee,#a78bfa); color:#06101c; border:0;
  border-radius:12px; font-weight:700; padding:.55rem 1.2rem; transition:.25s; }
.stButton>button:hover, .stDownloadButton>button:hover{ transform:translateY(-2px); box-shadow:0 12px 28px -10px rgba(34,211,238,.6); color:#06101c; }
[data-testid="stExpander"]{ border:1px solid var(--line); border-radius:14px; background:rgba(15,23,42,.45); }

/* charts */
[data-testid="stPlotlyChart"]{ background:var(--card); border:1px solid var(--line); border-radius:18px;
  padding:8px; backdrop-filter:blur(8px); animation:rise .8s both; }

/* alert table */
.tbl{ overflow-x:auto; }
.tbl table{ width:100%; border-collapse:separate; border-spacing:0 8px; }
.tbl th{ text-align:left; color:var(--muted); font-size:.72rem; letter-spacing:.09em; text-transform:uppercase; font-weight:700; padding:0 14px 2px; white-space:nowrap; }
.tbl td{ padding:12px 14px; background:var(--card); border-top:1px solid var(--line); border-bottom:1px solid var(--line);
  font-size:.88rem; color:#e2e8f0; white-space:nowrap; }
.tbl td:first-child{ border-left:1px solid var(--line); border-radius:12px 0 0 12px; color:var(--muted); }
.tbl td:last-child{ border-right:1px solid var(--line); border-radius:0 12px 12px 0; }
.tbl tr.row{ animation:rise .5s both; }
.tbl tr.row:hover td{ background:rgba(34,211,238,.07); }
.tag{ display:inline-block; padding:3px 10px; border-radius:999px; font-size:.76rem; font-weight:700; }
.tag.ok{ color:#34d399; background:rgba(52,211,153,.14); } .tag.warn{ color:#fbbf24; background:rgba(251,191,36,.14); }
.tag.crit{ color:#fb7185; background:rgba(251,113,133,.15); }
.bar{ display:inline-block; width:110px; height:7px; border-radius:99px; background:rgba(148,163,184,.18); vertical-align:middle; overflow:hidden; margin-right:8px; }
.bar i{ display:block; height:100%; border-radius:99px; background:linear-gradient(90deg,var(--cyan),var(--violet)); }
.mono{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; color:#cbd5e1; }

/* pipeline */
.steps{ display:grid; grid-template-columns:repeat(auto-fit,minmax(190px,1fr)); gap:14px; }
.step{ position:relative; padding:18px; border-radius:18px; background:var(--card); border:1px solid var(--line); animation:rise .7s both; transition:.3s; }
.step:hover{ transform:translateY(-5px); border-color:rgba(34,211,238,.5); }
.step .num{ width:34px; height:34px; border-radius:11px; display:grid; place-items:center; font-weight:800; color:#06101c;
  background:linear-gradient(135deg,var(--cyan),var(--violet)); margin-bottom:10px; }
.step h4{ margin:.1rem 0 .3rem; font-size:1rem; color:#f1f5f9; }
.step p{ margin:0; font-size:.86rem; color:var(--muted); line-height:1.55; }

.classcard{ border-left:3px solid var(--c); }
.footer{ text-align:center; color:var(--muted); font-size:.82rem; margin-top:3rem; padding-top:1.2rem; border-top:1px solid var(--line); }
.footer a{ color:var(--cyan); text-decoration:none; }
.empty{ text-align:center; padding:48px 20px; color:var(--muted); }
.empty b{ color:#f1f5f9; font-size:1.1rem; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


# ----------------------------------------------------------------------------
# Data + model
# ----------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = XGBClassifier()
    model.load_model(str(BASE / "xgb_model.json"))
    feats = json.loads((BASE / "features.json").read_text())
    classes = json.loads((BASE / "classes.json").read_text())
    return model, feats, classes


@st.cache_data(show_spinner=False)
def load_sample():
    return pd.read_csv(BASE / "sample_flows.csv")


def run_detection(df, model, feats, classes):
    X = df[feats].replace([np.inf, -np.inf], np.nan).fillna(0).astype("float32")
    proba = model.predict_proba(X)
    out = df.copy()
    out[feats] = X  # cleaned values (no inf/NaN) for charts and tables
    out["Prediction"] = np.array(classes)[proba.argmax(1)]
    out["Confidence"] = proba.max(1) * 100
    benign_i = classes.index("BENIGN") if "BENIGN" in classes else 0
    out["Threat score"] = (1 - proba[:, benign_i]) * 100
    return out


# ----------------------------------------------------------------------------
# Charts
# ----------------------------------------------------------------------------
def style(fig, height=360, title=None):
    fig.update_layout(
        height=height,
        title=dict(text=title, font=dict(size=15, color="#f1f5f9"), x=0.02) if title else None,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color="#cbd5e1"),
        margin=dict(l=14, r=14, t=50 if title else 14, b=14),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0f172a", font_color="#e2e8f0", bordercolor="#334155"),
    )
    fig.update_xaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.2)")
    fig.update_yaxes(gridcolor="rgba(148,163,184,.10)", zerolinecolor="rgba(148,163,184,.2)")
    return fig


def donut_chart(counts):
    total = int(counts.sum())
    fig = go.Figure(go.Pie(
        labels=list(counts.index), values=list(counts.values), hole=0.7, sort=False,
        marker=dict(colors=[PALETTE.get(k, "#94a3b8") for k in counts.index], line=dict(color="#070b14", width=4)),
        textinfo="percent", pull=[0.03] * len(counts),
        hovertemplate="%{label}: %{value:,} flows<extra></extra>",
    ))
    fig.add_annotation(text=f"<b>{total:,}</b><br><span style='font-size:12px;color:#94a3b8'>flows</span>",
                       showarrow=False, font=dict(size=26, color="#f8fafc"))
    return style(fig, 340, "Traffic composition")


def gauge_chart(pct):
    level = "Low" if pct < 20 else "Elevated" if pct < 50 else "High"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", valueformat=".1f", font=dict(size=42, color="#f8fafc")),
        title=dict(text=f"Malicious share · <b>{level}</b>", font=dict(size=14, color="#94a3b8")),
        gauge=dict(
            axis=dict(range=[0, 100], tickcolor="#64748b", tickfont=dict(color="#64748b")),
            bar=dict(color="#22d3ee", thickness=0.28), bgcolor="rgba(0,0,0,0)", borderwidth=0,
            steps=[dict(range=[0, 20], color="rgba(52,211,153,.22)"),
                   dict(range=[20, 50], color="rgba(251,191,36,.22)"),
                   dict(range=[50, 100], color="rgba(251,113,133,.26)")],
        ),
    ))
    return style(fig, 340, "Threat level")


def timeline_chart(out, max_points=2500):
    d = out.reset_index(drop=True)
    d = d.iloc[np.linspace(0, len(d) - 1, min(len(d), max_points)).astype(int)] if len(d) > max_points else d
    fig = go.Figure()
    for cls in d["Prediction"].unique():
        s = d[d["Prediction"] == cls]
        fig.add_trace(go.Scatter(
            x=s.index, y=s["Threat score"], mode="markers", name=cls,
            marker=dict(color=PALETTE.get(cls, "#94a3b8"), size=7, opacity=0.8, line=dict(width=0)),
            customdata=s["Confidence"],
            hovertemplate="Flow #%{x}<br>Threat score %{y:.1f}<br>Confidence %{customdata:.2f}%<extra>" + cls + "</extra>",
        ))
    fig.add_hline(y=50, line=dict(color="rgba(148,163,184,.5)", dash="dot"),
                  annotation_text="alert threshold", annotation_font_color="#94a3b8")
    fig.update_yaxes(title="Threat score (%)", range=[-4, 104])
    fig.update_xaxes(title="Flow index")
    return style(fig, 360, "Threat timeline")


def fingerprint_chart(out, feats, top=10):
    g = out.groupby("Prediction")[feats].mean()
    if len(g) < 2:
        return None
    spread = g.std() / (g.abs().mean() + 1e-9)
    cols = list(spread.sort_values(ascending=False).head(top).index)
    sub = g[cols]
    z = (sub - sub.mean()) / (sub.std(ddof=0) + 1e-9)
    fig = go.Figure(go.Heatmap(
        z=z.T.values, x=list(z.index), y=cols, customdata=sub.T.values, showscale=False, xgap=4, ygap=4,
        colorscale=[[0, "#0b1020"], [0.5, "#5b21b6"], [1, "#22d3ee"]],
        hovertemplate="%{y}<br>%{x}: mean %{customdata:,.2f}<extra></extra>",
    ))
    fig.update_yaxes(automargin=True, autorange="reversed")
    return style(fig, 380, "Class fingerprint · what sets each class apart")


def confusion_chart(matrix, labels, title, height=360):
    m = np.array(matrix, dtype=float)
    norm = m / np.maximum(m.sum(axis=1, keepdims=True), 1)
    fig = go.Figure(go.Heatmap(
        z=norm, x=labels, y=labels, text=m.astype(int), texttemplate="%{text:,}", showscale=False, xgap=5, ygap=5,
        colorscale=[[0, "#0b1020"], [0.02, "#1e1b4b"], [1, "#22d3ee"]],
        textfont=dict(size=15, color="#f8fafc"),
        hovertemplate="Actual %{y} → predicted %{x}: %{text:,}<extra></extra>",
    ))
    fig.update_xaxes(title="Predicted", side="bottom")
    fig.update_yaxes(title="Actual", autorange="reversed")
    return style(fig, height, title)


# ----------------------------------------------------------------------------
# HTML builders
# ----------------------------------------------------------------------------
def kpi(label, value, sub="", count=False):
    if count:
        val = f'<div class="k-value count" style="--n:{int(value)}"></div>'
    else:
        val = f'<div class="k-value">{_html.escape(str(value))}</div>'
    return f'<div class="kpi"><div class="k-label">{label}</div>{val}<div class="k-sub">{sub}</div></div>'


def fmt_num(v):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return _html.escape(str(v))
    return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:,.2f}"


def alerts_table(alerts, show_feats, has_actual, limit=25):
    head = ["#", "Prediction", "Severity", "Confidence", "Threat score"] + [_html.escape(c) for c in show_feats]
    if has_actual:
        head.append("Ground truth")
    rows = []
    for i, (_, r) in enumerate(alerts.head(limit).iterrows()):
        sev_txt, sev_cls = SEVERITY.get(r["Prediction"], ("-", "ok"))
        cls_color = PALETTE.get(r["Prediction"], "#94a3b8")
        cells = [
            f'<td>{i + 1}</td>',
            f'<td><span class="tag {sev_cls}" style="color:{cls_color}">{_html.escape(str(r["Prediction"]))}</span></td>',
            f'<td><span class="tag {sev_cls}">{sev_txt}</span></td>',
            f'<td><span class="bar"><i style="width:{r["Confidence"]:.1f}%"></i></span>{r["Confidence"]:.2f}%</td>',
            f'<td class="mono">{r["Threat score"]:.1f}</td>',
        ]
        cells += [f'<td class="mono">{fmt_num(r[c])}</td>' for c in show_feats]
        if has_actual:
            ok = str(r["Actual"]) == str(r["Prediction"])
            mark = "✓" if ok else "✗"
            color = "#34d399" if ok else "#fb7185"
            cells.append(f'<td><span style="color:{color};font-weight:700">{mark}</span> {_html.escape(str(r["Actual"]))}</td>')
        rows.append(f'<tr class="row" style="animation-delay:{i * 35}ms">{"".join(cells)}</tr>')
    th = "".join(f"<th>{c}</th>" for c in head)
    return f'<div class="tbl"><table><thead><tr>{th}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
show("""
<div class="hero">
  <div class="badge"><span class="pulse"></span>Model online &nbsp;·&nbsp; XGBoost &nbsp;·&nbsp; CICIDS2017</div>
  <div class="title">Network Intrusion<br><span class="grad">Detection System</span></div>
  <p class="lead">A machine-learning detector that reads network flow statistics and separates normal traffic
  from <b style="color:#fb7185">DoS</b> and <b style="color:#fbbf24">brute-force</b> attacks in milliseconds.
  Upload your own flows or explore the built-in sample, then see where the model is strong and where it is not.</p>
  <div class="chips">
    <span class="chip">Python</span><span class="chip">XGBoost</span><span class="chip">Random Forest</span>
    <span class="chip">Scikit-learn</span><span class="chip">Pandas</span><span class="chip">Plotly</span><span class="chip">Streamlit</span>
  </div>
</div>
""")

show("""
<div class="kpis">
  """ + kpi("Accuracy", "99.98%", "XGBoost · random split")
         + kpi("Macro-F1", "0.9995", "3 traffic classes")
         + kpi("False-positive rate", "0.019%", "49 of 264,720 benign flows")
         + kpi("Training flows", "1.22M", "78 flow features each") + """
</div>
""")

tab_detect, tab_insight, tab_how, tab_about = st.tabs(
    ["⚡  Live Detection", "📊  Model Insights", "🧭  How It Works", "ℹ️  About"]
)

try:
    model, feats, classes = load_artifacts()
except Exception as e:  # noqa: BLE001
    st.error(f"Could not load model files: {e}")
    st.stop()

# ----------------------------------------------------------------------------
# TAB 1 · Live detection
# ----------------------------------------------------------------------------
with tab_detect:
    show('<div class="section-title"><span>01</span> Choose traffic to analyse</div>')
    c1, c2 = st.columns([3, 2])
    with c1:
        up = st.file_uploader("Upload a CSV of flow features", type="csv", label_visibility="collapsed")
    with c2:
        use_sample = st.toggle("Use built-in sample (300 labelled flows)", value=True)
        st.caption("An uploaded file always takes priority over the sample.")

    with st.expander("Expected input format"):
        st.write(
            "One row per network flow, with the same columns CICFlowMeter produces for CICIDS2017 "
            f"({len(feats)} numeric features are used). Column names are matched after trimming spaces. "
            "Extra columns such as `Label` are ignored."
        )
        st.code(", ".join(feats[:8]) + ", ...", language="text")

    df, source = None, ""
    if up is not None:
        try:
            df = pd.read_csv(up, encoding="latin1", low_memory=False)
            source = up.name
        except Exception as e:  # noqa: BLE001
            st.error(f"Could not read that file: {e}")
    elif use_sample:
        df, source = load_sample().copy(), "built-in sample"

    if df is None:
        show("""
        <div class="glass empty"><b>No traffic loaded yet</b><br>
        Upload a CSV above, or switch on the built-in sample to see the detector in action.</div>
        """)
    else:
        df.columns = df.columns.str.strip()
        missing = [c for c in feats if c not in df.columns]
        if missing:
            st.error(f"{len(missing)} required columns are missing, for example: {', '.join(missing[:5])}")
        else:
            if len(df) > 500_000:
                df = df.sample(500_000, random_state=1).reset_index(drop=True)
                st.warning("Large file: analysing a random 500,000-flow sample.")
            with st.spinner("Scanning flows..."):
                out = run_detection(df, model, feats, classes)

            n = len(out)
            alerts = out[out["Prediction"] != "BENIGN"].sort_values("Confidence", ascending=False)
            n_alert = len(alerts)
            has_actual = "Actual" in out.columns
            fourth = (kpi("Label agreement", f"{(out['Actual'].astype(str) == out['Prediction']).mean() * 100:.1f}%",
                          "prediction vs ground truth")
                      if has_actual else kpi("Mean confidence", f"{out['Confidence'].mean():.1f}%", "across all flows"))

            show('<div class="section-title"><span>02</span> Detection results</div>')
            st.markdown(f'<p class="note">Source: <b>{_html.escape(source)}</b></p>', unsafe_allow_html=True)
            show('<div class="kpis">'
                 + kpi("Flows analysed", n, "rows scored", count=True)
                 + kpi("Alerts raised", n_alert, "flagged as attacks", count=True)
                 + kpi("Benign flows", n - n_alert, "normal traffic", count=True)
                 + fourth + "</div>")

            g1, g2 = st.columns(2)
            with g1:
                st.plotly_chart(gauge_chart(n_alert / n * 100), config={"displayModeBar": False})
            with g2:
                st.plotly_chart(donut_chart(out["Prediction"].value_counts()), config={"displayModeBar": False})

            st.plotly_chart(timeline_chart(out), config={"displayModeBar": False})

            f1, f2 = st.columns(2)
            fp = fingerprint_chart(out, feats)
            with f1:
                if fp is not None:
                    st.plotly_chart(fp, config={"displayModeBar": False})
                else:
                    show('<div class="glass empty">Class fingerprint needs at least two traffic types in the data.</div>')
            with f2:
                if has_actual:
                    labs = sorted(set(out["Actual"].astype(str)) | set(out["Prediction"]))
                    cm = pd.crosstab(out["Actual"].astype(str), out["Prediction"]).reindex(index=labs, columns=labs, fill_value=0)
                    st.plotly_chart(confusion_chart(cm.values, labs, "Sample confusion matrix", 380),
                                    config={"displayModeBar": False})
                else:
                    show('<div class="glass empty">Add an <b>Actual</b> column with true labels to see a confusion matrix here.</div>')

            show('<div class="section-title"><span>03</span> Top alerts</div>')
            if n_alert == 0:
                show('<div class="glass empty"><b>All clear</b><br>No malicious flows were detected in this traffic.</div>')
            else:
                st.markdown(f'<p class="note">Showing the {min(25, n_alert)} highest-confidence alerts out of {n_alert:,}.</p>',
                            unsafe_allow_html=True)
                st.markdown(alerts_table(alerts, feats[:3], has_actual), unsafe_allow_html=True)

            with st.expander("Browse all results"):
                cols = ["Prediction", "Confidence", "Threat score"] + (["Actual"] if has_actual else []) + feats[:6]
                st.dataframe(
                    out[cols],
                    column_config={
                        "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0, max_value=100, format="%.1f%%"),
                        "Threat score": st.column_config.ProgressColumn("Threat score", min_value=0, max_value=100, format="%.1f%%"),
                    },
                )
            st.download_button("⬇  Download full results (CSV)", out.to_csv(index=False), "nids_results.csv", "text/csv")

# ----------------------------------------------------------------------------
# TAB 2 · Model insights
# ----------------------------------------------------------------------------
with tab_insight:
    show('<div class="section-title"><span>01</span> How good is it, honestly?</div>')
    st.markdown('<p class="note">Random-split scores on CICIDS2017 are very high because attacks arrive in long bursts of near-identical flows. '
                'The time-based test trains on the first 70% of each class and tests on the last 30%, which is a tougher and more realistic check.</p>',
                unsafe_allow_html=True)

    names = ["Random Forest<br>random split", "XGBoost<br>random split", "Random Forest<br>time-based"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=names, y=[99.95, 99.98, 98.16], marker_color="#22d3ee",
                         text=["99.95%", "99.98%", "98.16%"], textposition="outside"))
    fig.add_trace(go.Bar(name="Macro-F1", x=names, y=[99.92, 99.95, 96.72], marker_color="#a78bfa",
                         text=["99.92%", "99.95%", "96.72%"], textposition="outside"))
    fig.update_layout(barmode="group", bargap=0.35)
    fig.update_yaxes(range=[94, 100.6], title="% (axis starts at 94)")
    a, b = st.columns(2)
    with a:
        st.plotly_chart(style(fig, 380, "Model comparison"), config={"displayModeBar": False})
    with b:
        fig2 = go.Figure()
        cls_names = ["BENIGN", "BruteForce", "DoS"]
        fig2.add_trace(go.Bar(name="Random split", x=cls_names, y=[99.98, 99.95, 99.74], marker_color="#34d399",
                              text=["99.98%", "99.95%", "99.74%"], textposition="outside"))
        fig2.add_trace(go.Bar(name="Time-based", x=cls_names, y=[99.98, 97.89, 85.79], marker_color="#fb7185",
                              text=["99.98%", "97.89%", "85.79%"], textposition="outside"))
        fig2.update_layout(barmode="group", bargap=0.35)
        fig2.update_yaxes(range=[80, 101.5], title="Recall % (axis starts at 80)")
        st.plotly_chart(style(fig2, 380, "Attack recall by class (Random Forest)"), config={"displayModeBar": False})

    c, d = st.columns(2)
    with c:
        st.plotly_chart(
            confusion_chart([[264671, 0, 49], [1, 1829, 0], [99, 0, 38650]], ["BENIGN", "BruteForce", "DoS"],
                            "Confusion matrix · Random Forest, 305,299 test flows", 380),
            config={"displayModeBar": False})
    with d:
        show("""
        <div class="glass"><h4>🔎 What the numbers say</h4>
        <ul>
          <li><b>Very few false alarms:</b> only 49 of 264,720 normal flows (0.019%) were wrongly flagged.</li>
          <li><b>Attacks are rarely confused with each other:</b> mistakes happen between normal traffic and attacks.</li>
          <li><b>The honest drop:</b> on later traffic, DoS recall falls to 85.8%. Newer DoS variants look more like normal flows.</li>
          <li><b>Next improvements:</b> train on more days (PortScan, web attacks, botnet) and add drift monitoring.</li>
        </ul></div>
        """)

# ----------------------------------------------------------------------------
# TAB 3 · How it works
# ----------------------------------------------------------------------------
with tab_how:
    show('<div class="section-title"><span>01</span> The pipeline</div>')
    show("""
    <div class="steps">
      <div class="step"><div class="num">1</div><h4>Ingest</h4><p>CICIDS2017 flow records (Mon-Wed), 78 numeric features per flow, about 1.67M rows.</p></div>
      <div class="step" style="animation-delay:.08s"><div class="num">2</div><h4>Clean</h4><p>Trim column names, drop inf/NaN and duplicate rows, group 7 attack labels into 3 classes. 1.53M rows remain.</p></div>
      <div class="step" style="animation-delay:.16s"><div class="num">3</div><h4>Train</h4><p>Random Forest and XGBoost on a stratified 80/20 split, with class weights for the rare brute-force class.</p></div>
      <div class="step" style="animation-delay:.24s"><div class="num">4</div><h4>Validate</h4><p>Precision, recall, macro-F1, false-positive rate, plus a time-based test for realistic generalisation.</p></div>
      <div class="step" style="animation-delay:.32s"><div class="num">5</div><h4>Serve</h4><p>XGBoost model (0.7 MB) behind this Streamlit app: upload flows, get predictions, confidence and alerts.</p></div>
    </div>
    """)

    show('<div class="section-title"><span>02</span> Traffic classes</div>')
    show("""
    <div class="steps">
      <div class="glass classcard" style="--c:#34d399"><h4 style="color:#34d399">BENIGN</h4><p>Normal everyday traffic. 1,323,598 flows after cleaning, about 87% of the data.</p></div>
      <div class="glass classcard" style="--c:#fb7185;animation-delay:.08s"><h4 style="color:#fb7185">DoS · Critical</h4><p>Hulk, GoldenEye, slowloris and Slowhttptest attacks that exhaust a server. 193,745 flows.</p></div>
      <div class="glass classcard" style="--c:#fbbf24;animation-delay:.16s"><h4 style="color:#fbbf24">BruteForce · High</h4><p>FTP-Patator and SSH-Patator password guessing. Only 9,150 flows, so the hardest class to learn.</p></div>
    </div>
    """)

    show('<div class="section-title"><span>03</span> Why flows and not packets?</div>')
    show("""
    <div class="glass"><p>A flow summarises a whole conversation between two hosts: durations, packet counts, byte rates, inter-arrival times, flag counts.
    Attacks leave statistical fingerprints in these numbers. A brute-force tool sends many short, similar connections, while a DoS tool holds connections open or
    floods requests. The <b>Class fingerprint</b> heatmap on the detection tab shows exactly which features differ most between the classes in the data you analyse.</p></div>
    """)

# ----------------------------------------------------------------------------
# TAB 4 · About
# ----------------------------------------------------------------------------
with tab_about:
    show('<div class="section-title"><span>01</span> About this project</div>')
    a, b = st.columns(2)
    with a:
        show("""
        <div class="glass"><h4>🎯 Goal</h4>
        <p>Show an end-to-end security ML workflow: data cleaning, imbalance handling, model comparison, honest validation and a deployed demo
        that a non-technical reviewer can try in seconds.</p></div>
        """)
        show("""
        <div class="glass" style="margin-top:14px"><h4>📦 Dataset</h4>
        <p>CICIDS2017 from the Canadian Institute for Cybersecurity (UNB), Monday to Wednesday capture days.
        Heartbleed (11 rows) was dropped as too small to learn from. The dataset is not stored in this app.</p></div>
        """)
    with b:
        show("""
        <div class="glass"><h4>⚠️ Limitations</h4>
        <ul>
          <li>Only three traffic classes. PortScan, web attacks and botnet traffic are not covered yet.</li>
          <li>Random-split scores are optimistic because of burst-style attacks in the data.</li>
          <li>The app scores pre-extracted flow features. It does not capture live packets.</li>
          <li>Research dataset from 2017, so modern traffic may behave differently.</li>
        </ul></div>
        """)
        show("""
        <div class="glass" style="margin-top:14px"><h4>🛡️ Ethics</h4>
        <p>Built and tested only on a public research dataset. Do not upload traffic captured from networks you do not own or have permission to analyse.</p></div>
        """)

    show(f"""
    <div class="glass" style="margin-top:14px"><h4>🔗 Source code</h4>
    <p>Notebook, metrics and README live on GitHub: <a href="{GITHUB_URL}" target="_blank" style="color:#22d3ee">{GITHUB_URL.replace("https://", "")}</a></p></div>
    """)

show(f"""
<div class="footer">Built by <b>Vivek Tiwari</b> · B.Tech CSE (Cyber Security) ·
<a href="{GITHUB_URL}" target="_blank">GitHub</a></div>
""")
