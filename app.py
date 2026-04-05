import streamlit as st
import pandas as pd
import numpy as np
import json
import hashlib
import os
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from gemma_oracle import GemmaMetriplexOracle
from h7_qml_classifier import H7TernaryClassifier

load_dotenv()

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="H7 Metriplectic QML",
    page_icon="⚛",
    layout="wide",
)

st.markdown("""
<style>
  [data-testid="stAppViewContainer"] { background: #0a0a0f; }
  [data-testid="stSidebar"]          { background: #0d1117; }
  h1,h2,h3,p,label,div              { color: #c9d1d9 !important; }
  .metric-card {
    background:#161b22; border:1px solid #30363d;
    border-radius:10px; padding:16px; text-align:center;
  }
  .constructive { border-color:#22c55e !important; }
  .destructive  { border-color:#ef4444 !important; }
  .equilibrium  { border-color:#eab308 !important; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
#  H7 CORE
# ══════════════════════════════════════════════════════════════
phi    = (1 + np.sqrt(5)) / 2
PI     = np.pi
N_VALS = np.arange(1, 7)
BIN_A  = ['001','010','011','100','101','110']
BIN_B  = ['110','101','100','011','010','001']

def encode_input(data) -> int:
    if isinstance(data, (int, float)):
        raw = int(abs(data)) % 6
    elif isinstance(data, (str, bytes)):
        b   = data.encode() if isinstance(data, str) else data
        raw = int(hashlib.md5(b).hexdigest(), 16) % 6
    elif isinstance(data, (list, np.ndarray)):
        arr = np.asarray(data, dtype=float).flatten()
        raw = int(np.abs(arr).sum()) % 6
    else:
        raw = int(hashlib.md5(repr(data).encode()).hexdigest(), 16) % 6
    return int(N_VALS[raw])

def psi(n):       return float(np.cos(PI * phi * n))
def ternary(v, e=0.25): return 1 if v > e else (-1 if v < -e else 0)

# ══════════════════════════════════════════════════════════════
#  DATA LOADING (Con manejo de errores)
# ══════════════════════════════════════════════════════════════
@st.cache_data
def load_sample_dataset():
    """Carga dataset de tickets de estacionamiento de ejemplo."""
    try:
        # Intenta cargar desde datasets/ primero
        if os.path.exists("datasets/parking_tickets_sample.json"):
            with open("datasets/parking_tickets_sample.json") as f:
                return json.load(f)
    except Exception as e:
        st.warning(f"No se pudo cargar dataset local: {e}")
    
    # Fallback: Dataset hardcoded como respaldo
    return [
        {
            "id": 1,
            "description": "Vehicle double-parked on fire hydrant, blocking emergency access",
            "context": "Rush hour, school zone, driver called 311",
            "expected_class": -1,
            "reasoning": "High danger, no mitigation"
        },
        {
            "id": 2,
            "description": "Expired parking meter by 3 minutes",
            "context": "Driver inside store, returned immediately",
            "expected_class": 0,
            "reasoning": "Trivial administrative"
        },
        {
            "id": 3,
            "description": "Parked in no-parking zone during construction detour",
            "context": "City closed normal street, signage unclear",
            "expected_class": 1,
            "reasoning": "Mitigating circumstances"
        },
    ]

@st.cache_resource
def get_oracle():
    """Inicializa el Oráculo Gemma una sola vez."""
    return GemmaMetriplexOracle()

@st.cache_resource
def get_classifier():
    """Inicializa el Clasificador H7 una sola vez."""
    return H7TernaryClassifier()

# ══════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚛ H7 Metriplectic Control Panel")
    st.caption("Safety & Trust Track · Gemma Impact Challenge")
    st.divider()
    
    epsilon = st.slider("Epsilon ε (neutral zone width)", 0.05, 0.5, 0.25, 0.05)
    num_steps = st.slider("Evolution steps (↑ = more refined)", 20, 100, 50, 10)
    dt = st.slider("Time step dt", 0.01, 0.2, 0.1, 0.01)
    
    st.divider()
    st.markdown("**Ψₙ Reference Table**")
    st.markdown("where Ψₙ = cos(π · φ · n)")
    
    psi_ref = pd.DataFrame([
        {"n": n, "Ψₙ": round(psi(n), 4), "Ternary": ternary(psi(n), epsilon)}
        for n in range(1, 7)
    ])
    st.dataframe(psi_ref, hide_index=True, use_container_width=True)
    
    st.divider()
    st.markdown("**About H7**")
    st.caption(
        "Metriplectic QML framework bridges AI reasoning (Gemma Oracle) "
        "with deterministic physics (Lagrangian evolution) for transparent, "
        "auditable civic governance."
    )

# ══════════════════════════════════════════════════════════════
#  MAIN INTERFACE
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚛ H7 Metriplectic QML: Transparent Civic Governance")
st.markdown(
    "**Zero-Shot Explainable Classification for Parking Violations** · "
    f"φ = {phi:.6f} · ε = {epsilon}"
)
st.divider()

# Load data
tickets = load_sample_dataset()
oracle = get_oracle()
classifier_template = get_classifier()

st.markdown("## 📋 Sample Parking Violations Analysis")
st.caption(f"Processing {len(tickets)} tickets through H7 framework...")

# ══════════════════════════════════════════════════════════════
#  SINGLE TICKET INSPECTOR
# ══════════════════════════════════════════════════════════════
st.markdown("### 🔍 Deep Dive: Single Ticket Inspection")

selected_idx = st.selectbox(
    "Pick a ticket to analyze in detail:",
    range(len(tickets)),
    format_func=lambda i: f"#{tickets[i]['id']}: {tickets[i]['description'][:60]}..."
)

ticket = tickets[selected_idx]

with st.container(border=True):
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"**Ticket #{ticket['id']}**")
        st.markdown(f"**Description:** {ticket['description']}")
        st.markdown(f"**Context:** {ticket['context']}")
        st.markdown(f"**Expected Class:** {ticket['expected_class']}")
    
    with col2:
        st.metric("Expected Outcome", 
                  {-1: "🔴 Destructive", 0: "🟡 Equilibrium", 1: "🟢 Constructive"}[ticket['expected_class']])

# Run classification
st.markdown("**▶ Running H7 Classification...**")

metrics = oracle.get_initial_phase_state(ticket["description"])
rho = metrics["rho"]
v = metrics["v"]

classifier_single = H7TernaryClassifier(epsilon=epsilon)
ternary_class = classifier_single.fit_predict(rho, v, steps=num_steps, dt=dt)

col_metrics, col_evolution = st.columns(2)

with col_metrics:
    st.markdown("### 🎯 Oracle Extraction")
    st.metric("ρ (Density/Severity)", f"{rho:.4f}")
    st.metric("v (Velocity/Intent)", f"{v:+.4f}")
    
    st.markdown("### 📊 Classification Result")
    class_label = {-1: "🔴 Destructive", 0: "🟡 Equilibrium", 1: "🟢 Constructive"}[ternary_class]
    st.metric("Predicted Class", class_label)
    
    match = "✅ MATCH" if ternary_class == ticket["expected_class"] else "❌ MISMATCH"
    st.metric("vs Expected", match)

with col_evolution:
    st.markdown("### 📈 Phase Space Evolution")
    
    fig_lag = go.Figure()
    steps = range(len(classifier_single.history_psi))
    
    fig_lag.add_trace(go.Scatter(
        x=list(steps), y=classifier_single.history_symp,
        name="L_symp (Energy)", mode="lines",
        line=dict(color="#1D9E75", width=2)
    ))
    fig_lag.add_trace(go.Scatter(
        x=list(steps), y=classifier_single.history_metr,
        name="L_metr (Entropy)", mode="lines",
        line=dict(color="#D85A30", width=2)
    ))
    fig_lag.add_trace(go.Scatter(
        x=list(steps), y=classifier_single.history_psi,
        name="ψ(t) (State)", mode="lines+markers",
        line=dict(color="#7F77DD", width=3),
        marker=dict(size=4)
    ))
    
    # Add attractor lines
    fig_lag.add_hline(y=1, line_dash="dash", line_color="green", opacity=0.3, annotation_text="+1 (Constructive)")
    fig_lag.add_hline(y=-1, line_dash="dash", line_color="red", opacity=0.3, annotation_text="-1 (Destructive)")
    fig_lag.add_hline(y=0, line_dash="dash", line_color="orange", opacity=0.3)
    
    fig_lag.update_layout(
        title=f"Evolution: ρ={rho:.2f}, v={v:.2f} → Class {ternary_class:+d}",
        xaxis_title="Step",
        yaxis_title="Value",
        template="plotly_dark",
        height=400,
        hovermode="x unified"
    )
    st.plotly_chart(fig_lag, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  EXPLANATION
# ══════════════════════════════════════════════════════════════
st.markdown("### 📝 Why this classification?")

explanation = f"""
**Step 1: Oracle Extraction (Gemma 4)**
- Your ticket description was analyzed by Gemma 4
- Extracted **ρ = {rho:.4f}** (severity/magnitude of infraction)
- Extracted **v = {v:+.4f}** (intent/moral direction)

**Step 2: Initial Conditions**
- Starting state: ψ₀ = v · O_n(ρ·10) = {classifier_single.history_psi[0]:+.4f}
- Where O_n (Golden Operator) = cos(πn) · cos(πφn) modulates the phase space

**Step 3: Metriplectic Evolution** (50 steps)
- **Symplectic force** (Energy conservation): dψ/dt = {{ψ, H}} pulls the state according to ρ and v
- **Metric force** (Entropy dissipation): dψ/dt = [ψ, S] relaxes toward the nearest attractor (-1, 0, or +1)
- These competing forces create a deterministic trajectory: **ψ(t) → {classifier_single.history_psi[-1]:+.4f}**

**Step 4: Ternary Quantization**
- Final state ψ(t=50) = {classifier_single.history_psi[-1]:+.4f}
- Threshold ε = {epsilon}
- Result: **Class {ternary_class:+d}** = {class_label}

**Why is this transparent?**
Every step is deterministic, verifiable, and rooted in explicit mathematics.
An auditor can trace the entire evolution without referring to hidden neural network weights.
"""

st.markdown(explanation)

# ══════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("## 📊 Batch Analysis: All Tickets")

results = []
for t in tickets:
    metrics_t = oracle.get_initial_phase_state(t["description"])
    clf_t = H7TernaryClassifier(epsilon=epsilon)
    pred = clf_t.fit_predict(metrics_t["rho"], metrics_t["v"], steps=num_steps, dt=dt)
    
    results.append({
        "ID": t["id"],
        "Description": t["description"][:40] + "...",
        "ρ": f"{metrics_t['rho']:.3f}",
        "v": f"{metrics_t['v']:+.3f}",
        "Predicted": {-1: "🔴 Destructive", 0: "🟡 Equilibrium", 1: "🟢 Constructive"}[pred],
        "Expected": {-1: "🔴 Destructive", 0: "🟡 Equilibrium", 1: "🟢 Constructive"}[t["expected_class"]],
        "Match": "✅" if pred == t["expected_class"] else "❌"
    })

results_df = pd.DataFrame(results)
st.dataframe(results_df, use_container_width=True, hide_index=True)

# Summary metrics
st.markdown("### 📈 Summary Statistics")
col1, col2, col3 = st.columns(3)

matches = sum(1 for r in results if "✅" in r["Match"])
total = len(results)
accuracy = matches / total * 100

with col1:
    st.metric("Total Tickets", total)
with col2:
    st.metric("Correct Predictions", f"{matches}/{total}")
with col3:
    st.metric("Accuracy", f"{accuracy:.1f}%")

st.divider()
st.caption(
    "H7 Metriplectic QML · φ = (1+√5)/2 · "
    "Gemma Oracle + Lagrangian Evolution · "
    "Transparent Civic Governance"
)
