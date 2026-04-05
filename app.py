import streamlit as st
import pandas as pd
import numpy as np
import hashlib
import plotly.graph_objects as go
import plotly.express as px
from dotenv import load_dotenv

from gemma_oracle import GemmaMetriplexOracle
from h7_qml_classifier import H7TernaryClassifier

load_dotenv()
oracle = GemmaMetriplexOracle()
classifier = H7TernaryClassifier()

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="H7 Parking Violations",
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

def h7_process(data, epsilon=0.25):
    n   = encode_input(data)
    p   = psi(n)
    t   = ternary(p, epsilon)
    lbl = {1:"constructive", 0:"equilibrium", -1:"destructive"}
    return {"n":n, "state_vector":(n,7-n), "psi_value":round(p,6),
            "ternary_state":t, "binary_fw":BIN_A[n-1],
            "binary_bw":BIN_B[n-1], "label":lbl[t]}

def classify_row(row, epsilon, domain_prompt):
    key = f"{row['Registration State']}::{row['Violation Description']}"
    # Obtenemos densidades métricas del oráculo Gemma 4 (Agnóstico pero guiado por prompt)
    metrics = oracle.get_initial_phase_state(key, domain_prompt=domain_prompt)
    rho = metrics["rho"]
    v = metrics["v"]
    
    # H7 QML Computa el Lagrangiano y evalúa el Atractor Ternario (-1, 0, 1)
    classifier.epsilon = epsilon
    ternary_class = classifier.fit_predict(rho, v, steps=50, dt=0.1)
    # Extracción de valores finales
    final_psi = classifier.history_psi[-1]
    
    n = encode_input(key)  # mantenemos n visualmente congruente con el H7 core logic
    lbl = {1:"constructive", 0:"equilibrium", -1:"destructive"}
    
    return pd.Series({"H7_n":n, "H7_state_vector":f"({n},{7-n})",
                      "H7_psi":final_psi, "H7_ternary":ternary_class,
                      "H7_binary_FW":BIN_A[n-1], "H7_binary_BW":BIN_B[n-1],
                      "H7_label":lbl[ternary_class]})

# ══════════════════════════════════════════════════════════════
#  SIDEBAR — controles
# ══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚛ H7 Controls")
    st.caption("smokApp Quantum & AI Lab")
    st.divider()

    # Selector de origen de Datos
    data_source = st.radio(
        "Origen de Datos (Template)",
        options=["Sintético (Mock Demo)", "NYC Parking Tickets (CSV Local)"],
        index=0
    )

    num_rows = st.select_slider(
        "Tamaño (sintético/muestra)", options=[10_000, 100_000],
        value=10_000
    )
    epsilon = st.slider("Epsilon ε (zona neutra)", 0.05, 0.5, 0.25, 0.05)
    st.divider()
    st.markdown("**Ψₙ = cos(π · φ · n)**")
    psi_ref = pd.DataFrame([
        {"n":n, "Ψₙ":round(psi(n),4), "ternario":ternary(psi(n),epsilon)}
        for n in range(1,7)
    ])
    st.dataframe(psi_ref, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  DATASET
# ══════════════════════════════════════════════════════════════
@st.cache_data
def generate_data(num_rows):
    states      = ["NY","NJ","CA","TX"]
    violations  = ["Double Parking","Expired Meter","No Parking","Fire Hydrant","Bus Stop"]
    vtypes      = ["SUBN","SDN"]
    dates       = pd.date_range("2022-01-01","2022-12-31",freq='D')
    return pd.DataFrame({
        "Registration State"   : np.random.choice(states,     size=num_rows),
        "Violation Description": np.random.choice(violations, size=num_rows),
        "Vehicle Body Type"    : np.random.choice(vtypes,     size=num_rows),
        "Issue Date"           : np.random.choice(dates,      size=num_rows),
        "Ticket Number"        : np.random.randint(1_000_000_000,9_999_999_999,size=num_rows),
    })

@st.cache_data
def load_real_data(path, sample_size):
    """
    PLANTILLA: Carga dataset real (ej. Kaggle NYC Parking Tickets).
    Neutro y reusable para otros contextos.
    """
    try:
        df = pd.read_csv(path).sample(sample_size)
        return df
    except FileNotFoundError:
        st.error(f"Falta el archivo real CSV en {path}. Recurriendo a sintético.")
        return generate_data(sample_size)

import json
@st.cache_data
def load_sample_dataset():
    with open("datasets/parking_tickets_sample.json") as f:
        return json.load(f)

# Usa esto en lugar de datos sintéticos para demostración real
tickets = load_sample_dataset()

# ══════════════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════════════
st.markdown("# ⚛ H7 Civic Violations (Gemma Impact Challenge)")
st.caption(f"Zero-Shot Metriplectic Oracle · φ = {phi:.6f} · ε = {epsilon}")
st.divider()

with st.spinner("Conectando Oráculo y resolviendo atractores..."):
    if data_source == "Sintético (Mock Demo)":
        df = generate_data(num_rows)
        domain_prompt = "Analiza el contexto de esta infracción de tráfico municipal:"
    else:
        # Plantilla: Cambia 'nyc_tickets.csv' por la ruta de tu dataset real del hackathon.
        df = load_real_data("nyc_tickets.csv", num_rows)
        domain_prompt = "Analiza el contexto de este reporte oficial de la ciudad:"

    # Solo analizamos los principales casos para no saturar al Oráculo
    top = (
        df[["Registration State","Violation Description"]]
        .value_counts()
        .groupby("Registration State").head(1)
        .sort_index().reset_index()
        .rename(columns={"count":"Frequency"})
    )
    h7_cols = top.apply(lambda r: classify_row(r, epsilon, domain_prompt), axis=1)
    result  = pd.concat([top, h7_cols], axis=1)

# ══════════════════════════════════════════════════════════════
#  KPI CARDS
# ══════════════════════════════════════════════════════════════
c1, c2, c3, c4 = st.columns(4)
label_counts = result["H7_label"].value_counts()

for col, lbl, icon, css in [
    (c1, "constructive", "▲ +1", "constructive"),
    (c2, "destructive",  "▼ −1", "destructive"),
    (c3, "equilibrium",  "◆  0", "equilibrium"),
    (c4, None,           None,   None),
]:
    with col:
        if lbl:
            cnt = label_counts.get(lbl, 0)
            st.markdown(f"""
            <div class="metric-card {css}">
              <div style="font-size:22px;font-weight:bold">{icon}</div>
              <div style="font-size:28px;font-weight:bold">{cnt}</div>
              <div style="font-size:12px;opacity:.6">{lbl}</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="metric-card">
              <div style="font-size:22px">🎫</div>
              <div style="font-size:28px;font-weight:bold">{num_rows:,}</div>
              <div style="font-size:12px;opacity:.6">tickets procesados</div>
            </div>""", unsafe_allow_html=True)

st.divider()

# ══════════════════════════════════════════════════════════════
#  TABLA PRINCIPAL
# ══════════════════════════════════════════════════════════════
st.markdown("### 📋 Top Infracción por Estado + Clasificación H7")

color_map = {"constructive":"#22c55e", "equilibrium":"#eab308", "destructive":"#ef4444"}

def style_table(df):
    def row_color(row):
        c = color_map.get(row["H7_label"],"#fff")
        return [f"color:{c}" if col=="H7_label" else "" for col in df.columns]
    return df.style.apply(row_color, axis=1).format({"H7_psi":"{:+.6f}"})

st.dataframe(style_table(result), use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════
#  CHARTS
# ══════════════════════════════════════════════════════════════
col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### 📊 Frecuencia de Infracciones")
    viol_counts = (
        df.groupby(["Registration State","Violation Description"])
        .size().reset_index(name="count")
    )
    fig1 = px.bar(
        viol_counts, x="Registration State", y="count",
        color="Violation Description", barmode="group",
        template="plotly_dark",
        color_discrete_sequence=px.colors.qualitative.Bold,
    )
    fig1.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        legend=dict(font=dict(size=10)),
        margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig1, use_container_width=True)

with col_b:
    st.markdown("### ⚛ Estado Ternario H7 (distribución)")
    # Classify every unique state+violation combo
    all_combos = (
        df.groupby(["Registration State","Violation Description"])
        .size().reset_index(name="count")
    )
    h7_all = all_combos.apply(lambda r: classify_row(r, epsilon, domain_prompt), axis=1)
    all_combos = pd.concat([all_combos, h7_all], axis=1)

    dist = all_combos.groupby(["Registration State","H7_label"])["count"].sum().reset_index()
    fig2 = px.bar(
        dist, x="Registration State", y="count", color="H7_label",
        barmode="stack", template="plotly_dark",
        color_discrete_map=color_map,
    )
    fig2.update_layout(
        paper_bgcolor="#161b22", plot_bgcolor="#0d1117",
        margin=dict(t=20,b=20)
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("### 📈 Phase Space Evolution (single ticket)")
selected_ticket = st.selectbox("Pick a ticket to inspect...", 
                                [t["description"] for t in tickets])
ticket = next(t for t in tickets if t["description"] == selected_ticket)

# Ejecutar clasificador + capturar historial
metrics = oracle.get_initial_phase_state(ticket["description"])
rho, v = metrics["rho"], metrics["v"]

# Es necesario instanciar de nuevo o usar el existente
classifier_single = H7TernaryClassifier()
ternary_class = classifier_single.fit_predict(rho, v, steps=50, dt=0.1)

# Graficar L_symp + L_metr + L_total
fig_lag = go.Figure()
fig_lag.add_trace(go.Scatter(
    y=classifier_single.history_symp,
    name="L_symp (Energy)",
    mode="lines",
    line=dict(color="#1D9E75")
))
fig_lag.add_trace(go.Scatter(
    y=classifier_single.history_metr,
    name="L_metr (Entropy)",
    mode="lines",
    line=dict(color="#D85A30")
))
fig_lag.add_trace(go.Scatter(
    y=classifier_single.history_psi,
    name="ψ(t) (State)",
    mode="lines",
    line=dict(color="#7F77DD", width=3)
))
fig_lag.update_layout(
    title=f"Metriplectic Evolution: ρ={rho:.2f}, v={v:.2f} → Class {ternary_class:+d}",
    xaxis_title="Step",
    yaxis_title="Value",
    template="plotly_dark",
    height=400
)
st.plotly_chart(fig_lag, use_container_width=True)

# Explicación natural
st.markdown(f"""
**Why {['Destructive', 'Equilibrium', 'Constructive'][ternary_class+1]}?**

1. Oracle extracted: ρ = {rho:.3f} (severity), v = {v:.3f} (intent)
2. Initial state: ψ₀ = v · O_n(ρ·10) = {classifier_single.history_psi[0]:.3f}
3. Dynamics: Energy pushed toward {'+1' if v > 0 else '-1'}, Entropy pulled toward 0
4. Final: ψ(t=50) = {classifier_single.history_psi[-1]:.3f} → Class {ternary_class:+d}
""")


# ══════════════════════════════════════════════════════════════
#  GAUGE Ψₙ por estado
# ══════════════════════════════════════════════════════════════
st.markdown("### 🔮 Ψₙ por Estado (gauge)")
gcols = st.columns(len(result))
for i, (_, row) in enumerate(result.iterrows()):
    with gcols[i]:
        pv = row["H7_psi"]
        c  = color_map[row["H7_label"]]
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pv,
            number={"font":{"color":c},"valueformat":"+.4f"},
            gauge={
                "axis":{"range":[-1,1],"tickcolor":"#30363d"},
                "bar":{"color":c},
                "bgcolor":"#161b22",
                "bordercolor":"#30363d",
                "steps":[
                    {"range":[-1,-epsilon],"color":"#2b0d0d"},
                    {"range":[-epsilon,epsilon],"color":"#1a1a0d"},
                    {"range":[epsilon,1],"color":"#0d2b1a"},
                ],
                "threshold":{"line":{"color":"#fff","width":2},"value":pv}
            },
            title={"text":f"{row['Registration State']}<br><span style='font-size:11px'>{row['Violation Description']}</span>",
                   "font":{"color":"#8b949e","size":13}},
        ))
        fig.update_layout(
            height=220, margin=dict(t=50,b=10,l=10,r=10),
            paper_bgcolor="#0d1117", font=dict(color="#c9d1d9")
        )
        st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════
#  COLLISION GROUPS
# ══════════════════════════════════════════════════════════════
st.divider()
st.markdown("### 🔁 Collision Groups (n compartidos)")
cg = result.groupby("H7_n").agg(
    States=("Registration State", list),
    Violations=("Violation Description", list),
    Psi=("H7_psi","first"),
    Label=("H7_label","first"),
).reset_index()
cg["Collisions"] = cg["States"].apply(len)
cg = cg[cg["Collisions"] > 1]
if cg.empty:
    st.info("No hay colisiones en este run — todos los estados cayeron en n distintos.")
else:
    st.dataframe(cg, use_container_width=True, hide_index=True)

st.divider()
st.caption("H7 Protocol · Ψₙ = cos(π·φ·n) · smokApp Quantum & AI Lab")
