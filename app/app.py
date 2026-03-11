import streamlit as st
import torch
import pandas as pd
import numpy as np
import joblib
import torch.nn as nn
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Engine RUL Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0b0f1a;
    color: #e2e8f0;
}

.hero {
    padding: 2.5rem 0 1rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 2rem;
}

.hero h1 {
    font-size: 2.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
    letter-spacing: -1px;
}

.hero p {
    color: #64748b;
    font-size: 1rem;
    margin-top: 0.4rem;
}

.card {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 14px;
    padding: 1.6rem;
    margin-bottom: 1.2rem;
}

.rul-display {
    text-align: center;
    padding: 2.5rem 1rem;
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
    border-radius: 16px;
    border: 1px solid #334155;
}

.rul-label {
    font-size: 0.85rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.5rem;
}

.rul-value {
    font-size: 4rem;
    font-weight: 700;
    color: #22d3ee;
    line-height: 1;
}

.rul-unit {
    font-size: 1rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}

.status-chip {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 1rem;
}

.status-good { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.status-warn { background: #2d1a01; color: #fb923c; border: 1px solid #7c2d12; }
.status-crit { background: #1c0a0a; color: #f87171; border: 1px solid #7f1d1d; }

.stat-box {
    background: #111827;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
}

.stat-label {
    font-size: 0.75rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-top: 0.3rem;
}

[data-testid="stFileUploader"] {
    background: #111827;
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 1.5rem;
}

[data-testid="stFileUploader"]:hover {
    border-color: #22d3ee;
}

section[data-testid="stSidebar"] {
    display: none;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
header { visibility: hidden; }

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.8rem;
    margin-top: 1rem;
}

.info-item {
    background: #0f172a;
    border-radius: 8px;
    padding: 0.75rem 1rem;
    border: 1px solid #1e293b;
}

.info-item .key { font-size: 0.72rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.08em; }
.info-item .val { font-size: 0.92rem; color: #cbd5e1; font-weight: 600; margin-top: 0.2rem; }
</style>
""", unsafe_allow_html=True)


WINDOW_SIZE = 30

feature_cols = [
    'op1', 'op2',
    's2', 's3', 's4', 's7', 's8', 's9',
    's11', 's14', 's15', 's17', 's20', 's21'
]


class RUL_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(1)


@st.cache_resource
def load_model():
    model = RUL_LSTM(14, 144, 1, 0.2)
    model.load_state_dict(torch.load("best_model/best_rul_lstm.pth", map_location="cpu"))
    model.eval()
    scaler = joblib.load("scaler_fd001.pkl")
    return model, scaler


model, scaler = load_model()


# ─── Header ───────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero">
    <h1>✈️ Aircraft Engine RUL Predictor</h1>
    <p>LSTM-based Remaining Useful Life estimation · NASA C-MAPSS FD001 dataset</p>
</div>
""", unsafe_allow_html=True)


# ─── Layout ───────────────────────────────────────────────────────────────────

left_col, right_col = st.columns([1, 2], gap="large")

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**Upload Sensor CSV**")
    st.caption("File must contain the 14 required sensor and operational columns.")

    uploaded_file = st.file_uploader("", type=["csv"], label_visibility="collapsed")

    st.markdown("""
    <div class="info-grid">
        <div class="info-item"><div class="key">Model</div><div class="val">LSTM</div></div>
        <div class="info-item"><div class="key">Window</div><div class="val">30 cycles</div></div>
        <div class="info-item"><div class="key">Features</div><div class="val">14 sensors</div></div>
        <div class="info-item"><div class="key">Dataset</div><div class="val">FD001</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    if not uploaded_file:
        st.markdown("""
        <div style="height:100%;display:flex;align-items:center;justify-content:center;
                    flex-direction:column;color:#334155;padding:4rem 0;text-align:center;">
            <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
            <div style="font-size:1rem;font-weight:600;color:#475569;">No data uploaded yet</div>
            <div style="font-size:0.85rem;color:#334155;margin-top:0.4rem;">
                Upload a CSV file to see the RUL prediction and sensor trends
            </div>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {', '.join(missing)}")
            st.stop()

        df_scaled = df.copy()
        df_scaled[feature_cols] = df_scaled[feature_cols].fillna(0.0)
        df_scaled[feature_cols] = scaler.transform(df_scaled[feature_cols])

        seq = df_scaled[feature_cols].values[-WINDOW_SIZE:]
        seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            rul_value = model(seq).item()

        # health status based on RUL
        if rul_value > 80:
            chip_class, status_text = "status-good", "Healthy"
        elif rul_value > 30:
            chip_class, status_text = "status-warn", "Monitor Closely"
        else:
            chip_class, status_text = "status-crit", "Critical — Schedule Maintenance"

        # RUL display
        st.markdown(f"""
        <div class="rul-display">
            <div class="rul-label">Predicted Remaining Useful Life</div>
            <div class="rul-value">{rul_value:.0f}</div>
            <div class="rul-unit">engine cycles</div>
            <span class="status-chip {chip_class}">{status_text}</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # quick stats row
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Cycles Recorded</div>
                <div class="stat-value">{len(df)}</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">RUL (exact)</div>
                <div class="stat-value">{rul_value:.2f}</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            avg_op1 = df['op1'].mean() if 'op1' in df.columns else 0
            st.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Avg Op Cond 1</div>
                <div class="stat-value">{avg_op1:.3f}</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # sensor trend chart
        st.markdown("**Sensor Trend**")
        selected_sensor = st.selectbox("Choose a sensor to visualize", feature_cols, label_visibility="collapsed")

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=df[selected_sensor],
            mode='lines',
            name=selected_sensor,
            line=dict(color='#22d3ee', width=2),
            fill='tozeroy',
            fillcolor='rgba(34, 211, 238, 0.05)'
        ))
        fig.update_layout(
            template='plotly_dark',
            plot_bgcolor='#111827',
            paper_bgcolor='#111827',
            font=dict(family='Inter', color='#94a3b8', size=12),
            title=dict(
                text=f"{selected_sensor}  —  readings over cycles",
                font=dict(color='#f1f5f9', size=14)
            ),
            xaxis=dict(title="Cycle", color='#475569', gridcolor='#1e293b', showgrid=True),
            yaxis=dict(title="Value", color='#475569', gridcolor='#1e293b', showgrid=True),
            margin=dict(l=20, r=20, t=50, b=20),
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # raw data peek
        with st.expander("View raw data"):
            st.dataframe(df[feature_cols].tail(30), use_container_width=True)