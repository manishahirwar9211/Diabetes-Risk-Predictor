import streamlit as st
import joblib
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Load model ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

model = load_model()

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500;600&display=swap');

:root {
    --bg:        #0d1117;
    --surface:   #161b22;
    --border:    #30363d;
    --accent:    #e8a838;
    --accent2:   #58a6ff;
    --danger:    #f85149;
    --safe:      #3fb950;
    --muted:     #8b949e;
    --text:      #e6edf3;
}

html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] > .main {
    background: var(--bg);
}

/* Hide default header */
header[data-testid="stHeader"] { display: none !important; }
[data-testid="stToolbar"] { display: none !important; }

/* Typography */
h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
    color: var(--text) !important;
}

/* Cards */
.card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 16px;
}

.hero-card {
    background: linear-gradient(135deg, #161b22 0%, #0d1117 60%, #1a1a2e 100%);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 48px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-card::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(232,168,56,0.15) 0%, transparent 70%);
    border-radius: 50%;
}

.hero-tag {
    display: inline-block;
    background: rgba(232,168,56,0.15);
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 4px 12px;
    border-radius: 20px;
    border: 1px solid rgba(232,168,56,0.3);
    margin-bottom: 16px;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 42px;
    line-height: 1.15;
    color: var(--text) !important;
    margin: 8px 0;
}

.hero-title em {
    color: var(--accent);
    font-style: italic;
}

.hero-sub {
    color: var(--muted);
    font-size: 15px;
    margin-top: 10px;
    font-weight: 300;
}

.section-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
}

/* Sliders */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--accent) !important;
}

[data-testid="stSlider"] label {
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #0d1117 !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}

[data-testid="stNumberInput"] label {
    color: var(--text) !important;
    font-weight: 500 !important;
    font-size: 14px !important;
}

/* Button */
[data-testid="stButton"] button {
    width: 100%;
    background: var(--accent) !important;
    color: #0d1117 !important;
    border: none !important;
    padding: 14px 28px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 16px !important;
    border-radius: 10px !important;
    cursor: pointer;
    transition: all 0.2s;
    letter-spacing: 0.5px;
}

[data-testid="stButton"] button:hover {
    background: #f0b84a !important;
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(232,168,56,0.35) !important;
}

/* Results */
.result-positive {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.4);
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
}

.result-negative {
    background: rgba(63,185,80,0.08);
    border: 1px solid rgba(63,185,80,0.4);
    border-radius: 12px;
    padding: 28px 32px;
    text-align: center;
}

.result-emoji {
    font-size: 52px;
    margin-bottom: 8px;
}

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 26px;
    margin: 6px 0;
}

.result-prob {
    font-size: 13px;
    margin-top: 8px;
    color: var(--muted);
}

.prob-bar-wrap {
    background: var(--border);
    border-radius: 100px;
    height: 8px;
    margin: 12px 0 6px;
    overflow: hidden;
}

.prob-bar-fill-pos {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #f85149, #ff7b72);
    transition: width 0.6s ease;
}

.prob-bar-fill-neg {
    height: 100%;
    border-radius: 100px;
    background: linear-gradient(90deg, #3fb950, #56d364);
    transition: width 0.6s ease;
}

.feature-pill {
    display: inline-block;
    background: rgba(88,166,255,0.1);
    border: 1px solid rgba(88,166,255,0.25);
    color: var(--accent2);
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 12px;
    font-weight: 500;
    margin: 3px;
}

.info-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin-top: 8px;
}

.info-item {
    background: #0d1117;
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 12px 14px;
}

.info-item .val {
    font-family: 'DM Serif Display', serif;
    font-size: 22px;
    color: var(--accent);
}

.info-item .lbl {
    font-size: 11px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* Divider */
hr { border-color: var(--border) !important; }

/* Remove Streamlit padding */
.block-container { padding-top: 2rem !important; }

/* Metric override */
[data-testid="metric-container"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px !important;
}
</style>
""", unsafe_allow_html=True)

# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-card">
    <div class="hero-tag">🧬 ML Health Tool</div>
    <div class="hero-title">Diabetes <em>Risk</em> Predictor</div>
    <div class="hero-sub">
        Powered by a Gaussian Naïve Bayes classifier trained on the Pima Indians Diabetes Dataset.
        Enter patient metrics to estimate diabetes likelihood.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Layout ────────────────────────────────────────────────────────────────────
col_form, col_gap, col_result = st.columns([3, 0.2, 2])

with col_form:
    st.markdown('<div class="section-label">📋 Patient Metrics</div>', unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        pregnancies = st.number_input(
            "Pregnancies", min_value=0, max_value=20, value=3,
            help="Number of times pregnant"
        )
        blood_pressure = st.number_input(
            "Blood Pressure (mm Hg)", min_value=0, max_value=140, value=70,
            help="Diastolic blood pressure"
        )
        insulin = st.number_input(
            "Insulin (mu U/ml)", min_value=0, max_value=900, value=100,
            help="2-Hour serum insulin"
        )
        dpf = st.number_input(
            "Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
            value=0.47, step=0.01,
            help="Diabetes pedigree function (genetic influence)"
        )

    with c2:
        glucose = st.number_input(
            "Glucose (mg/dL)", min_value=0, max_value=300, value=120,
            help="Plasma glucose concentration (2h oral glucose tolerance test)"
        )
        skin_thickness = st.number_input(
            "Skin Thickness (mm)", min_value=0, max_value=100, value=20,
            help="Triceps skin fold thickness"
        )
        bmi = st.number_input(
            "BMI (kg/m²)", min_value=0.0, max_value=70.0, value=32.0,
            step=0.1, help="Body mass index"
        )
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=33,
            help="Age in years"
        )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_btn = st.button("🔍 Analyze Risk", use_container_width=True)

# ── Sidebar / Info panel ──────────────────────────────────────────────────────
with col_result:
    st.markdown('<div class="section-label">📊 Results</div>', unsafe_allow_html=True)

    if not predict_btn:
        st.markdown("""
        <div class="card" style="text-align:center; padding: 40px 24px;">
            <div style="font-size:48px; margin-bottom:12px;">🩺</div>
            <div style="color: var(--muted); font-size:14px; line-height:1.6;">
                Fill in the patient metrics on the left and click
                <strong style="color:var(--accent)">Analyze Risk</strong>
                to get your prediction.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card">
            <div class="section-label">About this model</div>
            <div style="font-size: 13px; color: var(--muted); line-height: 1.7;">
                <b style="color:var(--text)">Algorithm:</b> Gaussian Naïve Bayes<br>
                <b style="color:var(--text)">Dataset:</b> Pima Indians Diabetes<br>
                <b style="color:var(--text)">Features:</b> 8 clinical metrics<br>
                <b style="color:var(--text)">Output:</b> Binary classification
            </div>
            <div style="margin-top: 14px;">
                <span class="feature-pill">Pregnancies</span>
                <span class="feature-pill">Glucose</span>
                <span class="feature-pill">Blood Pressure</span>
                <span class="feature-pill">Skin Thickness</span>
                <span class="feature-pill">Insulin</span>
                <span class="feature-pill">BMI</span>
                <span class="feature-pill">DPF</span>
                <span class="feature-pill">Age</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        input_data = np.array([[
            pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, dpf, age
        ]])

        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0]
        prob_positive = proba[1]
        prob_negative = proba[0]

        if prediction == 1:
            pct = int(prob_positive * 100)
            bar_pct = pct
            st.markdown(f"""
            <div class="result-positive">
                <div class="result-emoji">⚠️</div>
                <div class="result-label" style="color: #f85149;">High Diabetes Risk</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill-pos" style="width:{bar_pct}%"></div>
                </div>
                <div class="result-prob">Confidence: <strong style="color:#f85149">{pct}%</strong> probability of diabetes</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            pct = int(prob_negative * 100)
            bar_pct = pct
            st.markdown(f"""
            <div class="result-negative">
                <div class="result-emoji">✅</div>
                <div class="result-label" style="color: #3fb950;">Low Diabetes Risk</div>
                <div class="prob-bar-wrap">
                    <div class="prob-bar-fill-neg" style="width:{bar_pct}%"></div>
                </div>
                <div class="result-prob">Confidence: <strong style="color:#3fb950">{pct}%</strong> probability of no diabetes</div>
            </div>
            """, unsafe_allow_html=True)

        # Probability breakdown
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Probability Breakdown</div>', unsafe_allow_html=True)

        m1, m2 = st.columns(2)
        with m1:
            st.metric("No Diabetes", f"{prob_negative*100:.1f}%")
        with m2:
            st.metric("Diabetes", f"{prob_positive*100:.1f}%")

        # Input summary
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Input Summary</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="card" style="padding: 16px 20px;">
        <div class="info-grid">
            <div class="info-item"><div class="val">{pregnancies}</div><div class="lbl">Pregnancies</div></div>
            <div class="info-item"><div class="val">{glucose}</div><div class="lbl">Glucose</div></div>
            <div class="info-item"><div class="val">{blood_pressure}</div><div class="lbl">Blood Pressure</div></div>
            <div class="info-item"><div class="val">{bmi}</div><div class="lbl">BMI</div></div>
            <div class="info-item"><div class="val">{age}</div><div class="lbl">Age</div></div>
            <div class="info-item"><div class="val">{dpf}</div><div class="lbl">DPF</div></div>
        </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="font-size:11px; color: var(--muted); margin-top: 12px; padding: 12px; border: 1px solid var(--border); border-radius: 8px; line-height: 1.6;">
        ⚕️ <strong>Disclaimer:</strong> This tool is for educational and research purposes only.
        Always consult a licensed medical professional for health decisions.
        </div>
        """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("<br><hr>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color: var(--muted); font-size: 12px; padding: 8px 0;">
    Gaussian Naïve Bayes · Pima Indians Diabetes Dataset · Built with Streamlit
</div>
""", unsafe_allow_html=True)