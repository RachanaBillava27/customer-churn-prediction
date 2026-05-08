# ============================================================
#  app.py  —  Customer Churn Prediction  |  Streamlit App
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ─────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnSense AI",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0a0e1a;
    --surface:   #111827;
    --card:      #1a2235;
    --border:    #2a3a55;
    --accent:    #00d4ff;
    --accent2:   #7c3aed;
    --danger:    #ff4d6d;
    --safe:      #06d6a0;
    --text:      #e8edf5;
    --muted:     #8899bb;
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: var(--bg) !important;
    color: var(--text) !important;
}

.stApp { background-color: var(--bg); }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: var(--text) !important; }

/* ── Inputs ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"]  > div,
div[data-baseweb="slider"] {
    background: var(--card) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
.stSlider > div > div > div { background: var(--accent) !important; }
label { color: var(--muted) !important; font-size: 0.82rem !important; letter-spacing: 0.04em; text-transform: uppercase; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.05em;
    transition: all 0.3s ease !important;
    width: 100%;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(0, 212, 255, 0.35) !important;
}

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
}
[data-testid="stMetricValue"] { color: var(--accent) !important; font-family: 'Syne', sans-serif !important; font-size: 2rem !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; }

/* ── Divider ── */
hr { border-color: var(--border) !important; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Load artifacts ───────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model    = joblib.load("model.pkl")
    scaler   = joblib.load("scaler.pkl")
    feat_cols = joblib.load("feature_columns.pkl")
    return model, scaler, feat_cols

try:
    model, scaler, feat_cols = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"⚠️ Could not load model files. Run `save_model.py` first.\n\n{e}")

# ════════════════════════════════════════════════════════════
#  SIDEBAR — Input Form
# ════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='padding: 1rem 0 1.5rem'>
        <div style='font-family:Syne,sans-serif;font-size:1.6rem;font-weight:800;
                    background:linear-gradient(90deg,#00d4ff,#7c3aed);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            📡 ChurnSense
        </div>
        <div style='color:#8899bb;font-size:0.82rem;margin-top:2px;'>AI-Powered Churn Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### Customer Profile")
    st.markdown("---")

    # ── Demographics ────────────────────────────────────────
    st.markdown("**👤 Demographics**")
    gender         = st.selectbox("Gender",          ["Male", "Female"])
    senior_citizen = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner        = st.selectbox("Has Partner",     ["No", "Yes"])
    dependents     = st.selectbox("Has Dependents",  ["No", "Yes"])

    st.markdown("---")
    st.markdown("**📋 Account Info**")
    tenure             = st.slider("Tenure (months)",    0, 72, 12)
    contract           = st.selectbox("Contract Type",   ["Month-to-month", "One year", "Two year"])
    paperless_billing  = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method     = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check",
        "Bank transfer (automatic)", "Credit card (automatic)"
    ])

    st.markdown("---")
    st.markdown("**📞 Services**")
    phone_service    = st.selectbox("Phone Service",       ["No", "Yes"])
    multiple_lines   = st.selectbox("Multiple Lines",      ["No", "Yes", "No phone service"])
    internet_service = st.selectbox("Internet Service",    ["DSL", "Fiber optic", "No"])
    online_security  = st.selectbox("Online Security",     ["No", "Yes", "No internet service"])
    online_backup    = st.selectbox("Online Backup",       ["No", "Yes", "No internet service"])
    device_protection= st.selectbox("Device Protection",   ["No", "Yes", "No internet service"])
    tech_support     = st.selectbox("Tech Support",        ["No", "Yes", "No internet service"])
    streaming_tv     = st.selectbox("Streaming TV",        ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies",    ["No", "Yes", "No internet service"])

    st.markdown("---")
    st.markdown("**💰 Charges**")
    monthly_charges = st.slider("Monthly Charges ($)",  18.0, 120.0, 65.0, 0.5)
    total_charges   = st.slider("Total Charges ($)",     0.0, 9000.0, float(tenure * monthly_charges), 10.0)

    st.markdown("---")
    predict_btn = st.button("🔍 Predict Churn Risk", use_container_width=True)


# ════════════════════════════════════════════════════════════
#  MAIN AREA
# ════════════════════════════════════════════════════════════

# ── Header ──────────────────────────────────────────────────
st.markdown("""
<div style='padding: 2rem 0 1rem'>
    <h1 style='font-family:Syne,sans-serif;font-size:2.6rem;font-weight:800;margin:0;
               background:linear-gradient(90deg,#00d4ff 20%,#7c3aed 80%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
        Customer Churn Intelligence
    </h1>
    <p style='color:#8899bb;font-size:1rem;margin-top:0.4rem;'>
        Predict customer attrition risk using Random Forest ML model trained on Telco data.
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPI Row ─────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
k1.metric("Model",         "Random Forest")
k2.metric("Accuracy",      "~81%")
k3.metric("ROC-AUC Score", "~86%")
k4.metric("Features",      "19+")

st.markdown("---")

# ════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ════════════════════════════════════════════════════════════
def build_input_df():
    data = {
        "gender"           : gender,
        "SeniorCitizen"    : 1 if senior_citizen == "Yes" else 0,
        "Partner"          : partner,
        "Dependents"       : dependents,
        "tenure"           : tenure,
        "PhoneService"     : phone_service,
        "MultipleLines"    : multiple_lines,
        "InternetService"  : internet_service,
        "OnlineSecurity"   : online_security,
        "OnlineBackup"     : online_backup,
        "DeviceProtection" : device_protection,
        "TechSupport"      : tech_support,
        "StreamingTV"      : streaming_tv,
        "StreamingMovies"  : streaming_movies,
        "Contract"         : contract,
        "PaperlessBilling" : paperless_billing,
        "PaymentMethod"    : payment_method,
        "MonthlyCharges"   : monthly_charges,
        "TotalCharges"     : total_charges,
    }
    return pd.DataFrame([data])


def predict(input_df):
    cat_cols    = input_df.select_dtypes(include="object").columns.tolist()
    encoded     = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    aligned     = encoded.reindex(columns=feat_cols, fill_value=0)
    probability = model.predict_proba(aligned)[0][1]
    prediction  = model.predict(aligned)[0]
    return prediction, probability


# ── Gauge chart ─────────────────────────────────────────────
def gauge_chart(prob):
    pct   = prob * 100
    color = "#ff4d6d" if pct >= 60 else "#ffb703" if pct >= 35 else "#06d6a0"
    fig = go.Figure(go.Indicator(
        mode   = "gauge+number+delta",
        value  = pct,
        number = {"suffix": "%", "font": {"size": 48, "color": color, "family": "Syne"}},
        delta  = {"reference": 50, "increasing": {"color": "#ff4d6d"}, "decreasing": {"color": "#06d6a0"}},
        gauge  = {
            "axis"      : {"range": [0, 100], "tickcolor": "#2a3a55", "tickfont": {"color": "#8899bb"}},
            "bar"       : {"color": color, "thickness": 0.25},
            "bgcolor"   : "#1a2235",
            "bordercolor": "#2a3a55",
            "steps"     : [
                {"range": [0,  35], "color": "#0d1f15"},
                {"range": [35, 60], "color": "#1f1a0d"},
                {"range": [60, 100],"color": "#1f0d14"},
            ],
            "threshold" : {"line": {"color": color, "width": 4}, "thickness": 0.8, "value": pct},
        },
        title  = {"text": "Churn Probability", "font": {"size": 16, "color": "#8899bb", "family": "DM Sans"}},
    ))
    fig.update_layout(
        paper_bgcolor="#111827", plot_bgcolor="#111827",
        font_color="#e8edf5", height=280, margin=dict(t=30, b=10, l=20, r=20)
    )
    return fig


# ── Feature contribution bar chart ──────────────────────────
def feature_importance_chart():
    importances = model.feature_importances_
    fi_df = pd.DataFrame({"Feature": feat_cols, "Importance": importances})
    fi_df = fi_df.nlargest(10, "Importance").sort_values("Importance")

    fig = go.Figure(go.Bar(
        x=fi_df["Importance"], y=fi_df["Feature"],
        orientation="h",
        marker=dict(
            color=fi_df["Importance"],
            colorscale=[[0, "#1a2235"], [0.5, "#7c3aed"], [1, "#00d4ff"]],
            showscale=False,
            line=dict(color="#2a3a55", width=0.5)
        ),
        text=[f"{v:.3f}" for v in fi_df["Importance"]],
        textposition="outside",
        textfont=dict(color="#8899bb", size=11),
    ))
    fig.update_layout(
        title=dict(text="Top 10 Feature Importances", font=dict(family="Syne", size=15, color="#e8edf5")),
        paper_bgcolor="#111827", plot_bgcolor="#1a2235",
        xaxis=dict(showgrid=True, gridcolor="#2a3a55", color="#8899bb"),
        yaxis=dict(showgrid=False, color="#e8edf5"),
        font=dict(family="DM Sans"),
        height=380, margin=dict(t=50, b=30, l=10, r=60)
    )
    return fig


# ── Render result ────────────────────────────────────────────
if predict_btn and model_loaded:
    input_df             = build_input_df()
    prediction, prob     = predict(input_df)
    pct                  = prob * 100

    risk_label = "🔴 HIGH RISK"   if pct >= 60 \
            else "🟡 MEDIUM RISK" if pct >= 35 \
            else "🟢 LOW RISK"

    risk_color = "#ff4d6d" if pct >= 60 else "#ffb703" if pct >= 35 else "#06d6a0"

    # ── Result banner ──────────────────────────────────────
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#1a2235,#0f1724);
                border:1px solid {risk_color}44;border-left:4px solid {risk_color};
                border-radius:14px;padding:1.5rem 2rem;margin-bottom:1.5rem;
                box-shadow:0 4px 30px {risk_color}22;'>
        <div style='font-family:Syne,sans-serif;font-size:0.75rem;
                    letter-spacing:0.15em;color:#8899bb;text-transform:uppercase;margin-bottom:4px;'>
            Prediction Result
        </div>
        <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{risk_color};'>
            {risk_label}
        </div>
        <div style='color:#8899bb;font-size:0.9rem;margin-top:6px;'>
            This customer has a <strong style='color:{risk_color}'>{pct:.1f}%</strong> probability of churning.
            {"Take immediate retention action." if pct >= 60 else "Monitor and engage proactively." if pct >= 35 else "Customer appears stable."}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Gauge + Insights ──────────────────────────────────
    col_gauge, col_insights = st.columns([1, 1])

    with col_gauge:
        st.plotly_chart(gauge_chart(prob), use_container_width=True)

    with col_insights:
        st.markdown("""
        <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                    color:#e8edf5;margin-bottom:1rem;'>📌 Key Risk Factors</div>
        """, unsafe_allow_html=True)

        insights = []
        if contract == "Month-to-month":
            insights.append(("⚠️", "Month-to-month contract", "Highest churn risk contract type"))
        if monthly_charges > 70:
            insights.append(("💸", f"High monthly charge: ${monthly_charges}", "Above average billing"))
        if tenure < 12:
            insights.append(("🕐", f"Low tenure: {tenure} months", "New customers churn more"))
        if internet_service == "Fiber optic":
            insights.append(("🌐", "Fiber optic internet", "Associated with higher churn rates"))
        if tech_support == "No":
            insights.append(("🛠️", "No tech support", "Lack of support increases churn"))
        if not insights:
            insights.append(("✅", "No major risk factors", "Customer profile looks stable"))

        for icon, title, desc in insights:
            st.markdown(f"""
            <div style='background:#1a2235;border:1px solid #2a3a55;border-radius:10px;
                        padding:0.75rem 1rem;margin-bottom:0.6rem;'>
                <span style='font-size:1.1rem;'>{icon}</span>
                <span style='font-family:Syne,sans-serif;font-weight:600;
                             color:#e8edf5;margin-left:8px;font-size:0.9rem;'>{title}</span>
                <div style='color:#8899bb;font-size:0.78rem;margin-top:2px;margin-left:28px;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    # ── Retention Recommendations ─────────────────────────
    st.markdown("---")
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1.1rem;font-weight:700;
                color:#e8edf5;margin-bottom:1rem;'>💡 Retention Recommendations</div>
    """, unsafe_allow_html=True)

    r1, r2, r3 = st.columns(3)
    recs = [
        ("📝", "Upgrade Contract",    "Offer a discounted 1- or 2-year plan to lock in commitment."),
        ("🎁", "Loyalty Reward",      "Provide a personalized discount or free service upgrade."),
        ("📞", "Proactive Outreach",  "Assign a retention specialist to contact this customer."),
    ]
    for col, (icon, title, body) in zip([r1, r2, r3], recs):
        col.markdown(f"""
        <div style='background:#1a2235;border:1px solid #2a3a55;border-radius:12px;
                    padding:1.25rem;height:100%;'>
            <div style='font-size:1.8rem;margin-bottom:8px;'>{icon}</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;
                        color:#00d4ff;margin-bottom:6px;'>{title}</div>
            <div style='color:#8899bb;font-size:0.85rem;line-height:1.5;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature Importance ────────────────────────────────
    st.markdown("---")
    st.plotly_chart(feature_importance_chart(), use_container_width=True)

# ── Default landing state ────────────────────────────────────
elif not predict_btn:
    st.markdown("""
    <div style='background:#1a2235;border:1px solid #2a3a55;border-radius:16px;
                padding:3rem;text-align:center;margin-top:2rem;'>
        <div style='font-size:4rem;margin-bottom:1rem;'>📡</div>
        <div style='font-family:Syne,sans-serif;font-size:1.5rem;font-weight:700;
                    color:#e8edf5;margin-bottom:0.5rem;'>Ready to Analyze</div>
        <div style='color:#8899bb;font-size:0.95rem;max-width:400px;margin:0 auto;line-height:1.7;'>
            Fill in the customer details in the sidebar panel,
            then click <strong style='color:#00d4ff;'>Predict Churn Risk</strong> to get an instant AI-powered result.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── How it works ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;
                color:#8899bb;text-align:center;letter-spacing:0.1em;
                text-transform:uppercase;margin-bottom:1.5rem;'>How It Works</div>
    """, unsafe_allow_html=True)

    h1, h2, h3, h4 = st.columns(4)
    steps = [
        ("01", "Input Data",      "Enter customer demographics, services & billing details."),
        ("02", "ML Processing",   "Random Forest model processes 19+ encoded features."),
        ("03", "Risk Score",      "Get a churn probability score from 0–100%."),
        ("04", "Take Action",     "Follow AI-generated retention recommendations."),
    ]
    for col, (num, title, body) in zip([h1, h2, h3, h4], steps):
        col.markdown(f"""
        <div style='background:#111827;border:1px solid #2a3a55;border-radius:12px;
                    padding:1.25rem;text-align:center;'>
            <div style='font-family:Syne,sans-serif;font-size:2rem;font-weight:800;
                        color:#2a3a55;'>{num}</div>
            <div style='font-family:Syne,sans-serif;font-weight:700;
                        color:#00d4ff;margin:6px 0;font-size:0.9rem;'>{title}</div>
            <div style='color:#8899bb;font-size:0.8rem;line-height:1.5;'>{body}</div>
        </div>
        """, unsafe_allow_html=True)
