import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import warnings
from scipy.optimize import brentq

warnings.filterwarnings("ignore")

BETA = 9.3693   
ETA  = 73843.0  

SAFETY_MARGIN = 0.15

# Pre-compute PDF mode — fixed for every vehicle
T_PEAK = ETA * ((BETA - 1) / BETA) ** (1 / BETA)

# REQUIRED CSV COLUMNS
FEATURES = [
    'Desired Fuel Injection Quantity',
    'Desired Fuel Rail Pressure (FRP)',
    'Desired Mass Air Flow (MAF)',
    'Fuel Rail Pressure (FRP)',
    'Mass Air Flow (MAF)',
    'Main Fuel Injection Quantity',
    'Pre Fuel Injection Quantity',
    'Engine Speed',
    'Boost Pressure'
]

# WEIBULL PDF FUNCTIONS
def weibull_pdf(t):
    return (BETA/ETA) * (t/ETA)**(BETA-1) * np.exp(-(t/ETA)**BETA)

F_PEAK = weibull_pdf(T_PEAK)

def solve_t_current(p):

    p = np.clip(p, 1e-6, 1 - 1e-6)

    def obj(t):
        return weibull_pdf(t) / F_PEAK - p

    if obj(T_PEAK - 1e-6) < 0:
        return T_PEAK

    try:
        return brentq(obj, ETA * 1e-4, T_PEAK - 1e-6, xtol=0.1)
    except ValueError:
        return T_PEAK


def compute_rul(failure_prob, confidence, current_km):

    t_curr  = solve_t_current(failure_prob)
    rul_raw = T_PEAK - t_curr
    overdue = rul_raw <= 0

    rul     = 0.0 if overdue else rul_raw * (1 - SAFETY_MARGIN)
    maint   = float(current_km) if overdue else current_km + rul

    if failure_prob >= 0.70 or overdue:
        urgency = "CRITICAL"
    elif failure_prob >= 0.40:
        urgency = "HIGH"
    elif failure_prob >= 0.20:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    p_flag = failure_prob >= 0.65
    c_flag = confidence  >= 0.75

    if overdue:
        action = "Vehicle is at or past peak failure zone. IMMEDIATE maintenance required."
    elif p_flag and c_flag:
        action = "Fault confirmed, HIGH confidence. Schedule maintenance immediately."
    elif p_flag and not c_flag:
        action = "Fault signal, LOW confidence. Inspect within 2 operating cycles."
    elif not p_flag and not c_flag:
        action = "No fault detected. Use visit-after estimate to plan next service."
    else:
        action = f"Healthy, HIGH confidence. Next service at {maint:,.0f} km."

    return {
        "t_current" : round(t_curr,0),
        "t_peak"    : round(T_PEAK,0),
        "rul_raw"   : round(max(rul_raw,0),0),
        "rul"       : round(rul,0),
        "maint"     : round(maint,0),
        "urgency"   : urgency,
        "action"    : action,
        "overdue"   : overdue,
    }


# LOAD MODELS
@st.cache_resource
def load_models():
    try:
        with open("base_model.pkl", "rb") as f:
            base = pickle.load(f)
        calib = joblib.load("calibrated.pkl")
        return base, calib, None
    except Exception as e:
        return None, None, str(e)

base_model, calibrated_model, load_error = load_models()


# PAGE CONFIG
st.set_page_config(
    page_title="ECU-BASED ML-DRIVEN PREDICTIVE MAINTENANCE FOR SOLENOID TYPE DIESEL FUEL INJECTORS",
    page_icon="🔧",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items=None
)


# SIDEBAR
with st.sidebar:
    st.header("🔧 About")
    st.markdown(f"""
ECU-BASED ML-DRIVEN PREDICTIVE MAINTENANCE FOR SOLENOID TYPE DIESEL FUEL INJECTORS

---

**Weibull Parameters**

β = `{BETA}`  
η = `{ETA:,.0f} km`  
t_peak = `{T_PEAK:,.0f} km`

---

**Visit-After Estimation**


Higher probability → shorter visit interval.

---

**Urgency**

🔴 CRITICAL : p ≥ 70%  
🟠 HIGH     : 40–70%  
🟡 MEDIUM   : 20–40%  
🟢 LOW      : <20%
""")

    st.markdown("**Required CSV columns:**")
    for f in FEATURES:
        st.caption(f"• {f}")


# TITLE
st.title("🔧 ECU-BASED ML-DRIVEN PREDICTIVE MAINTENANCE FOR SOLENOID TYPE DIESEL FUEL INJECTORS")
st.markdown("Upload CSV to estimate injector health and **recommended service visit interval**.")
st.divider()


# MODEL STATUS
if load_error:
    st.error(load_error)
    st.stop()

st.success("ML models loaded.")


# INPUTS
col_l, col_r = st.columns([1,1])

with col_l:
    vehicle_id = st.text_input("Vehicle ID / Registration")
    current_km = st.number_input("Current Odometer Reading (km)",0,500000,0)

with col_r:
    uploaded = st.file_uploader("Upload ECU CSV",type=["csv"])
    if uploaded:
        st.caption(uploaded.name)

st.divider()


# PREDICTION
if uploaded:

    df = pd.read_csv(uploaded)

    missing = set(FEATURES) - set(df.columns)
    if missing:
        st.error(f"Missing columns: {missing}")
        st.stop()

    X = df[FEATURES]

    with st.spinner("Running ML prediction and Weibull analysis..."):
        failure_prob = base_model.predict_proba(X)[:,1].mean()
        confidence   = calibrated_model.predict_proba(X).max(axis=1).mean()
        res          = compute_rul(failure_prob,confidence,current_km)

    st.subheader("📊 Results")

    icons = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}

    m1,m2,m3,m4 = st.columns(4)

    m1.metric("Failure Probability",f"{failure_prob*100:.1f}%")
    m2.metric("Model Confidence",f"{confidence*100:.1f}%")
    m3.metric("Urgency",res["urgency"])
    m4.metric("Visit After",
              f"{res['rul']:,.0f} km" if res['rul']>0 else "OVERDUE")

    st.divider()


    st.subheader("Maintenance Recommendation")
    st.info(res["action"])


    st.subheader("Calculation Breakdown")

    c1,c2 = st.columns(2)

    with c1:
        st.table(pd.DataFrame({
            "Parameter":[
                "Rows analysed",
                "Failure probability",
                "Model confidence"
            ],
            "Value":[
                len(X),
                f"{failure_prob*100:.2f}%",
                f"{confidence*100:.2f}%"
            ]
        }))

    with c2:
        st.table(pd.DataFrame({
            "Parameter":[
                "Weibull β",
                "Weibull η",
                "t_peak",
                "t_current",
                "VisitAfter_raw",
                "Visit After",
                "Maintenance due"
            ],
            "Value":[
                BETA,
                ETA,
                res["t_peak"],
                res["t_current"],
                res["rul_raw"],
                res["rul"],
                res["maint"]
            ]
        }))


    with st.expander("Step-by-step Visit-After calculation"):

        st.markdown(f"""
1️⃣ Failure probability from ML model = **{failure_prob:.4f}**

2️⃣ Solve Weibull PDF position → **t_current = {res['t_current']} km**

3️⃣ Peak failure density → **t_peak = {T_PEAK:,.0f} km**

4️⃣ VisitAfter_raw  
`t_peak − t_current = {res['rul_raw']:,.0f} km`

5️⃣ Apply safety margin  

Visit After = `{res['rul']:,.0f} km`

6️⃣ Recommended service odometer  

`{current_km:,.0f} + {res['rul']:,.0f} = {res['maint']:,.0f} km`
""")


    with st.expander("Preview CSV"):
        st.dataframe(X.head())

else:

    st.info("Upload a CSV file to start analysis.")


st.markdown("---")
st.markdown(
"<center>Prepared by: Manish, Navaraj, Prabesh, Prashanta.</center>",
unsafe_allow_html=True
)

