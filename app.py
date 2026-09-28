"""
Streamlit Demo App: FraudShield - Credit Card Fraud Detection System
Full-depth version showing all technical contributions
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import time
import random

sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts'))
from config import PREDICTORS, TARGET

st.set_page_config(page_title="FraudShield", page_icon="🛡️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main-header { background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem; border-radius: 15px; margin-bottom: 1.5rem; text-align: center; }
    .main-header h1 { color: white; font-size: 2.3rem; margin-bottom: 0.3rem; }
    .main-header p { color: #d0d9f0; font-size: 1.05rem; }
    .stat-card { background: linear-gradient(135deg, #232946 0%, #16213e 100%);
        border-radius: 12px; padding: 1.3rem; text-align: center; border-left: 4px solid #4ecca3; }
    .decision-block { padding: 1.3rem; border-radius: 12px; text-align: center;
        font-size: 1.2rem; font-weight: bold; margin: 1rem 0; }
    .approve { background: linear-gradient(135deg, #134e4a, #0d9488); color: white; }
    .review { background: linear-gradient(135deg, #78350f, #d97706); color: white; }
    .block { background: linear-gradient(135deg, #7f1d1d, #dc2626); color: white; }
    .finding-box { background: #1a1f2e; border-left: 4px solid #4ecca3; padding: 1rem 1.3rem;
        border-radius: 8px; margin: 0.8rem 0; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    xgb_model = xgb.Booster()
    xgb_model.load_model('saved_models/xgb_model.json')
    with open('saved_models/iso_forest.pkl', 'rb') as f:
        iso_forest = pickle.load(f)
    with open('saved_models/shap_explainer.pkl', 'rb') as f:
        explainer = pickle.load(f)
    demo_df = pd.read_csv('saved_models/demo_sample.csv')
    return xgb_model, iso_forest, explainer, demo_df


@st.cache_data
def load_app_data():
    data = {}
    data['shap_sample'] = pd.read_csv('saved_models/shap_sample.csv')
    data['cost_curve'] = pd.read_csv('saved_models/cost_curve_data.csv')
    data['drift_summary'] = pd.read_csv('saved_models/drift_summary.csv')
    data['psi_results'] = pd.read_csv('saved_models/psi_results.csv')
    data['alpha_results'] = pd.read_csv('saved_models/alpha_results.csv')
    return data


xgb_model, iso_forest, explainer, demo_df = load_models()
app_data = load_app_data()


def get_anomaly_score(iso_forest, row_df):
    raw_score = iso_forest.decision_function(row_df[PREDICTORS])
    anomaly_score = -raw_score[0]
    return 1 / (1 + np.exp(-anomaly_score))


def make_gauge(value, title):
    color = "#dc2626" if value >= 70 else "#d97706" if value >= 30 else "#0d9488"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value, title={'text': title, 'font': {'size': 15, 'color': 'white'}},
        number={'suffix': "%", 'font': {'color': 'white', 'size': 26}},
        gauge={'axis': {'range': [0, 100], 'tickcolor': 'white'}, 'bar': {'color': color},
               'bgcolor': "#1a1f2e", 'borderwidth': 0,
               'steps': [{'range': [0, 30], 'color': '#0f2419'}, {'range': [30, 70], 'color': '#2b1e05'},
                         {'range': [70, 100], 'color': '#2b0a0a'}]}))
    fig.update_layout(height=200, margin=dict(l=20, r=20, t=45, b=10),
                       paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"})
    return fig


def dark_layout(fig, height=350):
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                       font={'color': "white"}, height=height, margin=dict(l=10, r=10, t=40, b=10))
    return fig


# ============================================================
# HEADER + HIGHLIGHTS
# ============================================================
st.markdown("""
<div class="main-header"><h1>🛡️ FraudShield</h1>
<p>Hybrid ML + DL Framework | 13 Models | Cost-Aware Thresholds | Drift-Tested | Explainable AI</p></div>
""", unsafe_allow_html=True)

hcols = st.columns(5)
highlights = [("13", "Models Compared", "🤖"), ("94.4%", "Fraud Density in BLOCK", "🎯"),
              ("27.2%", "Cost Reduction", "💰"), ("86.3%", "Fraud Caught (Consensus)", "🛡️"),
              ("8.28", "Max PSI Drift (Time)", "📉")]
for col, (val, label, icon) in zip(hcols, highlights):
    col.markdown(f'<div class="stat-card"><div style="font-size:1.6rem">{icon}</div>'
                 f'<div style="font-size:1.4rem; font-weight:bold; color:#4ecca3;">{val}</div>'
                 f'<div style="font-size:0.8rem; color:#aab2c8;">{label}</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "🔍 Live Fraud Check", "🧠 Global Model Behavior", "💰 Cost Optimization",
    "📉 Concept Drift", "⚖️ Fairness & Consensus", "🎛️ What-If Simulator",
    "📡 Live Stream Monitor", "👤 Analyst Review Queue"
])

# ============================================================
# TAB 1: LIVE CHECK
# ============================================================
with tab1:
    left, right = st.columns([1, 2.5])
    with left:
        st.markdown("#### Select a Transaction")
        selection_mode = st.radio("Input:", ["Sample transaction", "Custom values"], label_visibility="collapsed")
        if selection_mode == "Sample transaction":
            transaction_idx = st.selectbox("Transaction index:", options=demo_df.index.tolist())
            selected_row = demo_df.loc[[transaction_idx]]
            actual_label = demo_df.loc[transaction_idx, TARGET]
            st.markdown(f"**Actual Label:** {'🚨 FRAUD' if actual_label == 1 else '✅ GENUINE'}")
            st.markdown(f"**Amount:** ${demo_df.loc[transaction_idx, 'Amount']:.2f}")
        else:
            custom_values = {'Time': st.number_input("Time", value=50000.0),
                              'Amount': st.number_input("Amount", value=100.0)}
            with st.expander("Advanced feature values (V1-V28)"):
                for feat in PREDICTORS:
                    if feat not in ['Time', 'Amount']:
                        custom_values[feat] = st.number_input(feat, value=0.0, format="%.4f")
            selected_row = pd.DataFrame([custom_values])
            actual_label = None

    dmatrix = xgb.DMatrix(selected_row[PREDICTORS])
    xgb_proba = xgb_model.predict(dmatrix)[0]
    anomaly_score = get_anomaly_score(iso_forest, selected_row)
    hybrid_score = 0.7 * xgb_proba + 0.3 * anomaly_score

    with right:
        g1, g2, g3 = st.columns(3)
        g1.plotly_chart(make_gauge(xgb_proba * 100, "XGBoost Probability"), use_container_width=True)
        g2.plotly_chart(make_gauge(anomaly_score * 100, "Anomaly Score"), use_container_width=True)
        g3.plotly_chart(make_gauge(hybrid_score * 100, "Hybrid Score"), use_container_width=True)

        if hybrid_score >= 0.7:
            st.markdown('<div class="decision-block block">🚫 BLOCK — High confidence fraud</div>', unsafe_allow_html=True)
        elif hybrid_score >= 0.3:
            st.markdown('<div class="decision-block review">⚠️ REVIEW — Manual check recommended</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="decision-block approve">✅ APPROVE — Appears genuine</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔍 Local Explanation (SHAP)")
    shap_values = explainer.shap_values(selected_row[PREDICTORS])
    contributions = pd.DataFrame({'Feature': PREDICTORS, 'Value': selected_row[PREDICTORS].values[0],
                                   'SHAP_Contribution': shap_values[0]})
    contributions['Abs'] = contributions['SHAP_Contribution'].abs()
    contributions = contributions.sort_values('Abs', ascending=False).reset_index(drop=True)
    top_5 = contributions.head(5)

    ec1, ec2 = st.columns([1, 1.3])
    with ec1:
        for _, row in top_5.iterrows():
            color = "🔴" if row['SHAP_Contribution'] > 0 else "🟢"
            label = "Increases risk" if row['SHAP_Contribution'] > 0 else "Decreases risk"
            st.markdown(f"{color} **{row['Feature']}** = `{row['Value']:.3f}` → {label} ({row['SHAP_Contribution']:.4f})")
    with ec2:
        fig = go.Figure(go.Bar(x=top_5['SHAP_Contribution'][::-1], y=top_5['Feature'][::-1], orientation='h',
                                marker_color=['#dc2626' if x > 0 else '#0d9488' for x in top_5['SHAP_Contribution'][::-1]]))
        st.plotly_chart(dark_layout(fig, 280), use_container_width=True)

# ============================================================
# TAB 2: GLOBAL MODEL BEHAVIOR (SHAP global + Alpha tuning)
# ============================================================
with tab2:
    st.markdown("### 🧠 Global Feature Importance (SHAP, 1000-transaction sample)")
    st.caption("Which features drive fraud predictions ACROSS the entire dataset, not just one transaction")

    shap_sample = app_data['shap_sample']
    global_shap_values = explainer.shap_values(shap_sample[PREDICTORS])
    mean_abs_shap = np.abs(global_shap_values).mean(axis=0)
    global_importance = pd.DataFrame({'Feature': PREDICTORS, 'Mean |SHAP|': mean_abs_shap})
    global_importance = global_importance.sort_values('Mean |SHAP|', ascending=False).head(12)

    fig = go.Figure(go.Bar(x=global_importance['Mean |SHAP|'][::-1], y=global_importance['Feature'][::-1],
                            orientation='h', marker_color='#4ecca3'))
    fig.update_layout(title="Top 12 Most Influential Features Overall")
    st.plotly_chart(dark_layout(fig, 450), use_container_width=True)

    st.markdown(f"""
    <div class="finding-box">
    💡 <b>Finding:</b> <b>{global_importance.iloc[0]['Feature']}</b> is the single most influential feature 
    across the entire dataset, followed by <b>{global_importance.iloc[1]['Feature']}</b> and 
    <b>{global_importance.iloc[2]['Feature']}</b>. This confirms these anonymized PCA components 
    consistently encode the strongest fraud-discriminative signal.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔬 Hybrid Model Alpha Tuning")
    st.caption("Testing different weightings between XGBoost (supervised) and Isolation Forest (unsupervised)")

    alpha_df = app_data['alpha_results']
    fig2 = go.Figure()
    for metric, color in zip(['F1-Score', 'AUPRC', 'Precision', 'Recall'], ['#4ecca3', '#e94560', '#f4a261', '#2a9d8f']):
        fig2.add_trace(go.Scatter(x=alpha_df['Alpha'], y=alpha_df[metric], mode='lines+markers',
                                   name=metric, line=dict(color=color, width=3)))
    fig2.update_layout(title="Performance vs Alpha (Weight given to XGBoost)", xaxis_title="Alpha", yaxis_title="Score")
    st.plotly_chart(dark_layout(fig2, 400), use_container_width=True)

    best_alpha_row = alpha_df.loc[alpha_df['F1-Score'].idxmax()]
    st.markdown(f"""
    <div class="finding-box">
    💡 <b>Finding:</b> Alpha = <b>{best_alpha_row['Alpha']}</b> gives the best F1-Score 
    (<b>{best_alpha_row['F1-Score']:.4f}</b>) and Precision (<b>{best_alpha_row['Precision']:.4f}</b>), 
    representing the optimal balance between supervised (XGBoost) and unsupervised (Isolation Forest) 
    signals — this is the configuration used in our live demo.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 3: COST OPTIMIZATION
# ============================================================
with tab3:
    st.markdown("### 💰 Cost-Sensitive Threshold Optimization")
    st.caption("Instead of a fixed 0.5 cutoff, we find the threshold that minimizes real-world cost")

    cost_df = app_data['cost_curve']
    best_row = cost_df.loc[cost_df['total_cost'].idxmin()]
    default_row = cost_df.iloc[(cost_df['threshold'] - 0.5).abs().idxmin()]

    c1, c2, c3 = st.columns(3)
    c1.metric("Default Threshold (0.5) Cost", f"₹{default_row['total_cost']:,.0f}")
    c2.metric("Optimal Threshold Cost", f"₹{best_row['total_cost']:,.0f}",
              delta=f"-₹{default_row['total_cost'] - best_row['total_cost']:,.0f}")
    savings_pct = ((default_row['total_cost'] - best_row['total_cost']) / default_row['total_cost']) * 100
    c3.metric("Cost Reduction", f"{savings_pct:.1f}%")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=cost_df['threshold'], y=cost_df['total_cost'], mode='lines',
                              line=dict(color='#4ecca3', width=3), name='Total Cost'))
    fig.add_vline(x=0.5, line_dash="dash", line_color="#dc2626", annotation_text="Default (0.5)")
    fig.add_vline(x=best_row['threshold'], line_dash="dash", line_color="#0d9488",
                  annotation_text=f"Optimal ({best_row['threshold']:.2f})")
    fig.update_layout(title="Total Cost vs Decision Threshold", xaxis_title="Threshold", yaxis_title="Total Cost (₹)")
    st.plotly_chart(dark_layout(fig, 400), use_container_width=True)

    st.markdown(f"""
    <div class="finding-box">
    💡 <b>Finding:</b> Lowering the threshold from 0.5 to <b>{best_row['threshold']:.2f}</b> catches more fraud 
    (missed fraud drops from {default_row['false_negatives']:.0f} to {best_row['false_negatives']:.0f}), 
    at the cost of more false alarms (from {default_row['false_positives']:.0f} to {best_row['false_positives']:.0f}). 
    Since missed fraud (₹5,000) costs far more than a false alarm (₹50), this trade-off saves 
    <b>₹{default_row['total_cost'] - best_row['total_cost']:,.0f}</b> overall.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 4: CONCEPT DRIFT
# ============================================================
with tab4:
    st.markdown("### 📉 Concept Drift Testing")
    st.caption("Does the model still work on 'future' transactions it has never seen the pattern of?")

    drift_df = app_data['drift_summary']
    d1, d2 = st.columns(2)
    d1.metric("Same-Period AUC", f"{drift_df.iloc[0]['AUC']:.4f}")
    d2.metric("Later-Period AUC (Drift Test)", f"{drift_df.iloc[1]['AUC']:.4f}",
              delta=f"{drift_df.iloc[1]['AUC'] - drift_df.iloc[0]['AUC']:.4f}")

    fig = go.Figure(go.Bar(x=drift_df['Period'], y=drift_df['AUC'], marker_color=['#4ecca3', '#e94560']))
    fig.update_layout(title="AUC: Same-Period vs Later-Period", yaxis_range=[0, 1])
    st.plotly_chart(dark_layout(fig, 350), use_container_width=True)

    st.markdown("---")
    st.markdown("### 📊 Feature-Level Drift (Population Stability Index)")
    psi_df = app_data['psi_results']

    def psi_color(val):
        return '#dc2626' if val > 0.25 else '#d97706' if val > 0.1 else '#0d9488'

    fig2 = go.Figure(go.Bar(x=psi_df['PSI'], y=psi_df['Feature'], orientation='h',
                             marker_color=[psi_color(v) for v in psi_df['PSI']]))
    fig2.add_vline(x=0.25, line_dash="dash", line_color="white", annotation_text="Significant Drift Threshold")
    fig2.update_layout(title="PSI by Feature (Red = Significant Drift, Orange = Moderate, Green = Stable)")
    st.plotly_chart(dark_layout(fig2, 400), use_container_width=True)

    st.markdown(f"""
    <div class="finding-box">
    💡 <b>Finding:</b> <b>{psi_df.iloc[0]['Feature']}</b> shows the most extreme drift (PSI = {psi_df.iloc[0]['PSI']:.2f}), 
    followed by {psi_df.iloc[1]['Feature']} and {psi_df.iloc[2]['Feature']}. This confirms genuine behavioral 
    shifts occurred between the early and later transaction periods, validating the need for periodic model retraining.
    </div>
    """, unsafe_allow_html=True)

# ============================================================
# TAB 5: FAIRNESS & CONSENSUS
# ============================================================
with tab5:
    st.markdown("### 🎯 Multi-Model Consensus Triage")
    st.caption("Combining votes from multiple models to prioritize review queues")

    triage_data = pd.DataFrame({'Category': ['APPROVE', 'REVIEW', 'BLOCK'],
                                 'Transactions': [44697, 800, 72], 'Fraud Rate %': [0.031, 2.5, 94.444]})
    tc1, tc2 = st.columns(2)
    with tc1:
        fig = go.Figure(go.Bar(x=triage_data['Category'], y=triage_data['Transactions'],
                                marker_color=['#0d9488', '#d97706', '#dc2626']))
        fig.update_layout(title="Volume per Triage Category (log scale)", yaxis_type="log")
        st.plotly_chart(dark_layout(fig, 350), use_container_width=True)
    with tc2:
        fig = go.Figure(go.Bar(x=triage_data['Category'], y=triage_data['Fraud Rate %'],
                                marker_color=['#0d9488', '#d97706', '#dc2626']))
        fig.update_layout(title="Fraud Concentration per Category (%)")
        st.plotly_chart(dark_layout(fig, 350), use_container_width=True)

    st.markdown("""
    <div class="finding-box">
    💡 <b>Finding:</b> Just 72 transactions (0.16% of volume) in the BLOCK category contain 94.4% fraud density, 
    while 86.3% of all fraud is captured across BLOCK + REVIEW — dramatically reducing manual review workload.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚖️ Fairness Audit — Amount-Based Bias Check")
    fairness_data = pd.DataFrame({'Bracket': ['Low (<$50)', 'Medium ($50-500)', 'High (>$500)'],
                                   'False Positive Rate': [0.000132, 0.000291, 0.000000],
                                   'False Negative Rate': [0.328, 0.185, 0.500]})
    fig = go.Figure()
    fig.add_trace(go.Bar(x=fairness_data['Bracket'], y=fairness_data['False Positive Rate'], name='False Positive Rate', marker_color='#dc2626'))
    fig.add_trace(go.Bar(x=fairness_data['Bracket'], y=fairness_data['False Negative Rate'], name='False Negative Rate', marker_color='#3498db'))
    fig.update_layout(title="Error Rates Across Transaction Amount Brackets", barmode='group')
    st.plotly_chart(dark_layout(fig, 380), use_container_width=True)

    st.markdown("""
    <div class="finding-box">
    💡 <b>Finding:</b> False Positive Rate is consistent across brackets (no bias in false alarms), but 
    False Negative Rate varies significantly — high-value transactions (>$500) have a 50% miss rate vs 
    18.5% for medium transactions, suggesting amount-specific threshold calibration could improve fairness.
    </div>
    """, unsafe_allow_html=True)
    
# ============================================================
# TAB 6: WHAT-IF SIMULATOR
# ============================================================
with tab6:
    st.markdown("### 🎛️ Interactive What-If Simulator")
    st.caption("Adjust key transaction features and watch the fraud score respond in real time")

    base_row = demo_df.sample(n=1, random_state=1).iloc[0].copy()

    sim_col1, sim_col2 = st.columns([1, 1.5])

    with sim_col1:
        st.markdown("#### Adjust Feature Values")
        sim_amount = st.slider("Amount ($)", 0.0, 5000.0, float(base_row['Amount']), step=10.0)
        sim_time = st.slider("Time (seconds elapsed)", 0.0, 172800.0, float(base_row['Time']), step=100.0)
        sim_v14 = st.slider("V14 (top fraud-predictive feature)", -20.0, 10.0, float(base_row['V14']), step=0.1)
        sim_v4 = st.slider("V4", -10.0, 10.0, float(base_row['V4']), step=0.1)
        sim_v10 = st.slider("V10", -20.0, 10.0, float(base_row['V10']), step=0.1)

    sim_row = base_row.copy()
    sim_row['Amount'] = sim_amount
    sim_row['Time'] = sim_time
    sim_row['V14'] = sim_v14
    sim_row['V4'] = sim_v4
    sim_row['V10'] = sim_v10
    sim_df = pd.DataFrame([sim_row[PREDICTORS]])

    sim_dmatrix = xgb.DMatrix(sim_df[PREDICTORS])
    sim_proba = xgb_model.predict(sim_dmatrix)[0]
    sim_anomaly = get_anomaly_score(iso_forest, sim_df)
    sim_hybrid = 0.7 * sim_proba + 0.3 * sim_anomaly

    with sim_col2:
        st.markdown("#### Live Prediction")
        st.plotly_chart(make_gauge(sim_hybrid * 100, "Live Hybrid Fraud Score"), use_container_width=True)

        if sim_hybrid >= 0.7:
            st.markdown('<div class="decision-block block">🚫 BLOCK</div>', unsafe_allow_html=True)
        elif sim_hybrid >= 0.3:
            st.markdown('<div class="decision-block review">⚠️ REVIEW</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="decision-block approve">✅ APPROVE</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div class="finding-box">
        Try dragging <b>V14</b> below -5 to see the score spike — this is the model's single most 
        fraud-predictive signal, confirmed by our SHAP analysis.
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 7: LIVE STREAM MONITOR
# ============================================================
with tab7:
    st.markdown("### 📡 Live Transaction Stream Monitor")
    st.caption("Simulates a real-time payment monitoring feed, scoring each transaction as it arrives")

    if 'stream_log' not in st.session_state:
        st.session_state.stream_log = []
        st.session_state.stream_stats = {'total': 0, 'approved': 0, 'reviewed': 0, 'blocked': 0}

    run_stream = st.toggle("▶️ Start Live Stream Simulation")

    stream_placeholder = st.empty()
    stats_placeholder = st.empty()

    if run_stream:
        for _ in range(15):
            row_idx = random.choice(demo_df.index.tolist())
            row = demo_df.loc[[row_idx]]
            dmatrix_s = xgb.DMatrix(row[PREDICTORS])
            proba_s = xgb_model.predict(dmatrix_s)[0]
            anomaly_s = get_anomaly_score(iso_forest, row)
            hybrid_s = 0.7 * proba_s + 0.3 * anomaly_s

            if hybrid_s >= 0.7:
                decision, icon = "BLOCK", "🚫"
                st.session_state.stream_stats['blocked'] += 1
            elif hybrid_s >= 0.3:
                decision, icon = "REVIEW", "⚠️"
                st.session_state.stream_stats['reviewed'] += 1
            else:
                decision, icon = "APPROVE", "✅"
                st.session_state.stream_stats['approved'] += 1

            st.session_state.stream_stats['total'] += 1
            st.session_state.stream_log.insert(0, {
                'Txn ID': f"TXN-{random.randint(100000,999999)}",
                'Amount': f"${row['Amount'].values[0]:.2f}",
                'Score': f"{hybrid_s*100:.1f}%",
                'Decision': f"{icon} {decision}"
            })
            st.session_state.stream_log = st.session_state.stream_log[:10]

            with stats_placeholder.container():
                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Total Processed", st.session_state.stream_stats['total'])
                sc2.metric("✅ Approved", st.session_state.stream_stats['approved'])
                sc3.metric("⚠️ Reviewed", st.session_state.stream_stats['reviewed'])
                sc4.metric("🚫 Blocked", st.session_state.stream_stats['blocked'])

            with stream_placeholder.container():
                st.dataframe(pd.DataFrame(st.session_state.stream_log), use_container_width=True, hide_index=True)

            time.sleep(0.8)
    else:
        st.info("Toggle the switch above to start simulating live incoming transactions")
        if st.session_state.stream_log:
            st.dataframe(pd.DataFrame(st.session_state.stream_log), use_container_width=True, hide_index=True)

# ============================================================
# TAB 8: ANALYST REVIEW QUEUE
# ============================================================
with tab8:
    st.markdown("### 👤 Fraud Analyst Review Queue")
    st.caption("Transactions flagged for REVIEW, ready for human decision — with full SHAP context")

    if 'review_decisions' not in st.session_state:
        st.session_state.review_decisions = {}

    review_candidates = demo_df.sample(n=5, random_state=7)

    for idx, row in review_candidates.iterrows():
        row_df = pd.DataFrame([row])
        dmatrix_r = xgb.DMatrix(row_df[PREDICTORS])
        proba_r = xgb_model.predict(dmatrix_r)[0]
        anomaly_r = get_anomaly_score(iso_forest, row_df)
        hybrid_r = 0.7 * proba_r + 0.3 * anomaly_r

        shap_vals_r = explainer.shap_values(row_df[PREDICTORS])
        contrib_r = pd.DataFrame({'Feature': PREDICTORS, 'SHAP': shap_vals_r[0]})
        contrib_r['Abs'] = contrib_r['SHAP'].abs()
        top_reason = contrib_r.sort_values('Abs', ascending=False).iloc[0]

        with st.expander(f"Transaction #{idx} — Amount ${row['Amount']:.2f} — Score: {hybrid_r*100:.1f}%"):
            qc1, qc2 = st.columns([2, 1])
            with qc1:
                st.markdown(f"**Top risk factor:** {top_reason['Feature']} (impact: {top_reason['SHAP']:.4f})")
                st.markdown(f"**Fraud Score:** {hybrid_r*100:.1f}% | **Amount:** ${row['Amount']:.2f} | **Time:** {row['Time']:.0f}s")
            with qc2:
                decision_key = f"decision_{idx}"
                current = st.session_state.review_decisions.get(idx, "Pending")
                colA, colB = st.columns(2)
                if colA.button("✅ Approve", key=f"approve_{idx}"):
                    st.session_state.review_decisions[idx] = "Approved"
                if colB.button("🚫 Reject", key=f"reject_{idx}"):
                    st.session_state.review_decisions[idx] = "Rejected"
                st.markdown(f"**Status:** {st.session_state.review_decisions.get(idx, 'Pending')}")

    st.markdown("---")
    decided = sum(1 for v in st.session_state.review_decisions.values())
    st.markdown(f"""
    <div class="finding-box">
    📋 Analyst has processed <b>{decided}/5</b> queued transactions in this session.
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.caption("🛡️ FraudShield — 13-Model Comparison | Hybrid Architecture | Cost-Aware Thresholds | Drift-Tested | Explainable AI | Fairness-Audited")