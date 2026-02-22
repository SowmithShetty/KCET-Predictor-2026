import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Setup & Assets
st.set_page_config(page_title="KCET College Predictor (Engineering)", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        height: 3em;
    }
    div.stButton > button:first-child:hover { background-color: #218838; color: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = joblib.load('kcet_model_slim.pkl')
    le_coll = joblib.load('encoder_college.pkl')
    le_cour = joblib.load('encoder_course.pkl')
    le_cat = joblib.load('encoder_base_cat.pkl')
    le_quot = joblib.load('encoder_quota.pkl')
    le_reg = joblib.load('encoder_region.pkl')
    df = pd.read_csv('kcet_ai_ready.csv')
    return model, le_coll, le_cour, le_cat, le_quot, le_reg, df

model, le_coll, le_cour, le_cat, le_quot, le_reg, df = load_assets()

if 'results_df' not in st.session_state:
    st.session_state.results_df = None

# 2. Sidebar
with st.sidebar:
    st.header("📋 Student Profile")
    u_rank = st.number_input("Your Rank", min_value=1, value=25000)
    u_reg = st.selectbox("Region", le_reg.classes_)
    u_cat = st.selectbox("Category", le_cat.classes_)
    u_quot = st.selectbox("Quota", le_quot.classes_)
    u_cour = st.selectbox("Target Branch", sorted(le_cour.classes_))
    
    if st.button("🚀 Prediction for 2026"):
        with st.spinner('Analyzing trends...'):
            targets = df[df['CourseName'] == u_cour].drop_duplicates('CollegeName')
            preds = []
            for _, row in targets.iterrows():
                try:
                    feat = np.array([[le_coll.transform([row['CollegeName']])[0], 
                                     le_cour.transform([u_cour])[0], le_cat.transform([u_cat])[0],
                                     le_quot.transform([u_quot])[0], le_reg.transform([u_reg])[0], 2026]])
                    p_rank = np.expm1(model.predict(feat))[0]
                    if u_rank <= (p_rank * 1.3):
                        status = "✅ High Chance" if u_rank <= p_rank else "🟡 Borderline"
                        preds.append({"College": row['CollegeName'], "2026 Est. Cutoff": int(p_rank), "Status": status})
                except: continue
            st.session_state.results_df = pd.DataFrame(preds).sort_values("2026 Est. Cutoff") if preds else "No Matches"

# 3. Main UI
st.title("🎓 KCET 2026 College Predictor")
st.markdown("---")

if st.session_state.results_df is not None:
    if isinstance(st.session_state.results_df, str):
        st.error("No matches found.")
    else:
        st.subheader("📍 Recommended Colleges")
        st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
        
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.subheader("📈 AI Forecast & Historical Trend")
        
        graph_coll = st.selectbox("Select College to see Forecast:", st.session_state.results_df['College'].tolist())
        
        # Get History
        hist = df[(df['CollegeName'] == graph_coll) & (df['CourseName'] == u_cour) & 
                  (df['Base_Category'] == u_cat) & (df['Region'] == u_reg)].copy()
        hist = hist.sort_values('Year')
        
        # Get the 2026 Prediction again for the specific selected college
        pred_2026 = st.session_state.results_df[st.session_state.results_df['College'] == graph_coll]["2026 Est. Cutoff"].values[0]
        
        if not hist.empty:
            # Create the Forecast Graph
            fig = go.Figure()

            # 1. Historical Data (Blue Solid Line)
            fig.add_trace(go.Scatter(x=hist['Year'], y=hist['Cutoff_Rank'], mode='lines+markers+text',
                                     name='Historical Cutoff', text=hist['Cutoff_Rank'],
                                     textposition="top center", line=dict(color='#007bff', width=3)))

            # 2. Connect 2025 to 2026 (Green Dashed Line for Prediction)
            last_year = hist['Year'].max()
            last_rank = hist[hist['Year'] == last_year]['Cutoff_Rank'].values[0]
            
            fig.add_trace(go.Scatter(x=[last_year, 2026], y=[last_rank, pred_2026],
                                     mode='lines+markers+text', name='AI 2026 Forecast',
                                     text=["", f"<b>{int(pred_2026)}</b>"], textposition="top center",
                                     line=dict(color='#28a745', width=4, dash='dash')))

            fig.update_layout(title=f"Cutoff Forecast: {graph_coll}", xaxis=dict(tickmode='linear', dtick=1),
                              yaxis_title="Cutoff Rank", hovermode="x unified", showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
            st.success(f"The AI predicts the 2026 cutoff for this college will be around **{int(pred_2026)}**.")
        else:
            st.warning("Not enough historical data to generate a forecast for this combination.")
