import streamlit as st
import pandas as pd
import joblib
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# 1. Setup & Assets
st.set_page_config(page_title="KCET 2026 Pro Predictor", page_icon="🎓", layout="centered")

st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    div.stButton > button:first-child {
        background-color: #28a745;
        color: white;
        border-radius: 6px;
        font-weight: bold;
        height: 3em;
        width: 100%;
    }
    .difficulty-tag { font-weight: bold; padding: 2px 8px; border-radius: 4px; }
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

# Initialize Session States
if 'results_df' not in st.session_state: st.session_state.results_df = None
if 'top_5_df' not in st.session_state: st.session_state.top_5_df = None

# Helper: Calculate Difficulty Score
def get_difficulty(college, course, cat, reg):
    hist = df[(df['CollegeName'] == college) & (df['CourseName'] == course) & 
              (df['Base_Category'] == cat) & (df['Region'] == reg)].sort_values('Year')
    if len(hist) < 2: return "⚪ Stable"
    
    first = hist['Cutoff_Rank'].iloc[0]
    last = hist['Cutoff_Rank'].iloc[-1]
    change = ((last - first) / first) * 100
    
    if change < -5: return "🔴 Getting Harder"
    elif change > 5: return "🟢 Getting Easier"
    else: return "🔵 Stable"

# 2. Sidebar
with st.sidebar:
    st.header("📋 Student Profile")
    u_rank = st.number_input("Your Rank", min_value=1, value=25000)
    u_reg = st.selectbox("Region", le_reg.classes_)
    u_cat = st.selectbox("Category", le_cat.classes_)
    u_quot = st.selectbox("Quota", le_quot.classes_)
    u_cour = st.selectbox("Target Branch", sorted(le_cour.classes_))
    
    predict_btn = st.button("🚀 Predict for 2026")
    best_fit_btn = st.button("📍 Find My Best Fit (All Branches)")

# 3. Logic: Single Branch Prediction
if predict_btn:
    st.session_state.top_5_df = None
    with st.spinner('Analyzing trends...'):
        targets = df[df['CourseName'] == u_cour].drop_duplicates('CollegeName')
        preds = []
        for _, row in targets.iterrows():
            try:
                feat = np.array([[le_coll.transform([row['CollegeName']])[0], 
                                 le_cour.transform([u_cour])[0], le_cat.transform([u_cat])[0],
                                 le_quot.transform([u_quot])[0], le_reg.transform([u_reg])[0], 2026]])
                p_rank = np.expm1(model.predict(feat))[0]
                
                if u_rank <= (p_rank * 1.4):
                    status = "✅ High Chance" if u_rank <= p_rank else "🟡 Borderline"
                    diff = get_difficulty(row['CollegeName'], u_cour, u_cat, u_reg)
                    
                    # FEATURE 1: Safety Range (±10%)
                    range_str = f"{int(p_rank*0.9):,} - {int(p_rank*1.1):,}"
                    
                    preds.append({
                        "College": row['CollegeName'], 
                        "Safety Range": range_str,
                        "Avg Cutoff": int(p_rank),
                        "Status": status,
                        "Trend": diff # FEATURE 5: Difficulty Score
                    })
            except: continue
        st.session_state.results_df = pd.DataFrame(preds).sort_values("Avg Cutoff") if preds else "No Matches"

# 4. Logic: Top 5 Best Fit (FEATURE 4)
if best_fit_btn:
    st.session_state.results_df = None
    with st.spinner('Scanning all branches for best matches...'):
        all_colleges = df.drop_duplicates(['CollegeName', 'CourseName']).sample(150) # Sampled for speed
        fits = []
        for _, row in all_colleges.iterrows():
            try:
                feat = np.array([[le_coll.transform([row['CollegeName']])[0], 
                                 le_cour.transform([row['CourseName']])[0], le_cat.transform([u_cat])[0],
                                 le_quot.transform([u_quot])[0], le_reg.transform([u_reg])[0], 2026]])
                p_rank = np.expm1(model.predict(feat))[0]
                
                # Find colleges where predicted cutoff is just above user rank
                if p_rank >= u_rank:
                    fits.append({
                        "College": row['CollegeName'],
                        "Branch": row['CourseName'],
                        "Est. Cutoff": int(p_rank),
                        "Match Score": int(p_rank - u_rank)
                    })
            except: continue
        
        if fits:
            st.session_state.top_5_df = pd.DataFrame(fits).sort_values("Match Score").head(5)
        else:
            st.session_state.top_5_df = "No Matches"

# 5. Main UI Display
st.title("🎓 KCET 2026 College Predictor")

# Display Single Branch Results
if st.session_state.results_df is not None:
    if isinstance(st.session_state.results_df, str):
        st.error("No matches found for this branch.")
    else:
        st.subheader(f"📍 Recommended Colleges for {u_cour}")
        st.dataframe(st.session_state.results_df, use_container_width=True, hide_index=True)
        
        # FEATURE 2: College Comparison Tool
        st.markdown("---")
        st.subheader("⚖️ Compare College Trends")
        st.write("Select up to 2 colleges to see their 2026 forecast trajectories.")
        
        selected_colls = st.multiselect("Select colleges to compare:", 
                                       st.session_state.results_df['College'].tolist(), 
                                       max_selections=2)
        
        if selected_colls:
            fig = go.Figure()
            colors = ['#007bff', '#e83e8c']
            forecast_notes = []
            
            for i, c_name in enumerate(selected_colls):
                h = df[(df['CollegeName'] == c_name) & (df['CourseName'] == u_cour) & 
                       (df['Base_Category'] == u_cat) & (df['Region'] == u_reg)].sort_values('Year')
                
                if not h.empty:
                    # History
                    fig.add_trace(go.Scatter(x=h['Year'], y=h['Cutoff_Rank'], name=f"{c_name} (History)",
                                             mode='lines+markers', line=dict(color=colors[i], width=2)))
                    
                    # Forecast
                    p_val = st.session_state.results_df[st.session_state.results_df['College'] == c_name]["Avg Cutoff"].values[0]
                    fig.add_trace(go.Scatter(x=[2025, 2026], y=[h['Cutoff_Rank'].iloc[-1], p_val],
                                             mode='lines+markers+text',
                                             name=f"{c_name} (2026 Forecast)", 
                                             text=["", f"<b>{int(p_val)}</b>"],
                                             textposition="top center",
                                             line=dict(color=colors[i], width=4, dash='dash')))
                    
                    forecast_notes.append(f"AI predicts **{c_name}** 2026 cutoff around: **{int(p_val)}**")
                else:
                    st.warning(f"Not enough historical data to generate a forecast for {c_name}.")

            fig.update_layout(title="Cutoff Comparison & 2026 Forecast", 
                              xaxis=dict(tickmode='linear', dtick=1),
                              yaxis_title="Rank", 
                              hovermode="x unified")
            st.plotly_chart(fig, use_container_width=True)
            
            # RE-ADDED: The Success messages for the forecasts
            for note in forecast_notes:
                st.success(note)
        else:
            st.info("Pick a college from the list above to view the trend graph.")

# Display Top 5 Best Fit
if st.session_state.top_5_df is not None:
    if isinstance(st.session_state.top_5_df, str):
        st.error("Could not find suitable matches across branches.")
    else:
        st.success("🎯 Here are the Top 5 best colleges you can get based on your rank!")
        st.table(st.session_state.top_5_df[["College", "Branch", "Est. Cutoff"]])

# FEATURE 3: User Feedback Loop
st.markdown("---")
st.write("### 💬 Was this prediction helpful?")
col_f1, col_f2 = st.columns([1, 5])
with col_f1:
    if st.button("👍 Yes"): st.toast("Thanks for your feedback!")
with col_f2:
    if st.button("👎 No"): st.toast("We'll work on improving the AI.")

st.caption("Note: Predictions are based on historical data trends and AI modeling. Actual 2026 cutoffs may vary.")
