import streamlit as st
import pandas as pd
import plotly.express as px
import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.inference import VariableElimination
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="AI Macro-Risk Dashboard", layout="wide")

@st.cache_resource
def load_and_train_ai():
    df = pd.read_csv('macro_portfolio_data.csv')
    
    # Convert text columns to category type for pgmpy
    for col in df.columns:
        df[col] = df[col].astype('category')
        
    # Define Complex 5-Node Architecture (Unit 1)
    model = DiscreteBayesianNetwork([
        ('Global_Event', 'Interest_Rates'),
        ('Inflation_Level', 'Interest_Rates'),
        ('Global_Event', 'Market_Volatility'),
        ('Interest_Rates', 'Market_Volatility'),
        ('Market_Volatility', 'Portfolio_Strategy'),
        ('Inflation_Level', 'Portfolio_Strategy')
    ])
    
    # Parameter Estimation: Learning probabilities (Unit 3)
    model.fit(df, estimator=MaximumLikelihoodEstimator)
    infer = VariableElimination(model)
    return df, model, infer

df, model, infer = load_and_train_ai()

st.title("📊 Institutional Macro-Risk & Portfolio Dashboard")
st.markdown("A 5-Variable Probabilistic Graphical Model trained on 5,000 historical market records.")

tab1, tab2, tab3 = st.tabs(["📈 Dataset & Network View", "🎛️ AI Strategy Simulator (Inference)", "🔍 Root-Cause Analytics (MAP)"])

with tab1:
    st.header("1. Training Data & System Architecture")
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Historical Distribution (5,000 Records)")
        dist_fig = px.histogram(df, x="Portfolio_Strategy", color="Market_Volatility", 
                                barmode="group", title="Historical Strategy vs. Volatility",
                                color_discrete_sequence=px.colors.qualitative.Set2)
        st.plotly_chart(dist_fig, use_container_width=True)
        
    with col2:
        st.subheader("Learned Bayesian Network (DAG)")
        fig_net, ax = plt.subplots(figsize=(5, 3.5))
        pos = nx.spring_layout(model, seed=42)
        nx.draw(model, pos, with_labels=True, node_color="#34495e", edge_color="#7f8c8d", 
                node_size=2000, font_size=8, font_weight="bold", font_color="white", arrows=False)
        st.pyplot(fig_net)

with tab2:
    st.header("2. Dynamic AI Strategy Simulator (Exact Inference)")
    st.markdown("### 🎛️ Set Market Conditions")
    c1, c2, c3, c4 = st.columns(4)
    with c1: e_input = st.selectbox("Global Event", df['Global_Event'].unique())
    with c2: i_input = st.selectbox("Inflation Level", df['Inflation_Level'].unique())
    with c3: r_input = st.selectbox("Interest Rates", df['Interest_Rates'].unique())
    with c4: v_input = st.selectbox("Market Volatility", df['Market_Volatility'].unique())

    if st.button("Run Exact Inference Simulator", type="primary", use_container_width=True):
        evidence_dict = {
            'Global_Event': e_input, 'Inflation_Level': i_input, 
            'Interest_Rates': r_input, 'Market_Volatility': v_input
        }
        res = infer.query(['Portfolio_Strategy'], evidence=evidence_dict)
        
        strategies = res.state_names['Portfolio_Strategy']
        probabilities = [res.values[strategies.index(s)] * 100 for s in strategies]
        
        st.markdown("### 📊 Calculated Probability Distribution")
        bar_fig = px.bar(
            x=strategies, y=probabilities, 
            text=[f"{p:.1f}%" for p in probabilities],
            labels={'x': 'Recommended Strategy', 'y': 'Confidence Probability (%)'},
            title="AI Confidence in Portfolio Allocation",
            color=strategies, color_discrete_sequence=['#e74c3c', '#f1c40f', '#2ecc71']
        )
        bar_fig.update_traces(textposition='outside')
        bar_fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(bar_fig, use_container_width=True)

with tab3:
    st.header("3. MAP Inference Diagnostics")
    st.write("Input an observed outcome to calculate the most probable combination of hidden economic factors.")
    
    strat_input = st.selectbox("Observe an Institutional Strategy:", df['Portfolio_Strategy'].unique())
    
    if st.button("Decode Hidden Economic State"):
        map_res = infer.map_query(['Global_Event', 'Inflation_Level', 'Interest_Rates', 'Market_Volatility'], 
                                  evidence={'Portfolio_Strategy': strat_input})
        
        st.success(f"To force a **{strat_input}** strategy, the hidden market state is mathematically most likely:")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Global Event", map_res['Global_Event'])
        m2.metric("Inflation", map_res['Inflation_Level'])
        m3.metric("Rates", map_res['Interest_Rates'])
        m4.metric("Volatility", map_res['Market_Volatility'])