import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Align path configurations to handle direct modular calls smoothly
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.synthetic_generator import BiomarkerDataEngine
from data.loader import GenomicLoader
from model.train import execute_training_pipeline
from agent.diagnostic_agent import IntegratedDiagnosticAgent

st.set_page_config(page_title="AI Blood microRNA Platform", layout="wide")

@st.cache_resource
def boot_system_core():
    """Initializes training matrices and constructs execution frameworks on system startup."""
    with st.spinner("Initializing Deep Learning Core & Training Tabular Transformer Engine..."):
        trained_model = execute_training_pipeline(epochs=5)
        reference_data = BiomarkerDataEngine.generate_synthetic_cohort(400)
        system_agent = IntegratedDiagnosticAgent(trained_model, reference_data)
    return system_agent, reference_data

agent, ref_df = boot_system_core()

st.title("🧬 AI Blood microRNA Multi-Cancer Detection Agent")
st.markdown("---")

# Navigation and input side panel setup
sidebar = st.sidebar
sidebar.header("🔬 Input Pipeline Controls")

input_mode = sidebar.radio(
    "Select System Action Mode:",
    ("Option B: Synthetic Generator", "Option A: Upload Ext-Data Matrix", "Bonus: Cohort Explorer")
)

active_df = None

if "Option B" in input_mode:
    size = sidebar.slider("Specimen batch size:", 40, 400, 100)
    if sidebar.button("Synthesize Active Panel Matrix"):
        active_df = BiomarkerDataEngine.generate_synthetic_cohort(size)
        st.success(f"Generated {size} balanced patient profiles.")

elif "Option A" in input_mode:
    uploaded = sidebar.file_uploader("Upload CSV Profile Matrix:", type=["csv"])
    if uploaded is not None:
        active_df = GenomicLoader.parse_input_matrix(uploaded)
        st.success("External matrix schema successfully verified.")
    else:
        st.info("Awaiting file matrix upload. You can utilize the Synthetic Generator to generate clean mock inputs.")

elif "Bonus" in input_mode:
    st.subheader("Cohort Visualization Model")
    st.markdown("### Global Profile Distributions")
    fig_cohort, ax_cohort = plt.subplots(figsize=(6, 3.5))
    for c_idx, c_name in BiomarkerDataEngine.CLASS_MAP.items():
        subset = ref_df[ref_df['label'] == c_idx].iloc[:, :4].mean(axis=1)
        ax_cohort.hist(subset, alpha=0.5, label=c_name, bins=15)
    ax_cohort.set_title("Mean Intensity Cohort Overview (First 4 Markers)")
    ax_cohort.legend()
    st.pyplot(fig_cohort)

# Execution blocks
if active_df is not None:
    st.markdown("### 🗂️ Active Patient Sample Registry (Top 5 Records)")
    st.dataframe(active_df.head(5))
    
    st.markdown("---")
    patient_idx = st.number_input(
        f"Select Target Patient Record Row Index (0 to {len(active_df)-1}):",
        min_value=0, max_value=len(active_df)-1, value=0
    )
    
    if st.button("🚀 Run Agent Diagnostic Engine"):
        record = active_df.iloc[patient_idx]
        features = record[BiomarkerDataEngine.MIRNA_PANEL].values
        actual_label = int(record["label"])
        
        # Pipeline Execution via Agent Interface
        pipeline_out = agent.execute_agent_pipeline(features)
        metrics = pipeline_out["metrics"]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Target Classifier Risk Dist")
            chart_data = pd.DataFrame({
                'Diagnostic Class': [BiomarkerDataEngine.CLASS_MAP[i] for i in range(4)],
                'Risk Score': metrics['probabilities']
            })
            st.bar_chart(data=chart_data, x='Diagnostic Class', y='Risk Score')
            
            # Print core telemetry outputs
            st.metric("Agent Prediction Result:", BiomarkerDataEngine.CLASS_MAP[metrics['predicted_class']])
            st.metric("Confirmed Ground Truth Label:", BiomarkerDataEngine.CLASS_MAP[actual_label])
            st.metric("Algorithmic Confidence:", f"{metrics['confidence'] * 100:.1f}%")
            
        with col2:
            st.markdown("### 🔍 Feature Importance (XAI Attributions)")
            attr_indices = np.argsort(np.abs(pipeline_out['attributions']))[::-1][:10]
            top_m = [BiomarkerDataEngine.MIRNA_PANEL[i] for i in attr_indices]
            top_w = [pipeline_out['attributions'][i] for i in attr_indices]
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars_color = ['#ff4d4d' if w > 0 else '#4da6ff' for w in top_w]
            ax.barh(top_m[::-1], top_w[::-1], color=bars_color[::-1])
            ax.set_xlabel("Attribution Directional Value")
            ax.set_title("Top 10 Biomarker Drivers Resonating Prediction Class")
            plt.tight_layout()
            st.pyplot(fig)
            
        st.markdown("---")
        st.header("🤖 Autonomous Clinical Agent Summary Report")
        
        with st.expander("📋 View Underlying Prompt Context Payload Structure"):
            st.code(pipeline_out["prompt"], language="text")
            
        st.markdown("### 🧾 Clinical Diagnostic Report Output")
        st.text_area(label="Official Agent Readout Log", value=pipeline_out["report"], height=250)
