import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# --- SETTINGS ---
st.set_page_config(page_title="Network Attack Detector", page_icon="🛡️", layout="wide")

MODEL_PATH = 'models/random_forest_baseline.pkl'
SCALER_PATH = 'models/scaler.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

# --- LOAD ASSETS ---
@st.cache_resource
def load_assets():
    if not os.path.exists(MODEL_PATH):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    le = joblib.load(ENCODER_PATH)
    return model, scaler, le

# --- UI ---
st.title("🛡️ IDS Network Attack Detection Dashboard")
st.markdown("""
This dashboard uses a **Random Forest** model trained on the CICIDS2017 dataset to identify network attacks.
Upload a CSV file containing network flow features to begin analysis.
""")

model, scaler, le = load_assets()

if model is None:
    st.error("❌ Model files not found. Please run the training script first.")
    st.stop()

# Sidebar
with st.sidebar:
    st.header("Settings")
    uploaded_file = st.file_uploader("Upload CSV Network Traffic", type=["csv"])
    st.info("The CSV should have features compatible with CICIDS2017.")

if uploaded_file is not None:
    # 1. Load Data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Raw Data Preview")
    st.dataframe(df.head(10), use_container_width=True)
    
    # 2. Preprocessing
    # Ensure consistency with model training
    REDUNDANT = ['Fwd Packet Length Mean', 'Bwd Packet Length Mean', 'Average Packet Size', 'Fwd IAT Mean', 'Attack Type']
    X = df.drop(columns=[col for col in REDUNDANT if col in df.columns])
    
    # Check if we have missing columns
    if X.empty:
        st.error("CSV doesn't contain valid features.")
    else:
        # Run Prediction
        with st.status("Analyzing traffic...", expanded=True) as status:
            st.write("Scaling features...")
            X_scaled = scaler.transform(X)
            st.write("Predicting Attack Types...")
            preds = model.predict(X_scaled)
            labels = le.inverse_transform(preds)
            df['Predicted_Attack'] = labels
            status.update(label="Analysis Complete!", state="complete", expanded=False)

        # 3. Visualization
        st.divider()
        st.subheader("📊 Analysis Results")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Stats
            total_flows = len(df)
            attacks = df[df['Predicted_Attack'] != 'Normal Traffic']
            attack_count = len(attacks)
            attack_per = (attack_count / total_flows) * 100
            
            st.metric("Total Flows Analyzed", f"{total_flows:,}")
            st.metric("Attacks Detected", f"{attack_count:,}", f"{attack_per:.2f}%", delta_color="inverse")
            
            # Pie Chart
            fig_pie = px.pie(df, names='Predicted_Attack', title='Overall Traffic Distribution',
                             color_discrete_sequence=px.colors.qualitative.Set3)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col2:
            # Bar Chart for Attacks only
            if attack_count > 0:
                attack_dist = attacks['Predicted_Attack'].value_counts().reset_index()
                attack_dist.columns = ['Attack Type', 'Count']
                fig_bar = px.bar(attack_dist, x='Attack Type', y='Count', 
                                 title='Detailed Break-down of Detected Attacks',
                                 color='Attack Type', template='plotly_dark')
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.success("✅ No attacks detected in this batch!")

        # 4. Detailed Table
        st.subheader("🔍 Detailed Prediction Table")
        # Let user filter by label
        filter_label = st.multiselect("Filter by Label", options=le.classes_, default=list(le.classes_))
        filtered_df = df[df['Predicted_Attack'].isin(filter_label)]
        
        st.dataframe(filtered_df, use_container_width=True)
        
        # Download button
        csv = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv,
            file_name=f"attack_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime='text/csv',
        )
else:
    # Home state
    st.info("Please upload a CSV file in the sidebar to start the detection.")
    
    # Feature Importance Info
    if st.checkbox("Show Model Technical Details"):
        st.write("### Top Contributing Features")
        feat_imp = pd.Series(model.feature_importances_, index=X.columns if 'X' in locals() else range(48))
        top10 = feat_imp.nlargest(10).sort_values(ascending=True)
        fig_imp = px.bar(x=top10.values, y=top10.index, orientation='h', 
                         labels={'x': 'Importance Score', 'y': 'Feature Name'},
                         title='Random Forest Feature Importance')
        st.plotly_chart(fig_imp, use_container_width=True)

st.divider()
st.caption("Developed by Antigravity AI | Technology: Python, Streamlit, Scikit-learn, Plotly")
