import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt

# ===============================
# 1. Page configuration
# ===============================
st.set_page_config(
    page_title="Web Predictor",
    page_icon="🩺",
    layout="wide"
)

st.title("Web Predictor")
st.markdown("### Web Predictor for Post-EVAR T2EL")

# ===============================
# 2. Feature names
# ===============================
feature_label = [
    "DeepFeature7",
    "DeepFeature474",
    "DeepFeature105",
    "DeepFeature375",
    "DeepFeature48",
    "DeepFeature318",
    "DeepFeature332",
    "AAA_wavelet-HL_firstorder_Skewness",
    "DeepFeature316",
    "DeepFeature73",
    "PVAT_wavelet-HL_firstorder_Maximum",
    "PVAT_wavelet-LL_glcm_ClusterShade",
    "AAA_square_gldm_DependenceNonUniformityNormalized",
    "AAA_square_glcm_Imc1",
    "IMA patency",
    "Maximum aneurysm diameter (mm)"
]

# ===============================
# 3. Default sample values
# IMA patency: No = 0, Yes = 1
# ===============================
default_values = {
    "DeepFeature7": -0.194331593,
    "DeepFeature474": 0.567902798,
    "DeepFeature105": 0.837165944,
    "DeepFeature375": -0.640949675,
    "DeepFeature48": -0.321764867,
    "DeepFeature318": 0.6806,
    "DeepFeature332": 0.76790509,
    "AAA_wavelet-HL_firstorder_Skewness": -1.511827266,
    "DeepFeature316": -1.518833969,
    "DeepFeature73": -0.606873453,
    "PVAT_wavelet-HL_firstorder_Maximum": 1.813605379,
    "PVAT_wavelet-LL_glcm_ClusterShade": 0.064459839,
    "AAA_square_gldm_DependenceNonUniformityNormalized": -0.771297444,
    "AAA_square_glcm_Imc1": 10.20636933,
    "IMA patency": "Yes",
    "Maximum aneurysm diameter (mm)": 48.06
}

# ===============================
# 4. Streamlit page
# ===============================
st.title('Web Predictor for Occult LNM in Patients with HNSCC')
st.sidebar.header('Input Features')

# ===============================
# 5. Input features
# ===============================
inputs = {}

continuous_features = [
    'IC_DL_57',
    'VMI_original_glszm_SmallAreaHighGrayLevelEmphasis',
    'IC_wavelet-LLH_glszm_ZoneEntropy',
    'Zeff_DL_91',
    'VMI_DL_45',
    'IC_DL_121',
    'Zeff_wavelet-LLH_gldm_LargeDependenceLowGrayLevelEmphasis',
    'VMI_DL_139',
    'Zeff_DL_137',
    'IC_wavelet-LHL_firstorder_Skewness',
    'Zeff_wavelet-LHH_glcm_Idn',
    'VMI_wavelet-LLH_glrlm_GrayLevelNonUniformity',
    'VMI_wavelet-HHH_glcm_Correlation',
    'PEI_wavelet-LLH_glszm_SmallAreaEmphasis',
    'IC_wavelet-LLL_glszm_SizeZoneNonUniformity',
    'PEI_DL_243'
]

# Continuous variables
for feature in continuous_features:
    inputs[feature] = st.sidebar.number_input(
        label=feature,
        min_value=-100.0,
        max_value=100.0,
        value=float(default_values[feature]),
        step=0.01,
        format="%.6f"
    )

# ===============================
# Histological grade
# 0: Well
# 1: Moderate
# 2: Poor
# ===============================
histological_options = ['Well', 'Moderate', 'Poor']

histological_grade = st.sidebar.selectbox(
    'Histological grade',
    options=histological_options,
    index=int(default_values['Histological grade'])
)

histological_map = {
    'Well': 0,
    'Moderate': 1,
    'Poor': 2
}

inputs['Histological grade'] = histological_map[histological_grade]

# ===============================
# Clinical T stage
# 0: T1-2
# 1: T3-4
# ===============================
clinical_t_options = ['T1-2', 'T3-4']

clinical_t_stage = st.sidebar.selectbox(
    'Clinical T stage',
    options=clinical_t_options,
    index=int(default_values['Clinical T stage'])
)

clinical_t_map = {
    'T1-2': 0,
    'T3-4': 1
}

inputs['Clinical T stage'] = clinical_t_map[clinical_t_stage]

# Convert input values into DataFrame
input_df = pd.DataFrame([inputs])

# Ensure the order is exactly the same as training
input_df = input_df[feature_label]

# 6. Prediction
# ===============================
if st.sidebar.button('Predict'):
    try:

        input_data = xgb.DMatrix(input_df.values)

        prediction = model_xgb.predict(
            input_data,
            validate_features=False
        )[0]

        st.subheader('Predicted probability of occult LNM')

        st.markdown(
            f"""
            <p style="font-size:18px; font-weight:bold;">
                Predicted probability:
                <span style="color:red;">{prediction:.6f}</span>
            </p>
            """,
            unsafe_allow_html=True
        )

        # ===============================
        # 7. SHAP explanation
        # ===============================
        st.subheader('SHAP Force Plot')

        explainer = shap.TreeExplainer(model_xgb)

        # Use values only to avoid feature name mismatch in SHAP
        shap_values = explainer.shap_values(input_df.values)

        plt.figure()

        shap.force_plot(
            explainer.expected_value,
            shap_values[0],
            input_df.iloc[0, :],
            feature_names=feature_label,
            matplotlib=True,
            contribution_threshold=0.11,
            show=False
        )

        plt.savefig(
            "shap_force_plot.png",
            bbox_inches='tight',
            dpi=300
        )

        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
