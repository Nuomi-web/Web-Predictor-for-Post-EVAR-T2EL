import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import xgboost as xgb
import matplotlib.pyplot as plt
from io import BytesIO

# ===============================
# 1. Page configuration
# ===============================
st.set_page_config(
    page_title="Web Predictor",
    page_icon="🩺",
    layout="wide"
)

# ===============================
# 2. Custom CSS
# ===============================
st.markdown(
    """
    <style>
    .main {
        background-color: #0e1117;
    }

    .block-container {
        padding-top: 6rem;
        padding-left: 6rem;
        padding-right: 6rem;
        max-width: 1500px;
    }

    h1 {
        color: white;
        font-size: 64px !important;
        font-weight: 800 !important;
        margin-bottom: 50px !important;
    }

    h2 {
        color: white;
        font-size: 42px !important;
        font-weight: 700 !important;
        margin-top: 30px !important;
        margin-bottom: 30px !important;
    }

    h3 {
        color: white;
        font-size: 34px !important;
        font-weight: 700 !important;
        margin-top: 40px !important;
        margin-bottom: 25px !important;
       .pred-value {
        color: white;
        font-size: 26px;
        font-weight: 600;
        margin-bottom: 60px;
    }

    [data-testid="stSidebar"] {
        background-color: #262730;
    }

    [data-testid="stSidebar"] label {
        color: white !important;
        font-size: 20px !important;
        font-weight: 600 !important;
    }

    [data-testid="stSidebar"] input {
        font-size: 22px !important;
    }

    .stRadio label {
        color: white !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ===============================
# 3. Feature names
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
# 4. Default sample values
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
# 5. Load model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("XGBoost.pkl")

model_xgb = load_model()

# ===============================
# 6. Load SHAP explainer
# ===============================
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model_xgb)

# ===============================
# 7. Sidebar input
# ===============================
st.sidebar.header("Input Features")

inputs = {}

for feature in feature_label:
    if feature in ["IMA patency", "Maximum aneurysm diameter (mm)"]:
        continue

    inputs[feature] = st.sidebar.number_input(
        label=feature,
        min_value=-1000.0,
        max_value=1000.0,
        value=float(default_values[feature]),
        step=0.01,
        format="%.9f"
    )

st.sidebar.markdown("---")

ima_options = ["No", "Yes"]

inputs["IMA patency"] = st.sidebar.radio(
    "IMA patency",
    options=ima_options,
    index=ima_options.index(default_values["IMA patency"])
)

inputs["Maximum aneurysm diameter (mm)"] = st.sidebar.number_input(
    "Maximum aneurysm diameter (mm)",
    min_value=0.0,
    max_value=200.0,
    value=float(default_values["Maximum aneurysm diameter (mm)"]),
    step=0.01,
    format="%.2f"
)

# ===============================
# 8. Encode categorical variable
# ===============================
ima_map = {
    "No": 0,
    "Yes": 1
}

inputs["IMA patency"] = ima_map[inputs["IMA patency"]]

# ===============================
# 9. Convert to DataFrame
# ===============================
input_df = pd.DataFrame([inputs])
input_df = input_df[feature_label]

# ===============================
# 10. Prediction function
# ===============================
def predict_model(model, input_df):
    """
    Compatible with:
    - xgboost.Booster
    - sklearn API XGBClassifier/XGBRegressor
    """

    if isinstance(model, xgb.Booster):
        dmatrix = xgb.DMatrix(input_df)
        pred = model.predict(dmatrix)
    else:
        if hasattr(model, "predict_proba"):
            pred = model.predict_proba(input_df)

            if pred.ndim == 2 and pred.shape[1] == 2:
                pred = pred[:, 1]
        else:
            pred = model.predict(input_df)

    return pred

# ===============================
# 11. SHAP helper
# ===============================
def get_shap_values_for_plot(explainer, input_df):
    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value

    # Binary / multiclass compatibility
    if isinstance(shap_values, list):
        shap_value_single = shap_values[-1][0]
    else:
        shap_value_single = shap_values[0]

    if isinstance(expected_value, list):
        expected_value_single = expected_value[-1]
    elif isinstance(expected_value, np.ndarray):
        if expected_value.ndim > 0:
            expected_value_single = expected_value[-1]
        else:
            expected_value_single = expected_value
    else:
        expected_value_single = expected_value

    return expected_value_single, shap_value_single

def make_shap_force_plot(explainer, input_df):
    expected_value, shap_value_single = get_shap_values_for_plot(
        explainer,
        input_df
    )

    shap.force_plot(
        expected_value,
        shap_value_single,
        input_df.iloc[0, :],
        feature_names=feature_label,
        matplotlib=True,
        show=False,
        contribution_threshold=0.135
    )

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    plt.close()
    buf.seek(0)

    return buf

# ===============================
# 12. Main simplified output
# ===============================
try:
    prediction = predict_model(model_xgb, input_df)
    pred_value = float(np.asarray(prediction).ravel()[0])

    st.markdown(
        """
        <h1>Web Predictor for Post-EVAR T2EL</h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2>Predicted Possibility of Post-EVAR T2EL</h2>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="pred-value">
            Predicted Value: {pred_value:.16f}
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h2>SHAP Force Plot</h2>
        """,
        unsafe_allow_html=True
    )

    shap_img = make_shap_force_plot(explainer, input_df)

    st.image(
        shap_img,
        use_container_width=True
    )

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
