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
# 2. Page title
# ===============================
st.title("Web Predictor for Post-EVAR T2EL")

# ===============================
# 3. Feature names
# ===============================
default_values = {
    "DeepFeature7": None,
    "DeepFeature474": None,
    "DeepFeature105": None,
    "DeepFeature375": None,
    "DeepFeature48": None,
    "DeepFeature318": None,
    "DeepFeature332": None,
    "AAA_wavelet-HL_firstorder_Skewness": None,
    "DeepFeature316": None,
    "DeepFeature73": None,
    "PVAT_wavelet-HL_firstorder_Maximum": None,
    "PVAT_wavelet-LL_glcm_ClusterShade": None,
    "AAA_square_gldm_DependenceNonUniformityNormalized": None,
    "AAA_square_glcm_Imc1": None,
    "IMA patency": None,
    "Maximum aneurysm diameter (mm)": None
}

# ===============================
# 4. Default sample values
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
    "Maximum aneurysm diameter (mm)": ""
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

st.sidebar.markdown("### Radiomics / Deep Features")

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

st.sidebar.markdown("### Clinical/Radiological Features")

ima_options = ["No", "Yes"]

inputs["IMA patency"] = st.sidebar.radio(
    "IMA patency",
    options=ima_options,
    index=ima_options.index(default_values["IMA patency"]),
    help="No = 0, Yes = 1"
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
inputs["IMA patency"] = 1 if inputs["IMA patency"] == "Yes" else 0

# ===============================
# 9. Convert to DataFrame and keep feature order
# ===============================
input_df = pd.DataFrame([inputs])[feature_label]

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

    return np.asarray(pred).ravel()

# ===============================
# 11. SHAP functions
# ===============================
def get_single_shap_values(explainer, input_df):
    shap_values = explainer.shap_values(input_df)
    expected_value = explainer.expected_value

    # Handle binary / multiclass SHAP values
    if isinstance(shap_values, list):
        shap_value_single = shap_values[-1][0]
    else:
        shap_value_single = shap_values[0]

    if isinstance(expected_value, list):
        expected_value_single = expected_value[-1]
    elif isinstance(expected_value, np.ndarray):
        expected_value_single = expected_value.ravel()[-1]
    else:
        expected_value_single = expected_value

    return expected_value_single, shap_value_single

def create_shap_force_plot(explainer, input_df):
    expected_value, shap_value_single = get_single_shap_values(
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
# 12. Prediction and SHAP output
# ===============================
if st.sidebar.button("Predict"):
    try:
        prediction = predict_model(model_xgb, input_df)
        pred_value = float(prediction[0])

        st.markdown("## Predicted Possibility of Post-EVAR T2EL")

        st.markdown(
            f"""
            <div style="
                font-size:28px;
                font-weight:600;
                margin-top:20px;
                margin-bottom:50px;
            ">
                Predicted Value: {pred_value:.8f}
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown("## SHAP Force Plot")
        shap_img = create_shap_force_plot(explainer, input_df)
        
        st.image(
            shap_img,
            use_column_width=True
)

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
