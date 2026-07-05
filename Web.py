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

st.title("Web Predictor for Post-EVAR T2EL")

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
    "DeepFeature7": 0,
    "DeepFeature474": 0,
    "DeepFeature105": 0,
    "DeepFeature375": 0,
    "DeepFeature48": 0,
    "DeepFeature318": 0,
    "DeepFeature332": 0,
    "AAA_wavelet-HL_firstorder_Skewness": 0,
    "DeepFeature316": 0,
    "DeepFeature73": 0,
    "PVAT_wavelet-HL_firstorder_Maximum": 0,
    "PVAT_wavelet-LL_glcm_ClusterShade": 0,
    "AAA_square_gldm_DependenceNonUniformityNormalized": 0,
    "AAA_square_glcm_Imc1": 0,
    "IMA patency": "Yes",
    "Maximum aneurysm diameter (mm)": 0
}

# ===============================
# 4. Load model
# ===============================
@st.cache_resource
def load_model():
    return joblib.load("XGBoost.pkl")

model_xgb = load_model()

# ===============================
# 5. Load SHAP explainer
# ===============================
@st.cache_resource
def load_explainer(_model):
    return shap.TreeExplainer(_model)

explainer = load_explainer(model_xgb)

# ===============================
# 6. Sidebar input
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

st.sidebar.markdown("### Clinical / Morphological Features")

# IMA patency: No=0, Yes=1
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
# 7. Encode categorical variable
# ===============================
ima_map = {
    "No": 0,
    "Yes": 1
}

inputs["IMA patency"] = ima_map[inputs["IMA patency"]]

# ===============================
# 8. Convert to DataFrame and keep feature order
# ===============================
input_df = pd.DataFrame([inputs])
input_df = input_df[feature_label]

# ===============================
# 9. Prediction function
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
# 10. SHAP plotting function
# ===============================
def plot_shap_force(explainer, shap_values, input_df):
    expected_value = explainer.expected_value

    if isinstance(expected_value, list):
        expected_value = expected_value[-1]

    if isinstance(expected_value, np.ndarray):
        if expected_value.ndim > 0:
            expected_value = expected_value[-1]

    if isinstance(shap_values, list):
        shap_value_single = shap_values[-1][0]
    else:
        shap_value_single = shap_values[0]

    shap.force_plot(
        expected_value,
        shap_value_single,
        input_df.iloc[0, :],
        feature_names=feature_label,
        matplotlib=True,
        show=False,
        contribution_threshold=0.05
    )

    fig = plt.gcf()
    return fig

# ===============================
# 11. Prediction and SHAP
# ===============================
if st.sidebar.button("Predict"):
    try:
        prediction = predict_model(model_xgb, input_df)



        if np.asarray(prediction).ndim == 1:
            pred_value = float(np.asarray(prediction)[0])
        
            st.subheader("Predicted Possibility of Post-EVAR T2EL")
            st.markdown(f"**Predicted Value: {pred_value:.8f}**")

         
        else:
            pred_array = np.asarray(prediction)[0]
            pred_class = int(np.argmax(pred_array))

            result_df = pd.DataFrame({
                "Class": [f"Class {i}" for i in range(len(pred_array))],
                "Probability": pred_array
            })

            st.dataframe(result_df, use_container_width=True)

            st.markdown(
                f"""
                <span style="color:red; font-size:30px;">
                    Predicted class: {pred_class}
                </span>
                """,
                unsafe_allow_html=True
            )

       # Compute SHAP values
        explainer = shap.TreeExplainer(model_xgb)
        shap_values = explainer.shap_values(input_df)

        # 5. Display SHAP force plot
        st.subheader('SHAP Force Plot')
        shap.initjs()
        force_plot = shap.force_plot(
            explainer.expected_value, 
            shap_values[0], 
            input_df.iloc[0, :], 
            feature_names=feature_label, 
            matplotlib=True, 
            contribution_threshold=0.135
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=120)
        plt.close()

        st.image("shap_force_plot.png")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        
