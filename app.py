import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from models.model_training import train_models, get_feature_names

st.title("🧠 Comparison Analysis for Parkinson's Disease Classification")

results, scaler = train_models("data/parkinsons.data")

st.header("📊 Classifier Performance Metrics")
for name, metrics in results.items():
    st.subheader(name)
    st.write(f"**Accuracy:** {metrics['accuracy']:.4f}")
    st.write(f"**Precision:** {metrics['precision']:.4f}")
    st.write(f"**Recall:** {metrics['recall']:.4f}")
    st.write(f"**F1 Score:** {metrics['f1']:.4f}")
    st.markdown("---")

# 📊 Model Comparison Table
st.header("📋 Model Comparison Table")

# Create a summary DataFrame
summary_df = pd.DataFrame([
    {
        "Classifier": name,
        "Accuracy": metrics["accuracy"],
        "Precision": metrics["precision"],
        "Recall": metrics["recall"],
        "F1 Score": metrics["f1"]
    }
    for name, metrics in results.items()
])

def highlight_best(s):
    is_max = s == s.max()
    return ['background-color: lightgreen' if v else '' for v in is_max]

styled_df = summary_df.style.format({
    "Accuracy": "{:.4f}",
    "Precision": "{:.4f}",
    "Recall": "{:.4f}",
    "F1 Score": "{:.4f}"
}).apply(highlight_best, subset=["Accuracy", "Precision", "Recall", "F1 Score"])

st.dataframe(styled_df, use_container_width=True)


# Metric comparison graph
def plot_metric(metric_name):
    st.subheader(f"{metric_name.capitalize()} Comparison")
    metric_values = [metrics[metric_name] for metrics in results.values()]
    classifiers = list(results.keys())
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(classifiers, metric_values, color='skyblue')
    for i, v in enumerate(metric_values):
        ax.text(i, v / 2, f"{v:.2%}", ha='center', color='black')
    ax.set_ylabel(metric_name.capitalize())
    st.pyplot(fig)

st.header("📈 Metric Comparison Graph")
metric_to_plot = st.selectbox("Choose metric to compare", ["accuracy", "precision", "recall", "f1"])
if st.button("Display"):
    plot_metric(metric_to_plot)

# Sidebar Prediction Form
st.sidebar.header("🧪 Test a Custom Sample")
with st.sidebar.form("prediction_form"):
    st.subheader("Input Voice Parameters")

    feature_names = get_feature_names()
    input_values = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", value=0.0)
        input_values.append(value)

    submit = st.form_submit_button("Predict")

    if submit:
        if submit:
            input_array = np.array(input_values).reshape(1, -1)
            scaled_input = scaler.transform(input_array)

            st.sidebar.subheader("Prediction Results")
            parkinson_votes = 0
            total_models = len(results)

            for name, metrics in results.items():
                prediction = metrics['model'].predict(scaled_input)[0]
                status = "Parkinson's" if prediction == 1 else "Healthy"
                if prediction == 1:
                    parkinson_votes += 1
                st.sidebar.write(f"**{name}**: {status}")

            # Calculate percentage and determine risk level
            parkinson_percentage = parkinson_votes / total_models
            if parkinson_percentage <= 0.33:
                risk_level = "Low"
                color = "green"
            elif parkinson_percentage <= 0.66:
                risk_level = "Moderate"
                color = "orange"
            else:
                risk_level = "High"
                color = "red"

            st.sidebar.markdown("---")
            st.sidebar.markdown(f"### 🧭 Health Risk Indicator")
            st.sidebar.markdown(
                f"<h3 style='color:{color};'>Risk Level: {risk_level}</h3>",
                unsafe_allow_html=True
            )

            st.sidebar.progress(int(parkinson_percentage * 100),
                                text=f"{int(parkinson_percentage * 100)}% Models Predict Parkinson's")
            emoji = "🟢" if risk_level == "Low" else "🟠" if risk_level == "Moderate" else "🔴"
            st.sidebar.markdown(f"## {emoji} {risk_level} Risk")

