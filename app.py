import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import matplotlib.pyplot as plt

st.set_page_config(page_title="Non-Verbal Communication Preferences", page_icon="🌍", layout="wide")

@st.cache_data
def load_data():
    return pd.read_csv('non-verbal tourist data(1).csv')

data = load_data()
X = data.drop('Type of Client', axis=1)
y = data['Type of Client']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

@st.cache_resource
def load_pipeline_and_model():
    pipeline = joblib.load('non_verbal_pipeline.pkl')  # Load the preprocessing pipeline
    model = joblib.load('non_verbal_tourist_model.pkl')  # Load the trained scikit-learn model
    return pipeline, model

pipeline, model = load_pipeline_and_model()

st.title("🌍 Non-Verbal Communication Preferences Prediction")
st.markdown("---")

st.sidebar.header("🛠️ Feature Selection")
input_data = {}
for feature in X.columns:
    unique_values = X[feature].unique().tolist()
    if len(unique_values) <= 10:
        input_data[feature] = st.sidebar.radio(f"🌿 {feature.replace('_', ' ').title()}:", unique_values)
    else:
        input_data[feature] = st.sidebar.selectbox(f"🌿 {feature.replace('_', ' ').title()}:", unique_values)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("✨ Selected Features")
    feature_df = pd.DataFrame([input_data])
    st.dataframe(feature_df.T, use_container_width=True)

with col2:
    st.subheader("🩺 Make a Prediction")

    if st.button("🔍 Predict Client Type", key="predict_button"):
        with st.spinner("Analyzing..."):
            input_df = pd.DataFrame([input_data])
            input_df = input_df.reindex(columns=X.columns)
            if input_df.isnull().values.any():
                st.warning("Input data contains NaN values. Please check your inputs.")
                st.stop()
            for column in input_df.columns:
                if input_df[column].dtype == 'object':
                    input_df[column] = input_df[column].astype(str)

            try:
                input_processed = pipeline.transform(input_df)
                prediction = model.predict_proba(input_processed)
                predicted_class_index = np.argmax(prediction)
                predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
                st.success(f"🌟 **Predicted Client Type:** {predicted_class.upper()} 🎯")
                confidence_scores = prediction[0]
                fig, ax = plt.subplots()
                ax.bar(label_encoder.classes_, confidence_scores, color='#1a76ff')
                ax.set_xlabel('Client Types')
                ax.set_ylabel('Confidence')
                ax.set_title('Prediction Confidence Scores')
                ax.set_ylim(0, 1)
                plt.xticks(rotation=90)

                st.pyplot(fig)

            except Exception as e:
                st.error(f"Error in transforming input data: {e}")
                st.stop()

st.markdown("---")
st.subheader("🔍 About This Tool")
st.write("""
🌿 This tool predicts the type of client based on non-verbal communication preferences. Choose the features in the sidebar and click '🔍 Predict Client Type' to get a prediction. The confidence scores show how confident the model is about each possible client type. Enhance your customer experience by understanding their preferences! 💼
""")

st.markdown("---")
st.markdown("Created by KISHORE")
