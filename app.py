import joblib
import streamlit as st
import numpy as np
import os

st.set_page_config(page_title="Crop Recommender", page_icon="🌿", layout='centered')

@st.cache_resource(ttl=3600)  # Cache model loading for 1 hour
def load_model(modelfile):
    if not os.path.exists(modelfile):
        st.error("Model file not found. Please ensure 'model.joblib' is in the directory.")
        return None
    model = joblib.load(modelfile)
    return model

def validate_inputs(N, P, K, temp, humidity, ph, rainfall):
    errors = []
    if not (0 < N < 5000):
        errors.append("Nitrogen must be between 1 and 5000")
    if not (0 < P < 5000):
        errors.append("Phosphorus must be between 1 and 5000")
    if not (0 < K < 5000):
        errors.append("Potassium must be between 1 and 5000")
    if not (0 <= temp <= 60):
        errors.append("Temperature must be between 0 and 60 Celsius")
    if not (0 <= humidity <= 100):
        errors.append("Humidity must be between 0 and 100%")
    if not (3 <= ph <= 11):
        errors.append("pH must be between 3 and 11")
    if not (0 <= rainfall <= 300):
        errors.append("Rainfall must be between 0 and 300 mm")
    return errors

def main():
    st.title("Crop Recommendation 🌱")
    
    model = load_model("model1.joblib")
    if model is None:
        return

    N = st.number_input("Nitrogen", min_value=1, max_value=5000, value=100)
    P = st.number_input("Phosphorus", min_value=1, max_value=5000, value=50)
    K = st.number_input("Potassium", min_value=1, max_value=5000, value=50)
    temp = st.number_input("Temperature (Celsius)", min_value=0.0, max_value=60.0, value=25.0)
    humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=50.0)
    ph = st.number_input("pH", min_value=3.0, max_value=11.0, value=7.0)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=300.0, value=100.0)

    errors = validate_inputs(N, P, K, temp, humidity, ph, rainfall)
    if errors:
        for e in errors:
            st.warning(e)
        return

    if st.button("Predict"):
        try:
            features = np.array([N, P, K, temp, humidity, ph, rainfall]).reshape(1, -1)
            
            # Use predict_proba if available for top-N
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)[0]
                classes = model.classes_
                top_n = 1
                top_indices = probs.argsort()[-top_n:][::-1]
                st.subheader(f"Top {top_n} Crop Recommendations:")
                for idx in top_indices:
                    st.write(f"{classes[idx]} - Probability: {probs[idx]*100:.2f}%")
            else:
                prediction = model.predict(features)
                st.success(f"Recommended crop: {prediction.item().title()}")
            
            # Optional: show feature importance if available
            if hasattr(model, "named_steps"):  # If pipeline
                base_model = model.named_steps.get(list(model.named_steps.keys())[-1], None)
                if base_model and hasattr(base_model, "feature_importances_"):
                    importances = base_model.feature_importances_
                    features_names = ["N", "P", "K", "Temperature", "Humidity", "pH", "Rainfall"]
                    st.subheader("Feature Importances")
                    for name, imp in zip(features_names, importances):
                        st.write(f"{name}: {imp:.3f}")
            
        except ValueError as ve:
            st.error("Input shape or type error. Please check your inputs.")
            st.error(str(ve))
        except Exception as e:
            st.error("An unexpected error occurred during prediction.")
            st.error(str(e))

if __name__ == '__main__':
    main()
