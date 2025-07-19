import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_model3.pkl")
le_edu = joblib.load("label_encoder_edu.pkl")   
le_occ = joblib.load("label_encoder_occ.pkl")

st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")

st.title("💼 Employee Salary perdiction App")
st.markdown("Predict whether an employee earns based on input features.")

# Sidebar inputs (these must match your training feature columns)
st.sidebar.header("Input Employee Details")

# ✨ Replace these fields with your dataset's actual input columns
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("education_level", [
    "Bachelor's","Master's","PhD","Diploma"
])
occupation = st.sidebar.selectbox("job_title", [
   "AI Researcher","ML Engineer","Software Engineer","System Admin","Data Engineer","Data Scientist",
"Web Developer","Product Manager","Data Analyst","DevOps Engineer"
])     


# Build input DataFrame (⚠️ must match preprocessing of your training data)
input_df = pd.DataFrame({
    'age': [age],
    'education_level': [le_edu.transform([education])[0]],
    'job_title': [le_job.transform([occupation])[0]],
    # Add defaults or more user inputs for other columns
    'experience_level': [0],  # or a dropdown
    'employment_type': [0],
    'location': [0],
    'company_size': [0],
    'department': [0],
    'primary_skill': [0]
})


st.write("### 🔎 Input Data")
st.write(input_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")

# Batch prediction
st.markdown("---")
st.markdown("#### 📂 Batch Prediction")
uploaded_file = st.file_uploader("Upload a CSV file for batch prediction", type="csv")

if uploaded_file is not None:
    batch_data = pd.read_csv(uploaded_file)
    st.write("Uploaded data preview:", batch_data.head())
    batch_preds = model.predict(batch_data)
    batch_data['PredictedClass'] = batch_preds
    st.write("✅ Predictions:")
    st.write(batch_data.head())
    csv = batch_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Predictions CSV", csv, file_name='predicted_classes.csv', mime='text/csv')
