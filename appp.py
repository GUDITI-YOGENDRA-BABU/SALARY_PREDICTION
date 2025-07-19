import streamlit as st
import pandas as pd
import joblib

# Load the trained model and encoders

model = joblib.load("best_model3.pkl")
le_exp = joblib.load("le_exp.pkl")
le_emp = joblib.load("le_emp.pkl")
le_job = joblib.load("le_job.pkl")
le_loc = joblib.load("le_loc.pkl")
le_comp = joblib.load("le_comp.pkl")
le_dept = joblib.load("le_dept.pkl")
le_skill = joblib.load("le_skill.pkl")
le_edu = joblib.load("le_edu.pkl")
feature_names = joblib.load("feature_names.pkl")



# Ensure feature order matches model
input_df = input_df[feature_names]


st.set_page_config(page_title="Employee Salary Classification", page_icon="💼", layout="centered")
st.title("💼 Employee Salary Prediction App")
st.markdown("Predict whether an employee earns above or below a threshold based on features.")

# Sidebar input
st.sidebar.header("Enter Employee Details")

age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "Diploma"])
occupation = st.sidebar.selectbox("Job Title", [
    "AI Researcher", "ML Engineer", "Software Engineer", "System Admin", "Data Engineer",
    "Data Scientist", "Web Developer", "Product Manager", "Data Analyst", "DevOps Engineer"
])

# Optional: make these also user inputs
experience_level = "Mid"
employment_type = "Full-time"
location = "Chicago"
company_size = "Medium"
department = "R&D"
primary_skill = "Python"

# Encode input
input_df = pd.DataFrame({
    'age': [age],
    'education_level': [le_edu.transform([education])[0]],
    'job_title': [le_job.transform([occupation])[0]],
    'experience_level': [le_exp.transform([experience_level])[0]],
    'employment_type': [le_emp.transform([employment_type])[0]],
    'location': [le_loc.transform([location])[0]],
    'company_size': [le_comp.transform([company_size])[0]],
    'department': [le_dept.transform([department])[0]],
    'primary_skill': [le_skill.transform([primary_skill])[0]]
})



st.write("### 🔍 Input Features Preview")
st.write(input_df)

# Predict
if st.button("Predict Salary Class"):
    prediction = model.predict(input_df)
    st.success(f"✅ Prediction: {prediction[0]}")
