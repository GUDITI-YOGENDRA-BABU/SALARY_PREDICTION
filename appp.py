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
le_sal = joblib.load("le_sal.pkl")


# Load feature names and encoders
feature_names = joblib.load("feature_names.pkl")

# User inputs
age = st.sidebar.slider("Age", 18, 65, 30)
education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "Diploma"])
occupation = st.sidebar.selectbox("Job Title", [
    "AI Researcher", "ML Engineer", "Software Engineer", "System Admin", "Data Engineer",
    "Data Scientist", "Web Developer", "Product Manager", "Data Analyst", "DevOps Engineer"
])

# Fixed example inputs (or you can make these also user selectable)
experience_level = st.sidebar.selectbox("experience_level", ["Entry","Mid","Senior"])
employment_type = st.sidebar.selectbox("employment_type", ["Contract","Full-time","Part-time"])
location = st.sidebar.selectbox("location", ["Chicago","New York","Chicago","Remote","Los Angeles","Seattle","Boston","Austin","San Jose","San Francisco"])
company_size = st.sidebar.selectbox("company_size", ["Large","Medium","Small"])
department = st.sidebar.selectbox("department", ["Operations","R&D","HR","IT","Analytics","Product",])
primary_skill = st.sidebar.selectbox("", ['Project Management","Machine Learning","Cloud","Python","Java","DevOps","C++","Deep Learning","Data Visualization","SQL"])

# 1. Display human-readable inputs
input_display_df = pd.DataFrame([{
    'age': age,
    'education_level': education,
    'job_title': occupation,
    'experience_level': experience_level,
    'employment_type': employment_type,
    'location': location,
    'company_size': company_size,
    'department': department,
    'primary_skill': primary_skill
}])

# 2. Encoded version for prediction
input_encoded_df = pd.DataFrame([{
    'age': age,
    'education_level': le_edu.transform([education])[0],
    'job_title': le_job.transform([occupation])[0],
    'experience_level': le_exp.transform([experience_level])[0],
    'employment_type': le_emp.transform([employment_type])[0],
    'location': le_loc.transform([location])[0],
    'company_size': le_comp.transform([company_size])[0],
    'department': le_dept.transform([department])[0],
    'primary_skill': le_skill.transform([primary_skill])[0]
}])

input_encoded_df = input_encoded_df[feature_names]

# Show readable inputs to user
st.write("Input Features (Readable)")
st.table(input_display_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_encoded_df)
    decoded_salary = le_sal.inverse_transform(prediction)[0]
    st.success(f"Predicted Salary : {decoded_salary}")


   

