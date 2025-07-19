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
experience_level = "Mid"
employment_type = "Full-time"
location = "Chicago"
company_size = "Medium"
department = "R&D"
primary_skill = "Python"

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
st.write("### 🔍 Input Features (Readable)")
st.table(input_display_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_encoded_df)
    st.success(f"✅ Predicted Salary Class: {prediction[0]}")
