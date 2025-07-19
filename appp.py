import streamlit as st
import pandas as pd
import joblib

st.markdown("""
    <style>
        /* Background for the whole app */
        .stApp {
            background-image: url('https://www.transparenttextures.com/patterns/connected.png');
            background-size: cover;
            font-family: 'Segoe UI', sans-serif;
        }

        /* Header text */
        h1, h2, h3, h4 {
            font-weight: 700;
            color: #003865;
        }

        /* Input section box */
        .input-container {
            background-color: #ffffffcc;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }

        /* Styled table header */
        thead tr th {
            font-weight: bold;
            background-color: #003865;
            color: white;
        }

        /* Predict button style */
        button[kind="primary"] {
            background-color: #0077b6;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)



st.markdown("""
    <h1 style="text-align:center; font-weight:bold; color:#003865; padding:20px;">
        💼 Salary Class Prediction Dashboard
    </h1>
""", unsafe_allow_html=True)


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
st.markdown('<div class="input-container">', unsafe_allow_html=True)

age = st.sidebar.slider("Age", 18, 65, 30)
st.markdown('</div>', unsafe_allow_html=True)

education = st.sidebar.selectbox("Education Level", ["Bachelor's", "Master's", "PhD", "Diploma"])
occupation = st.sidebar.selectbox("Job Title", [
    "AI Researcher", "ML Engineer", "Software Engineer", "System Admin", "Data Engineer",
    "Data Scientist", "Web Developer", "Product Manager", "Data Analyst", "DevOps Engineer"
])
experience_level = st.sidebar.selectbox("experience_level", ["Entry","Mid","Senior"])
employment_type = st.sidebar.selectbox("employment_type", ["Contract","Full-time","Part-time"])
location = st.sidebar.selectbox("location", ["Chicago","New York","Chicago","Remote","Los Angeles","Seattle","Boston","Austin","San Jose","San Francisco"])
company_size = st.sidebar.selectbox("company_size", ["Large","Medium","Small"])
department = st.sidebar.selectbox("department", ["Operations","R&D","HR","IT","Analytics","Product",])
primary_skill = st.sidebar.selectbox("primary_skill", ["Project Management","Machine Learning","Cloud","Python","Java","DevOps","C++","Deep Learning","Data Visualization","SQL"])

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

st.markdown(f"""
    <style>
        .stApp {{
            background-image: url('{background_url}');
            background-size: cover;
            font-family: 'Segoe UI', sans-serif;
        }}
        h1, h2, h3, h4 {{
            font-weight: 700;
            color: #003865;
        }}
        .input-container {{
            background-color: #ffffffcc;
            padding: 20px;
            border-radius: 12px;
            box-shadow: 0 0 12px rgba(0,0,0,0.1);
        }}
        thead tr th {{
            font-weight: bold;
            background-color: #003865;
            color: white;
        }}
        button[kind="primary"] {{
            background-color: #0077b6;
            color: white;
            font-weight: bold;
            border-radius: 10px;
        }}
    </style>
""", unsafe_allow_html=True)


st.table(input_display_df)

# Predict button
if st.button("Predict Salary Class"):
    prediction = model.predict(input_encoded_df)
    decoded_salary = le_sal.inverse_transform(prediction)[0]
    st.markdown(f"""
        <div style="background-color:#d1e7dd; padding:20px; border-radius:10px; margin-top:20px;">
            <h3 style="text-align:center; color:#0f5132; font-weight:bold;">
                ✅ Predicted Salary: {decoded_salary}
            </h3>
        </div>
    """, unsafe_allow_html=True)



   

