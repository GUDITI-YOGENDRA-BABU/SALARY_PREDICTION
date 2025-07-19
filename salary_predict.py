import pandas as pd
data = pd.read_csv(r"C:\Users\gudit\Downloads\PYTHON NOTES\PYTHON WORK SPACE\SALARY PREDICT-1\salaries.csv")
print(data)
data.drop(columns=['employee_id'],inplace=True)
data.drop(columns=['performance_rating'],inplace=True)
print(data.isna().sum())
print(data.job_title.value_counts())
print(data.experience_level.value_counts())
print(data.education_level.value_counts())
print(data.employment_type.value_counts())
print(data.location.value_counts())
print(data.company_size.value_counts())
print(data.salary.value_counts())
print(data.primary_skill.value_counts())
print(data.department.value_counts())
print(data.age.value_counts())

from sklearn.preprocessing import LabelEncoder
le_exp = LabelEncoder()
le_emp = LabelEncoder()
le_job = LabelEncoder()
le_loc = LabelEncoder()
le_comp = LabelEncoder()
le_edu = LabelEncoder()
le_skill = LabelEncoder()
le_dept = LabelEncoder()
le_sal = LabelEncoder()

data['experience_level'] = le_exp.fit_transform(data['experience_level'])
data['employment_type'] = le_emp.fit_transform(data['employment_type'])
data['job_title'] = le_job.fit_transform(data['job_title'])
data['location'] = le_loc.fit_transform(data['location'])
data['company_size'] = le_comp.fit_transform(data['company_size'])
data['education_level'] = le_edu.fit_transform(data['education_level'])
data['primary_skill'] = le_skill.fit_transform(data['primary_skill'])
data['department'] = le_dept.fit_transform(data['department'])
data['salary'] = le_sal.fit_transform(data['salary'])

print(data)
print(data.shape)
import matplotlib.pyplot as plt
plt.boxplot(data['age'])
plt.show()
plt.boxplot(data['job_title'])
plt.show()
plt.boxplot(data['experience_level'])
plt.show()
plt.boxplot(data['education_level'])
plt.show()
plt.boxplot(data['employment_type'])
plt.show()
plt.boxplot(data['location'])
plt.show()
plt.boxplot(data['department'])
plt.show()
plt.boxplot(data['primary_skill'])
plt.show()
plt.boxplot(data['company_size'])
plt.show()
x=data.drop(columns=['salary'])
y=data['salary']
print(x)
print(y)

from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

models = {
    "KNN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

for name, model in models.items():
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])
    
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc
    print(f"{name} Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
   
import matplotlib.pyplot as plt
plt.bar(results.keys(), results.values(), color='skyblue')
plt.ylabel('Accuracy Score')
plt.title('Model Comparison')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define models
models = {
    "KNN": KNeighborsClassifier(),
    "GradientBoosting": GradientBoostingClassifier()
}

results = {}

# Train and evaluate
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f"{name}: {acc:.4f}")

# Get best model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
print(f"\n✅ Best model: {best_model_name} with accuracy {results[best_model_name]:.4f}")

# Save the best model
joblib.dump(best_model, "best_model3.pkl")
joblib.dump(LabelEncoder, "label_encoder3.pkl")
joblib.dump(Pipeline, 'salary_prediction_model3.pkl')
le_job = joblib.load("le_job.pkl")
le_exp = joblib.load("le_exp.pkl")
le_emp = joblib.load("le_emp.pkl")
le_loc = joblib.load("le_loc.pkl")
le_comp = joblib.load("le_comp.pkl")
le_dept = joblib.load("le_dept.pkl")
le_skill = joblib.load("le_skill.pkl")



print("✅ Saved best model as best_model3.pkl")
