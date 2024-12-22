import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# Set page to wide mode for full-screen display
st.set_page_config(layout="wide")

# Streamlit App
st.markdown("<h1 style='text-align: center;'>Heart Stroke Prediction</h1>", unsafe_allow_html=True)

# Load the dataset
@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('D:/MSc_Project/Data-Science-Projects-main/healthcare-dataset-stroke-data.csv')
    return df

# Data preprocessing
df = load_data()
df['bmi'].fillna(df['bmi'].mode()[0], inplace=True)
df['age'] = pd.cut(x=df['age'], bins=[0, 12, 19, 30, 60, 100], labels=[0, 1, 2, 3, 4])

df['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 2}, inplace=True)
df['Residence_type'].replace({'Urban': 1, 'Rural': 0}, inplace=True)
df['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}, inplace=True)
df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}, inplace=True)

# Model training
X = df.drop('stroke', axis=1)
y = df['stroke']

lr = LogisticRegression()
lr.fit(X, y)

svm = SVC()
svm.fit(X, y)

dt = DecisionTreeClassifier()
dt.fit(X, y)

knn = KNeighborsClassifier()
knn.fit(X, y)

# User Input
st.sidebar.header('Enter Patient Information')

gender = st.sidebar.selectbox('Gender', ['Male', 'Female', 'Other'])
age = st.sidebar.slider('Age', 0, 100, 50)
hypertension = st.sidebar.radio('Hypertension', ['No', 'Yes'])
heart_disease = st.sidebar.radio('Heart Disease', ['No', 'Yes'])
ever_married = st.sidebar.radio('Ever Married', ['No', 'Yes'])
work_type = st.sidebar.selectbox('Work Type', ['Private', 'Self-employed', 'Children', 'Govt_job', 'Never_worked'])
residence_type = st.sidebar.radio('Residence Type', ['Rural', 'Urban'])
avg_glucose_level = st.sidebar.number_input('Average Glucose Level', value=100)
bmi = st.sidebar.number_input('BMI', value=25)
smoking_status = st.sidebar.selectbox('Smoking Status', ['formerly smoked', 'never smoked', 'smokes', 'Unknown'])

# Encode user input
gender_encoded = 1 if gender == 'Male' else 0 if gender == 'Female' else 2
hypertension_encoded = 1 if hypertension == 'Yes' else 0
heart_disease_encoded = 1 if heart_disease == 'Yes' else 0
ever_married_encoded = 1 if ever_married == 'Yes' else 0
work_type_encoded = ['Private', 'Self-employed', 'Children', 'Govt_job', 'Never_worked'].index(work_type)
residence_type_encoded = 1 if residence_type == 'Urban' else 0
smoking_status_encoded = ['formerly smoked', 'never smoked', 'smokes', 'Unknown'].index(smoking_status)

# Predict
input_data = [[gender_encoded, age, hypertension_encoded, heart_disease_encoded, ever_married_encoded,
               work_type_encoded, residence_type_encoded, avg_glucose_level, bmi, smoking_status_encoded]]
lr_prediction = lr.predict(input_data)
svm_prediction = svm.predict(input_data)
dt_prediction = dt.predict(input_data)
knn_prediction = knn.predict(input_data)

# Display predictions
st.write("""
## Predictions
""")
st.write("### Logistic Regression Model Prediction:", 'Stroke' if lr_prediction[0] == 1 else 'No Stroke')
st.write("### SVM Model Prediction:", 'Stroke' if svm_prediction[0] == 1 else 'No Stroke')
st.write("### Decision Tree Model Prediction:", 'Stroke' if dt_prediction[0] == 1 else 'No Stroke')
st.write("### KNN Model Prediction:", 'Stroke' if knn_prediction[0] == 1 else 'No Stroke')
