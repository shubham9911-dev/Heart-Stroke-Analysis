import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

# Set page to wide mode for full-screen display
# st.set_page_config(layout="wide")

# Streamlit App
st.markdown("<h1 style='text-align: center;'>Heart Stroke Prediction</h1>", unsafe_allow_html=True)

# Streamlit App
# st.title("Stroke Prediction")

# Load the image
image_path = 'heart.png'

# Display the image
st.image(image_path, caption='', use_column_width=True)

# Disable warnings and set configuration option
st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
@st.cache_data
def load_data():
    df = pd.read_csv('D:/MSc_Project/Heart_Stroke/healthcare-dataset-stroke-data.csv')
    return df

df = load_data()

# Data preprocessing
df.drop('id', axis=1, inplace=True)
df['bmi'].fillna(df['bmi'].mode()[0], inplace=True)
df['age'] = pd.cut(x=df['age'], bins=[0, 12, 19, 30, 60, 100], labels=[0, 1, 2, 3, 4])

df['ever_married'].replace({'Yes': 1, 'No': 0}, inplace=True)
df['gender'].replace({'Male': 1, 'Female': 0, 'Other': 2}, inplace=True)
df['Residence_type'].replace({'Urban': 1, 'Rural': 0}, inplace=True)
df['smoking_status'].replace({'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}, inplace=True)
df['work_type'].replace({'Private': 0, 'Self-employed': 1, 'children': 2, 'Govt_job': 3, 'Never_worked': 4}, inplace=True)

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(df.drop('stroke', axis=1), df['stroke'], test_size=0.2, random_state=42)

# Modeling
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)

svm = SVC()
svm.fit(X_train, y_train)
sv_pred = svm.predict(X_test)

dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)

knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)

# Evaluation
lr_accuracy = accuracy_score(y_test, lr_pred)
sv_accuracy = accuracy_score(y_test, sv_pred)
dt_accuracy = accuracy_score(y_test, dt_pred)
knn_accuracy = accuracy_score(y_test, knn_pred)

lr_log_loss = log_loss(y_test, lr_pred)
sv_log_loss = log_loss(y_test, sv_pred)
dt_log_loss = log_loss(y_test, dt_pred)
knn_log_loss = log_loss(y_test, knn_pred)


st.write("""
## Exploratory Data Analysis
""")
st.dataframe(df.head())

st.write("""
## Correlation Matrix
""")
plt.figure(figsize=(10, 10))
sns.heatmap(df.corr(), annot=True)
corr_plot = plt.gcf()  # Get the current figure
st.pyplot(corr_plot)  # Pass the figure explicitly

st.write("""
## Model Evaluation
""")
st.write("### Logistic Regression Model")
st.write("Accuracy Score:", lr_accuracy)
st.write("Log Loss:", lr_log_loss)
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, lr_pred))

st.write("### SVM Model")
st.write("Accuracy Score:", sv_accuracy)
st.write("Log Loss:", sv_log_loss)
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, sv_pred))

st.write("### Decision Tree Model")
st.write("Accuracy Score:", dt_accuracy)
st.write("Log Loss:", dt_log_loss)
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, dt_pred))

st.write("### KNN Model")
st.write("Accuracy Score:", knn_accuracy)
st.write("Log Loss:", knn_log_loss)
st.write("Confusion Matrix:")
st.write(metrics.confusion_matrix(y_test, knn_pred))

st.write("""
## Model Comparison
""")
models = ['Logistic Regression', 'SVM', 'Decision Tree', 'KNN']
accuracy_scores = [lr_accuracy, sv_accuracy, dt_accuracy, knn_accuracy]
plt.figure(figsize=(10, 5))
plt.bar(models, accuracy_scores, color='Maroon', width=0.4)
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
model_comparison_plot = plt.gcf()  # Get the current figure
st.pyplot(model_comparison_plot)  # Pass the figure explicitly



st.write("""
## Conclusion
""")

st.write("""
 The model accuracies of SVM  i.e. 88 %, Logistic Regression and KNN are quite similar i.e. 86 %.
The accuracy of Decision Tree Classifier is 82.5 %. So, we can use any of these models to
predict the heart stroke.
According to the graphs age v/s hypertension, heart disease showing chances of stroke,
the number of person having a stroke shows dependece upon heart disease and
hypertension. But when we plot the graph of heart disease and hypertension against the
stroke, the persons with lower chances of hypertension and heart disease has increased
chances of stroke. This is a peculiar thing and needs to be investigated further. In
addition to that non somkers have higher chances of stroke than smokers. This is also a
peculiar thing and needs to be investigated further. However person having BMI
between20 to 50 have higher chances of stroke.
Last but not least other features such as martial status, residence type as well as work
type are showing effect on the chances of stroke.
""")

# Load the image
image_path = 'heart1.png'

# Display the image
st.image(image_path, caption='', use_column_width=True)
