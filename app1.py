# streamlit_display.py
import streamlit as st
from pyspark.sql import SparkSession
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import metrics
from sklearn.metrics import roc_curve
from sklearn.metrics import recall_score, confusion_matrix, precision_score, f1_score, accuracy_score, classification_report
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')
st.set_option('deprecation.showPyplotGlobalUse', False)
# Initialize Spark session
spark = SparkSession.builder.appName("StreamlitDisplayApp").getOrCreate()
st. set_page_config(layout="wide") 
# Read data into Spark DataFrame
data_path = "/Users/aameerkhan/Desktop/sem7prog/telecom.csv"  # Update this with the actual path
df = spark.read.csv(data_path, header=True, inferSchema=True)

df = spark.read.csv(data_path, header=True, inferSchema=True)

# Data preprocessing
df = df.drop('customerID')
df = df.fillna(df.agg({'TotalCharges': 'mean'}).collect()[0][0])
df = df.withColumn("SeniorCitizen", df["SeniorCitizen"].cast("string"))

# Streamlit app
st.title("Data Preprocessing and Visualization using Streamlit")

# Display the preprocessed DataFrame using Streamlit
st.subheader("Preprocessed DataFrame")
st.dataframe(df.limit(5).toPandas())

# Describe numerical columns
st.subheader("Numerical Columns Summary")
numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
st.dataframe(df.select(numerical_cols).describe().toPandas())

# Visualize Gender and Churn Distributions
st.subheader("Gender and Churn Distributions")
g_labels = ['Male', 'Female']
c_labels = ['No', 'Yes']

# Count values for gender and churn
gender_counts = df.groupBy('gender').count().toPandas()
churn_counts = df.groupBy('Churn').count().toPandas()

# Create subplots: use 'domain' type for Pie subplot
fig = make_subplots(rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
fig.add_trace(go.Pie(labels=g_labels, values=gender_counts['count'], name="Gender"),
              1, 1)
fig.add_trace(go.Pie(labels=c_labels, values=churn_counts['count'], name="Churn"),
              1, 2)

# Use `hole` to create a donut-like pie chart
fig.update_traces(hole=.4, hoverinfo="label+percent+name", textfont_size=16)

fig.update_layout(
    title_text="Gender and Churn Distributions",
    # Add annotations in the center of the donut pies.
    annotations=[dict(text='Gender', x=0.16, y=0.5, font_size=20, showarrow=False),
                 dict(text='Churn', x=0.84, y=0.5, font_size=20, showarrow=False)])

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)


st.title("Churn Distribution w.r.t Gender: Male (M), Female (F)")

# Data
labels = ["Churn: Yes", "Churn: No"]
values = [1869, 5163]
labels_gender = ["F", "M", "F", "M"]
sizes_gender = [939, 930, 2544, 2619]
colors = ['#ff6666', '#66b3ff']
colors_gender = ['#c2c2f0', '#ffb3e6', '#c2c2f0', '#ffb3e6']
explode = (0.3, 0.3)
explode_gender = (0.1, 0.1, 0.1, 0.1)
textprops = {"fontsize": 15}

# Create a figure and axis
fig, ax = plt.subplots(figsize=(6, 6))

# Plot Churn data
ax.pie(values, labels=labels, autopct='%1.1f%%', pctdistance=1.08, labeldistance=0.8, colors=colors,
       startangle=90, frame=True, explode=explode, radius=10, textprops=textprops, counterclock=True)

# Plot Gender data
ax.pie(sizes_gender, labels=labels_gender, colors=colors_gender, startangle=90,
       explode=explode_gender, radius=7, textprops=textprops, counterclock=True)

# Draw circle
centre_circle = plt.Circle((0, 0), 5, color='black', fc='white', linewidth=0)
fig.gca().add_artist(centre_circle)

# Set title
ax.set_title('Churn Distribution w.r.t Gender: Male(M), Female(F)', fontsize=15, y=1.1)

# Equal aspect ratio ensures that pie is drawn as a circle.
ax.axis('equal')

# Display the plot using Streamlit
st.pyplot(fig)

st.title("Customer Contract Distribution")

# Convert Spark DataFrame to Pandas DataFrame for Plotly Express
df_pd = df.select("Churn", "Contract").toPandas()

# Create histogram using Plotly Express
fig = px.histogram(df_pd, x="Churn", color="Contract", barmode="group", title="Customer contract distribution")
fig.update_layout(width=700, height=500, bargap=0.1)

# Display the plot using Streamlit
st.plotly_chart(fig)
# Streamlit app
st.title("Internet Service and Churn Analysis")

# Display unique values of InternetService column
st.subheader("Unique Internet Service Values")
unique_internet_services = df.select("InternetService").distinct().toPandas()["InternetService"].tolist()
st.write(unique_internet_services)

# Analyze Churn for Male customers based on InternetService
st.subheader("Churn Analysis for Male Customers based on InternetService")
male_churn_internet = df.filter(df["gender"] == "Male").groupBy("InternetService", "Churn").count().toPandas()
st.write(male_churn_internet)


# Streamlit app
st.title("Churn Distribution w.r.t. Internet Service and Gender")

# Create a Plotly Figure
fig = go.Figure()

fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
         ["Female", "Male", "Female", "Male"]],
    y = [965, 992, 219, 240],
    name = 'DSL',
))

fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
         ["Female", "Male", "Female", "Male"]],
    y = [889, 910, 664, 633],
    name = 'Fiber optic',
))

fig.add_trace(go.Bar(
    x = [['Churn:No', 'Churn:No', 'Churn:Yes', 'Churn:Yes'],
         ["Female", "Male", "Female", "Male"]],
    y = [690, 717, 56, 57],
    name = 'No Internet',
))

# Update layout
fig.update_layout(
    title_text="<b>Churn Distribution w.r.t. Internet Service and Gender</b>",
    xaxis_title="Churn Status",
    yaxis_title="Count"
)

# Display the Plotly figure using Streamlit
st.plotly_chart(fig)



# Define color map for Churn values


st.title("Dependents Distribution")

# Define color map for Churn values
color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}

# Create histogram using Plotly Express
fig = px.histogram(df.toPandas(), x="Churn", color="Dependents", barmode="group",
                   title="Dependents distribution", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)

# Display the plot using Streamlit
st.plotly_chart(fig)
st.title("Churn Distribution w.r.t. Senior Citizen")

# Define color map for Churn values
color_map = {"Yes": '#00CC96', "No": '#B6E880'}

# Create histogram using Plotly Express
fig = px.histogram(df.toPandas(), x="Churn", color="SeniorCitizen",
                   title="Churn distribution w.r.t. Senior Citizen", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)

# Display the plot using Streamlit
st.plotly_chart(fig)
st.title("Churn Distribution w.r.t. Online Security")

# Define color map for Churn values
color_map = {"Yes": "#FF97FF", "No": "#AB63FA"}

# Create histogram using Plotly Express
fig = px.histogram(df.toPandas(), x="Churn", color="OnlineSecurity", barmode="group",
                   title="Churn w.r.t. Online Security", color_discrete_map=color_map)
fig.update_layout(width=700, height=500, bargap=0.1)

# Display the plot using Streamlit
st.plotly_chart(fig)
st.title("Correlation Heatmap")

# Convert Spark DataFrame to Pandas DataFrame
df_pd = df.toPandas()

plt.figure(figsize=(25, 10))

corr = df_pd.apply(lambda x: pd.factorize(x)[0]).corr()

mask = np.triu(np.ones_like(corr, dtype=bool))

ax = sns.heatmap(corr, mask=mask, xticklabels=corr.columns, yticklabels=corr.columns, annot=True, linewidths=.2, cmap='coolwarm', vmin=-1, vmax=1)
st.pyplot()
st.title("Feature Distribution Analysis")

# Drop 'Churn' column from PySpark DataFrame


# Convert PySpark DataFrame to Pandas DataFrame for visualization
df_pd = df.toPandas()

# Split data
X = df_pd.drop(columns=['Churn'])
y = df_pd['Churn'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

# Define distribution plot function


# Standardize numeric features
st.title("Column Categorization")

# Numeric columns
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

# Categorical columns for one-hot encoding
cat_cols_ohe = ['PaymentMethod', 'Contract', 'InternetService']

# Categorical columns for label encoding
cat_cols_le = list(set(X_train.columns) - set(num_cols) - set(cat_cols_ohe))

# Display categories

st.title("Standard Scaling")

# Convert Spark DataFrame to Pandas DataFrame
df_pd = df.toPandas()

# Numeric columns
num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

# Assuming you have X_train and X_test dataframes
# Replace this with your actual data preparation process

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=40, stratify=y)

# Convert "TotalCharges" column to float
X_train["TotalCharges"] = pd.to_numeric(X_train["TotalCharges"], errors='coerce')
X_test["TotalCharges"] = pd.to_numeric(X_test["TotalCharges"], errors='coerce')

# Initialize a StandardScaler
scaler = StandardScaler()

# Check for missing values
missing_cols_train = X_train[num_cols].columns[X_train[num_cols].isnull().any()]
missing_cols_test = X_test[num_cols].columns[X_test[num_cols].isnull().any()]

if not missing_cols_train.empty or not missing_cols_test.empty:
    st.write("Missing values found. Replacing with mean...")
    
    # Replace missing values with mean
    X_train[num_cols].fillna(X_train[num_cols].mean(), inplace=True)
    X_test[num_cols].fillna(X_train[num_cols].mean(), inplace=True)

else:
    # Standardize numeric features in X_train and X_test
    for col in num_cols:
        X_train[[col]] = scaler.fit_transform(X_train[[col]])
        X_test[[col]] = scaler.transform(X_test[[col]])

        # Convert to float
        X_train[[col]] = X_train[[col]].astype('float64')
        X_test[[col]] = X_test[[col]].astype('float64')

    # Display a message to indicate completion
st.write("Standard scaling completed for numeric columns!")
cat_cols = ['gender', 'Contract', 'InternetService']  # Add other categorical columns
X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
y_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
y_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
# Convert categorical columns to numerical types
X_train_encoded = X_train_encoded.astype('float64')
X_test_encoded = X_test_encoded.astype('float64')
y_train = y_train.astype('float64')
y_test = y_test.astype('float64')


# Create and fit KNN model
knn_model = KNeighborsClassifier(n_neighbors=11)
knn_model.fit(X_train_encoded, y_train)
accuracy_knn = knn_model.score(X_test_encoded, y_test)

# Display results
st.write("KNN accuracy:", accuracy_knn)