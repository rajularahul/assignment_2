import streamlit as st
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dry Bean Classifier", layout="wide")
st.title("Dry Bean Classification Model Evaluator")

# a. Dataset upload option (CSV)
uploaded_file = st.file_uploader("Upload test_data.csv", type=["csv"])

# b. Model selection dropdown
model_choice = st.selectbox("Select a Machine Learning Model", 
                            ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"])

model_paths = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Preview of uploaded test data:", df.head())
    
    # Separate features and target
    X_test = df.drop(columns=['Class'])
    y_test = df['Class']
    
    # Load the scaler and transform features
    scaler = joblib.load("model/scaler.joblib")
    X_test_scaled = scaler.transform(X_test)
    
    # Load the selected model and predict
    model = joblib.load(model_paths[model_choice])
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    # c. Display of evaluation metrics
    st.subheader(f"Evaluation Metrics for {model_choice}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", round(accuracy_score(y_test, y_pred), 4))
    col2.metric("Precision", round(precision_score(y_test, y_pred, average='weighted'), 4))
    col3.metric("Recall", round(recall_score(y_test, y_pred, average='weighted'), 4))
    
    col4, col5, col6 = st.columns(3)
    col4.metric("F1 Score", round(f1_score(y_test, y_pred, average='weighted'), 4))
    col5.metric("MCC", round(matthews_corrcoef(y_test, y_pred), 4))
    col6.metric("AUC Score", round(roc_auc_score(y_test, y_pred_proba, multi_class='ovr'), 4))
    
    # d. Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    st.pyplot(fig)