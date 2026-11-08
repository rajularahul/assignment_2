# ML Assignment 2: Dry Bean Classification Evaluator

## a. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models to accurately identify the type of a dry bean based on its form, shape, and structure. The project also involves deploying these models through an interactive web application to allow users to evaluate model performance on test data dynamically.

## b. Dataset Description
**Dataset:** Dry Bean Dataset (UCI Machine Learning Repository / Kaggle)
- **Characteristics:** Multivariate, Multi-class Classification
- **Number of Instances:** 13,611 (Reduced/Split for testing)
- **Number of Features:** 16 continuous numeric features (e.g., Area, Perimeter, MajorAxisLength, MinorAxisLength, AspectRatio, Eccentricity, ConvexArea, etc.)
- **Target Variable:** `Class` (7 distinct bean types: Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira)
- **Preprocessing:** Features were scaled using `StandardScaler` to optimize distance-based and gradient-based models. 

## c. GitHub Repository Link
**Link:** (https://github.com/rajularahul/assignment_2)

**Live Streamlit App Link:** (https://assignment2-bdea7jauv59kz6fewzmkz6.streamlit.app/)

## d. Models Used and Comparison Table
The following table summarizes the evaluation metrics for the 5 machine learning classification models implemented. Metrics were calculated on the held-out test/validation set.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Logistic Regression** | 0.9266 | 0.9950 | 0.9278 | 0.9266 | 0.9269 | 0.9117 |
| **Decision Tree** | 0.8928 | 0.9442 | 0.8932 | 0.8928 | 0.8929 | 0.8709 |
| **KNN** | 0.9192 | 0.9878 | 0.9203 | 0.9192 | 0.9195 | 0.9027 |
| **Naive Bayes** | 0.9038 | 0.9919 | 0.9050 | 0.9038 | 0.9039 | 0.8845 |
| **Support Vector Machine** | 0.9339 | 0.9959 | 0.9344 | 0.9339 | 0.9341 | 0.9204 |

## Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well as a baseline, yielding a high AUC (0.9950). The multinomial approach handled the 7 distinct classes very effectively. |
| **Decision Tree** | The weakest performer among the group (~89.3% accuracy, lowest AUC). It likely struggled with the continuous, overlapping geometric features and might be slightly overfitting the training data. |
| **KNN** | Showed strong performance (~91.9% accuracy). Because the features describe physical dimensions, KNN successfully grouped similar beans together using distance metrics after standard scaling. |
| **Naive Bayes** | Delivered a solid and very fast 90.3% accuracy. Despite its assumption of feature independence (which may not strictly hold for geometric traits like area and perimeter), it still generated an impressive 0.9919 AUC. |
| **Support Vector Machine** | The most accurate and robust model across all metrics. The RBF kernel successfully navigated the complex, non-linear boundaries between closely related bean shapes. |
| **Overall, Winner for your dataset?** | **Support Vector Machine (SVM)**. It achieved the highest scores across the board: Accuracy (93.39%), F1 Score (93.41%), and AUC (99.59%). |