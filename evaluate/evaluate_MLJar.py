import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.metrics import classification_report
import os
import cv2

from supervised.automl import AutoML

# Load the preprocessed data
X_test = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/X_test.csv')
y_test = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/y_test.csv')['diagnosis']

# Load the best model
RESULTS_PATH = "models/AutoML_results"
a = AutoML(results_path=RESULTS_PATH, mode="Predict", total_time_limit=300,
           n_jobs=1, ml_task="binary_classification",
           eval_metric='f1')

# Model evaluation
y_pred = a.predict(X_test)

print(f"Model score {accuracy_score(y_test, y_pred)} on set")

print("Confusion matrix: ", confusion_matrix(y_test, y_pred))

print("Classification report: ", classification_report(y_test, y_pred))

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/MLJar_predictions.csv', index=False)

# Load the leaderboard
RESULTS_PATH = "models/AutoML_results"
lb_csv = os.path.join(RESULTS_PATH, "leaderboard.csv")

NUMBER_OF_MODELS = 3
df = pd.read_csv(lb_csv)

df_sorted = df.sort_values("metric_value", ascending=False)

# Top models
top = df_sorted.head(NUMBER_OF_MODELS)

print(f"Top {NUMBER_OF_MODELS} models by validation score:")
for i, row in top.iterrows():
    print(f" {i+1}. {row['name']:20s} ↳ score = {row['metric_value']:.4f}")
    