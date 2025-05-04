from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load the preprocessed data
X_test = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/X_test.csv')
y_test = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/y_test.csv')['diagnosis']

# Load the best model
model = CatBoostClassifier()
model.load_model('/Users/marinorfolk/Desktop/MLops_rep/mlops/models/best_catboost_model.cbm')

# Evaluate the best model on the test set
y_pred = model.predict(X_test)
print(f"Accuracy on test set: {accuracy_score(y_test, y_pred)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Save the predictions to a CSV file
predictions_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
predictions_df.to_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/catboost_predictions.csv', index=False)
print("Predictions saved to catboost_predictions.csv")
