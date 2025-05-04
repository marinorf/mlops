from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd

# Load the preprocessed data
X_train = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/X_train.csv')
y_train = pd.read_csv('/Users/marinorfolk/Desktop/MLops_rep/mlops/data/processed/y_train.csv')

# Define parameter grid for hyperparameter tuning
param_grid = {
    'depth': [4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
    'iterations': [200, 500],
    'class_weights': [{0:1, 1:w} for w in [1,2,5]]
}

# Initialize CatBoostClassifier
catboost_model = CatBoostClassifier(loss_function='Logloss', 
                                    verbose=0,
                                    auto_class_weights=None)

# Perform GridSearchCV for hyperparameter tuning
grid_search = GridSearchCV(estimator=catboost_model, param_grid=param_grid, cv=5, scoring='recall', n_jobs=1)
grid_search.fit(X_train, y_train)

# Get the best model and parameters
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)

# Save the best model to file
best_model.save_model('/Users/marinorfolk/Desktop/MLops_rep/mlops/models/best_catboost_model.cbm')