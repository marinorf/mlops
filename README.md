# mlops
This repository contains an ML workflow for the Breast Cancer Wisconsin dataset, designed for ML operations course at EWHA university. The workflow is managed via a Makefile, which automates key steps such as data management, preprocessing, model training, and evaluation.

Workflow Overview
1. Data Management
Download Data:
Downloads the Breast Cancer Wisconsin dataset from Kaggle and saves it as raw.csv in the data directory.
Command: make download-data
Prerequisite: Ensure you have a valid kaggle.json file in your environment to authenticate with Kaggle.
Preprocess Data:
Preprocesses the raw data and splits it into training and testing datasets (X_train.csv, y_train.csv, X_test.csv, y_test.csv) in the data directory.
Command: make preprocess-data

2. Model Training
Train CatBoost Model:
Trains a CatBoost classifier using the preprocessed training data and saves the model in the models directory.
Command: make train-catboost

Train MLJar AutoML Model:
Trains an AutoML model using the MLJar library and saves the results in the models directory.
Command: make train-MLJar

3. Model Evaluation
Evaluate Models:
Evaluates the trained CatBoost and MLJar models using the test dataset and generates evaluation reports in the evaluate directory.
Command: make evaluate-models

4. Dependency Management
Install Dependencies:
Installs all required Python packages listed in the requirements.txt file.
Command: make install

File Structure
data: Contains raw and preprocessed datasets, as well as predictions when these are done.
models: Stores trained models and AutoML results.
evaluate: Contains evaluation scripts.
Makefile: Automates the workflow.
preprocess.py: Handles data preprocessing and splitting.
models/catboost_model.py: Script for training the CatBoost model.
models/MLJAR_model.py: Script for training the MLJar AutoML model.
evaluate: Scripts for evaluating the models.

Usage
Set Up Environment: 
    Install dependencies: make install

Download and Preprocess Data:
    Download the dataset: make install
    Preprocess the data: make preprocess-data

Train Models:
    Train the CatBoost model: make train-catboost
    Train the MLJar AutoML model: make train-MLJar

Evaluate Models:
    Evaluate the trained models: make evaluate-models

Notes
Testing: Automated tests for functions are currently not implemented in this project.
Kaggle Authentication: Ensure you have a valid kaggle.json file to access the Kaggle API for downloading the dataset.