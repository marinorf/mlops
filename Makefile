#Data management 
download-data:
	@mkdir -p data
	# download & unzip the CSV from Kaggle into data/
	kaggle datasets download \
	  -d uciml/breast-cancer-wisconsin-data \
	  -f data.csv \
	  -p data/ \
	  --unzip
	# rename to raw.csv
	mv data/data.csv data/raw.csv 

preprocess-data:
	# @mkdir -p data/processed
	# run the preprocessing script
	python3 preprocess.py

# Model training
train-catboost:
	@mkdir -p models
	# run the training script
	python3 models/catboost_model.py

train-MLJar:
	@mkdir -p models
	# run the training script
	python3 models/MLJAR_model.py

# Should be some testing here from automated tests - but skipped in this project

# Model evaluation
evaluate-models:
	@mkdir -p evaluate
	# run the evaluation script
	python3 evaluate/evaluate_catboost.py
	python3 evaluate/evaluate_MLJAR.py

# Dependencies
install:
	pip install -r requirements.txt