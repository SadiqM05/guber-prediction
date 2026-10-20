# guber-prediction
ML project for team gUBER

when you clone repo run the following:

py -m venv .venv [create virtual environment]
.\.venv\Scripts\Activate.ps1 [activate it]
python -m pip install -r requirements.txt [install dependencies]

before closing vs code:
run "deactivate" in terminal to close venv

files:
eda.ipynb: exploratory data analysis 1
this is where we understand the dataset before modifying it.
purpose: examine cancellation rates, missing values, relationships, etc.

preproc.ipynb: preprocessing
purpose: issing values, duplicates, encoding, train/test splits, etc.

model_training.ipynb: model training
purpose: train and compare models

eval.ipynb: model evaluation
purpose: accuracy, precision, F1, charts, etc.