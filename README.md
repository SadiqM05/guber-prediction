# gUBER Prediction

**Machine Learning Project | Team gUBER**

## Project Overview

**gUBER Prediction** is a machine learning project focused on predicting ride cancellations using historical ride-booking data.

Our target audience is **Uber**, with **Uber drivers** as the primary end users.

### Problem Statement

Uber drivers may accept rides that are subsequently cancelled, resulting in wasted time, fuel, and missed opportunities to complete other trips.

Repeated cancellations can negatively affect drivers' potential earnings and Uber's overall operational efficiency.

Our goal is to develop a machine learning model that predicts the likelihood of customer-initiated ride cancellations, helping inform better ride-management decisions.

## Machine Learning Approach

We will use **supervised binary classification** to predict whether a booking will be cancelled by a customer.

Our target variable is derived from `Booking Status`:

| Booking Status | Target |
|---|---|
| Completed | 0 |
| Cancelled by Customer | 1 |

Other booking outcomes are excluded from the initial classification task.

### Models

We plan to implement and compare:

1. **Logistic Regression** — Establishes a baseline classification model.
2. **Random Forest** — Captures nonlinear relationships and interactions between features.
3. **Gradient Boosting / XGBoost** — Explores whether ensemble boosting techniques improve predictive performance.

### Evaluation Metrics

Models will be compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

We will prioritize **precision and recall** because identifying likely customer cancellations is more important than accuracy alone.

## Dataset

**Source:** [Uber Ride Analytics Dataset — Kaggle](https://www.kaggle.com/datasets/yashdevladdha/uber-ride-analytics-dashboard)

The dataset contains 150,000 booking records and 21 original columns.

Our preprocessing includes:

- Selecting relevant booking outcomes
- Checking missing values and duplicate records
- Identifying potentially invalid or inconsistent values
- Selecting features available at prediction time
- Engineering date and time features
- Encoding categorical variables
- Splitting data into training and testing sets

To reduce data leakage, the initial model excludes information that may only become available after a booking is completed or cancelled.

The raw dataset is not tracked in GitHub. Each team member must download it separately.

## Getting Started

### 1. Clone the Repository

Clone this repository and open the project folder in **Visual Studio Code**.

### 2. Create a Virtual Environment and activate it

### 3. Install Dependencies

```powershell
python -m pip install -r reqs.txt
```

### 5. Download the Dataset

1. Download the dataset from the Kaggle link above.
2. Extract the downloaded files if necessary.
3. Place the CSV inside the `data/` directory.
4. Ensure the filename is `uberData.csv`.

Expected path:

```text
data/uberData.csv
```

### 6. Configure Jupyter

In VS Code:

1. Install the Python and Jupyter extensions.
2. Open a notebook from `notebooks/`.
3. Click **Select Kernel**.
4. Select the Python interpreter from `.venv`.

### 7. Deactivate the Environment

When finished working, optionally deactivate the environment:

```powershell
deactivate
```

## Project Structure

```text
guber-prediction/
│
├── data/
│   └── uberData.csv
│
├── notebooks/
│   ├── eda.ipynb
│   ├── preproc.ipynb
│   ├── model_training.ipynb
│   └── eval.ipynb
│
├── src/
│   └── preproc.py
│
├── models/
├── reports/
│
├── .gitignore
├── reqs.txt
└── README.md
```

### File Descriptions

| File | Purpose |
|---|---|
| `notebooks/eda.ipynb` | Exploratory data analysis, distributions, missing values, and cancellation patterns |
| `notebooks/preproc.ipynb` | Tests data cleaning, feature engineering, and preprocessing |
| `src/preproc.py` | Reusable functions for preparing the dataset and constructing preprocessing pipelines |
| `notebooks/model_training.ipynb` | Trains and compares classification models |
| `notebooks/eval.ipynb` | Evaluates models using classification metrics, visualizations, and comparisons |
| `data/` | Stores raw and processed datasets locally |
| `models/` | Stores trained model artifacts |
| `reports/` | Stores results, visualizations, and analysis |
| `reqs.txt` | Lists Python dependencies |
| `.gitignore` | Excludes local environments, datasets, and generated files from Git tracking |

## GitHub Collaboration Workflow

To maintain an organized development process, each team member should work on a separate feature branch.

### 1. Create a Branch

Before making changes, create a new branch from the latest `main` and dub it with your name.


### 2. Make Changes

Work within your assigned feature branch rather than directly modifying `main`.

### 3. Commit

Stage your changes and explain what you did in your commit!

Examples:

```text
Add initial exploratory data analysis
Implement reusable preprocessing functions
Train logistic regression baseline model
Add model evaluation metrics
```

### 4. Push

Push your commits to GitHub using **Push origin** in GitHub Desktop.

### 5. Create a Pull Request

When your feature is ready:

1. Open a pull request targeting `main`.
2. Describe the changes made.
3. Request a review from another team member.

### 6. Merge

After reviewing and approving the changes, merge the pull request into `main`.

### 7. Delete the Branch (optional)

Feel free to reuse your own branch

**Why we use this workflow:**

- Prevents teammates from overwriting each other's work
- Maintains a clear project history
- Makes individual contributions easy to track
- Encourages code reviews
- Keeps the main branch stable

## Project Workflow

The machine learning development process follows these stages:

1. **Exploratory Data Analysis (EDA)** — Understand the dataset and identify important patterns.
2. **Data Preprocessing** — Prepare features, define the target, and construct preprocessing pipelines.
3. **Model Training** — Train logistic regression, random forest, and potentially gradient boosting models.
4. **Model Evaluation** — Compare models using classification metrics and visualizations.
5. **Model Improvement** — Experiment with hyperparameters and additional features.
6. **Final Analysis & Reporting** — Document methodology, results, limitations, and conclusions.

## Team

**Team gUBER**

This project is developed collaboratively as part of a machine learning course.