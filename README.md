# Detecting Hidden Biases in Clinical AI: A Framework for Fair Stroke Outcome Prediction Using Clinical Data

This repository contains the code, data processing pipelines, and analysis notebooks for our project "Detecting Hidden Biases in Clinical AI: A Framework for Fair Stroke Outcome Prediction Using Clinical Data." We develop and evaluate machine learning models to predict stroke outcomes from demographic and socioeconomic factors, focusing on bias detection and mitigation across model lifecycles.

## Key Features

@@ -19,10 +19,117 @@ Engages clinicians, data scientists, policymakers, social scientists, medical hi


### Proof‑of‑Concept on NHANES Data
Leverages the NHANES Continuous dataset (1999–2018) to test bias detection methods, focusing on seven demographic features—US citizenship, education level, family size, marital status, poverty income ratio (PIR), gender, and age—and their impact on stroke prediction performance.

### Bias‑Improvement Workflow
Flags high‑risk feature subgroups and suggests alternative model outputs to mitigate unfair performance disparities, with results presented in stakeholder‑driven reports for transparent decision‑making 


By combining rigorous technical analyses with ethical and societal insights, this framework sets adaptable standards for responsible AI in medicine, aiming to reduce health disparities and build trust in AI‑enabled stroke care.

## Repository Structure
```
├── data/                           # Raw and preprocessed data files
│   └── NHANES-C_processed.csv     # Processed NHANES continuous dataset
├── results/                        # Output results and metrics
│   ├── combinations.csv            # Performance metrics for all feature combinations
│   ├── subsets.csv                 # Metrics after removing top/bottom quartiles
│   └── XGB_feature_importances/    # Feature importance CSVs for XGBoost
├── notebooks/                      # Jupyter notebooks for exploratory analysis
│   ├── data_preprocess.ipynb       # Data cleaning, imputation, and encoding
│   └── results_analysis.ipynb      # Visualization and statistical analysis of results
├── run.py                          # Main script to execute training pipeline
├── requirements.txt                # Python dependencies
├── FinalProposal.pdf               # Original project proposal document
├── Presentation.pdf                # Project slide deck
└── README.md                       # Project README (this file)
```
## Installation
### 1. Clone this repository:
```
git clone https://github.com/yourusername/DS9600-final.git
cd DS9600-final
```

### 2. Create a virtual environment and install dependencies:
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```
## Usage
- Preprocess Data: Open and run notebooks/data_preprocess.ipynb to inspect and modify preprocessing steps, including handling missing values and encoding categorical features.

- Run Pipeline: Execute the full pipeline via:
```
python run.py
```
- This will:

  - Load data/NHANES-C_processed.csv.

  - Generate all possible feature combinations and evaluate models.

  - Remove top/bottom 25% subsets to assess bias impact.

  - Save results to results/combinations.csv and results/subsets.csv.

  - Export XGBoost feature importances into results/XGB_feature_importances/.

- Analyze Results: Launch notebooks/results_analysis.ipynb to explore performance metrics, generate plots (AUC heatmaps, accuracy bar charts), and derive insights on bias patterns.

## Methods

- Feature Engineering: From NHANES-C dataset, we selected seven demographic variables:

  - US citizenship, education level, household size, marital status, poverty income ratio (PIR), gender, age.

  - We excluded individuals under 55 and created a binary target stroke.

  - All possible subsets of features were generated to examine intersectional effects.

- Data Imputation: We applied iterative Singular Value Decomposition (IterativeSVD) to handle missing values when multiple features are present. Single-feature experiments drop missing entries.

- Resampling: To address class imbalance (stroke vs. non-stroke), we used:

  - SMOTENC for mixed categorical/continuous data when PIR is included.

  - SMOTEN for fully continuous subsets.

- Normalization: Standard scaling was applied to resampled training data and validation splits.

- Model Training: We evaluated seven ML algorithms with default hyperparameters:

  - K-Nearest Neighbors (KNN)

  - Support Vector Machine (SVM)

  - Multi-Layer Perceptron (MLP)

  - Logistic Regression (LR)

  - Decision Tree (DT)

  - Random Forest (RF)

  - XGBoost (XGB)

- Bias Assessment:

  - All combinations: Trained on every feature subset and recorded AUC, accuracy, sensitivity, specificity.

  - Quartile subsets: Removed top/bottom 25% of each feature to simulate distribution shifts.

  - Feature importance: Extracted from XGBoost to identify feature-level contributions across subsets.

## Results

- combinations.csv: Performance metrics for each feature combination and model.
 
- subsets.csv: Metrics after excluding extreme quartile data to reveal bias robustness.

- Feature importance: Saved in results/XGB_feature_importances/ for deeper inspection.

## References

Please refer to FinalProposal.pdf and the project proposal for full bibliographic details of the methods and related work.
