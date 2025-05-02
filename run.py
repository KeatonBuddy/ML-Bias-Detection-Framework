import os
import itertools

import pandas as pd
import numpy as np

from fancyimpute import IterativeSVD
from imblearn.over_sampling import SMOTENC, SMOTEN

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def run_pipeline(data, X_cols, y_col, subset):
    if len(X_cols) == 1:
        data = data.dropna()
        if X_cols[0] != "PIR":
            data[X_cols[0]] = data[X_cols[0]].astype(int)

    X = data[X_cols]
    y = data[y_col]

    if len(X_cols) > 1:
        n_rank = 1 if len(X_cols) == 2 else 2

        # Data imputation: IterativeSVD
        imputer = IterativeSVD(rank=n_rank) # Hyperparmeters are not described in the paper -> set default
        X = imputer.fit_transform(X)
        X = pd.DataFrame(X, columns=X_cols)

        for col in X_cols:
            if col != "PIR":
                X[col] = X[col].astype(int)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handling imbalanced data: SMOTENC or SMOTEN
    categorical_features = ["US_citizenship", "education_level", "num_familiy", "marital_status", "gender"]
    categorical_features = list(set(categorical_features) & set(X_cols))
    categorical_features = "auto" if len(categorical_features) < 1 else categorical_features

    if "PIR" in X_cols:
        resampler = SMOTENC(categorical_features=categorical_features, random_state=42) # Hyperparameters are not described in the paper -> set default & seed only
    else:
        resampler = SMOTEN(random_state=42)
    
    X_train_bal, y_train_bal = resampler.fit_resample(X_train, y_train)

    # Normalizing data
    scaler = StandardScaler()
    X_train_bal = scaler.fit_transform(X_train_bal)
    X_val = scaler.transform(X_val)

    # Models: All set to default hyperparameters based on the paper
    models = {
        'KNN': KNeighborsClassifier(),
        'SVM': SVC(probability=True),
        'MLP': MLPClassifier(),
        'LR': LogisticRegression(),
        'DT': DecisionTreeClassifier(),
        'RF': RandomForestClassifier(),
        'XGB': XGBClassifier()
    }

    results = {
        "used_col": [],
        "subset": [], # 0 for non-subset, 1 for bottom 25% removed, 2 for top 25% removed
        "model": [],
        "auc_score": [],
        "accuracy": [],
        "sensitivity": [],
        "specificity": []
    }

    X_cols_str = X_cols[0] if len(X_cols) == 1 else ",".join(X_cols) 

    # Training & evaluating
    for name, model in models.items():
        model.fit(X_train_bal, y_train_bal)
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]
        auc_score = roc_auc_score(y_val, y_proba)
        accuracy = accuracy_score(y_val, y_pred)
        sensitivity = recall_score(y_val, y_pred)
        specificity = recall_score(y_val, y_pred, pos_label=0)

        results["used_col"].append(X_cols_str)
        results["subset"].append(subset)
        results["model"].append(name)
        results["auc_score"].append(auc_score)
        results["accuracy"].append(accuracy)
        results["sensitivity"].append(sensitivity)
        results["specificity"].append(specificity)

        print(f"{name}: auc_score={auc_score:.2f}, accuracy={accuracy:.2f}")

    # Feature importance using XGB
    if len(X_cols) > 1:
        importance = models['XGB'].feature_importances_
        feature_importance = pd.Series(importance, index=X_cols).sort_values(ascending=False)
        feature_importance.to_csv(f"results/XGB_feature_importances/used_col{X_cols_str.replace(',','-')}--subset{subset}.csv")

    return pd.DataFrame(results)


# 1: All possible combination
def run_all_combination(data, X_cols_orig, y_col):
    results_df = pd.DataFrame(columns=["used_col", "subset", "model", "auc_score", "accuracy", "sensitivity", "specificity"])

    for length in range(1, len(X_cols_orig) + 1):
        for X_cols in itertools.combinations(X_cols_orig, length):
            X_cols = list(X_cols)

            results_temp = run_pipeline(data, X_cols=X_cols, y_col=y_col, subset=0)
            results_df = pd.concat([results_df, results_temp])
    
    results_df.to_csv("results/combinations.csv")


# 2: Remove subsets (top or bottom 25%) from a column one by one
def run_subsets(data, X_cols_orig, y_col):
    X_cols = X_cols_orig.copy()
    for X_col in X_cols:
        # Remove bottom 25%
        lower_quantile = data[X_col].quantile(0.25)
        filtered_data = data[data[X_col] >= lower_quantile]

        results_temp = run_pipeline(filtered_data, X_cols=X_cols, y_col=y_col, subset=1)
        results_df = pd.concat([results_df, results_temp])

        # Remove top 25%
        upper_quantile = data[X_col].quantile(0.75)
        filtered_data = data[data[X_col] <= upper_quantile]

        results_temp = run_pipeline(filtered_data, X_cols=X_cols, y_col=y_col, subset=2)
        results_df = pd.concat([results_df, results_temp])

    results_df.to_csv("results/subsets.csv")


def main():
    os.makedirs("results/", exist_ok=True)
    os.makedirs("results/XGB_feature_importances/", exist_ok=True)

    data = pd.read_csv("data/NHANES-C_processed.csv")
    data = data.drop("SEQN_new", axis=1)

    X_cols_orig = ["US_citizenship", "education_level", "num_familiy", "marital_status", "PIR", "gender", "age"]
    y_col = "stroke"

    run_all_combination(data, X_cols_orig, y_col)
    run_subsets(data, X_cols_orig, y_col)


if __name__ == "__main__":
    main()
