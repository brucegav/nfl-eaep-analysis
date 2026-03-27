"""
modeling.py

Machine learning pipeline for NFL franchise decline prediction.
Handles temporal train/validation splitting, model training for Logistic
Regression and Random Forest Classifier, and standardized model evaluation.
"""

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

def train_test_split_temporal(feature_df, split_year, end_year, features, target):
    """
    Prepares test and training data for Logistic Regression and Random Forest modeling 
    over a specified timeframe

    Args:
        feature_df (pd.DataFrame): a dataframe of all features and the target, and their values, to be modeled
        split_year (integer): the NFL season the data is to be split on
        end_year (integer): the NFL season the data is to end on
        features (list): a list of strings containing the feature column names
        target (str): a string containing the name of the target variable

    Returns:
        X_train (pd.DataFrame): a dataframe holding the features training data
        y_train (pd.Series): a Series holding the target training data
        X_val (pd.DataFrame): a dataframe holding the features test data
        y_val (pd.Series): a Series holding the target test data
    """
    
    X_train = feature_df[feature_df['season'] <= split_year][features]
    y_train = feature_df[feature_df['season'] <= split_year][target]

    X_val = feature_df[(feature_df['season'] > split_year) & (feature_df['season'] <= end_year)][features]
    y_val = feature_df[(feature_df['season'] > split_year) & (feature_df['season'] <= end_year)][target]

    return X_train, y_train, X_val, y_val



def train_logistic_regression(X_train, y_train):
    """
    Trains a Logistic Regression model on the training dataset.  Measures the linear relationship
    between the feature set and the target variable.

    Args:
        X_train (pd.DataFrame): a dataframe holding the features training data
        y_train (pd.Series): a Series holding the target training data

    Returns:
        lr_model (LogisticRegression): a trained Logistic Regression model
    """
    
    lr_model = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr_model.fit(X_train, y_train)
    
    return lr_model



def train_random_forest(X_train, y_train):
    """
    Trains a Random Forest Classifier model on the training dataset.  Uses a decision tree techniques
    to determine the importance of each feature in the feature dataset

    Args:
        X_train (pd.DataFrame): a dataframe holding the features training data
        y_train (pd.Series): a Series holding the target training data

    Returns:
        rf_model (RandomForestClassifier): a trained Random Forest Classifier model
    """
    
    rf_model = RandomForestClassifier(
        random_state=42, 
        class_weight='balanced',
        n_estimators=500,
        max_depth=5,
        min_samples_leaf=5
    )

    rf_model.fit(X_train, y_train)

    return rf_model



def evaluate_model(model, X_val, y_val):
    """
    Evaluates a trained sklearn model and prints classification metrics.

    Args:
        model: a trained sklearn model with predict() and predict_proba() methods.
        X_Val (pd.DataFrame): a dataframe holding the features validation data.
        y_val (pd.Series): a Series holding the target validation data.

    Returns:
        None.  Prints the classification report, ROC-AUC score, and confusion matrix.
    """

    y_pred = model.predict(X_val)
    y_prob = model.predict_proba(X_val)[:, 1]

    print('\nClassification Report:')
    print(classification_report(y_val, y_pred))
    print(f'ROC-AUC Score: {roc_auc_score(y_val, y_prob):.3f}')
    print('\nConfusion Matrix')
    print(confusion_matrix(y_val, y_pred))
    