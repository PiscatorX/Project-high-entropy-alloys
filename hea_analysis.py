#!/usr/bin/env python
# coding: utf-8

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
import numpy as np
import joblib
import logging
import json
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import argparse
import logging
import pprint
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
import logging
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score




# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# List predictor variables and order of variables
PREDICTOR_VARS = ['Mg',
 'Al',
 'Si',
 'Ca',
 'Sc',
 'Ti',
 'V',
 'Cr',
 'Mn',
 'Fe',
 'Co',
 'Ni',
 'Cu',
 'Zn',
 'Ga',
 'Y',
 'Zr',
 'Nb',
 'Mo',
 'Tc',
 'Ru',
 'Rh',
 'Pd',
 'Ag',
 'Cd',
 'In',
 'Sn',
 'La',
 'Ce',
 'Pr',
 'Gd',
 'Ir',
 'Pt',
 'Weighted atomic radius',
 '$\\Delta$R$\\mathregular{_{}}$ $\\mathregular{^{}}$',
 'VEC',
 'Weighted Pauling EN',
 '$\\Delta$Pauling EN$\\mathregular{_{}}$ ',
 'Weighted Mulliken EN',
 '$\\Delta$Mulliken EN$\\mathregular{_{}}$ ',
 'Mixing entropy',
 'BET Surface area',
 'Pore volume',
 'Poresize',
 'Calcination',
 'Reduction Temperature',
 'Reaction Temperature',
 'Pressure',
 '$\\ $H$\\mathregular{_{2}}$/ $\\mathregular{{CO_2}}$ ratio' ]

RESPONSE_VAR = ['$\\mathregular{{CO_2}}$ Yield']


def get_db(db_filename):
    
    
    # Access the filename
    logging.info(f"Filename provided: {db_filename}")
    # Importing the dataset
    dataset = pd.read_csv(db_filename, encoding='latin-1')
    
    return dataset


def get_missingness_cols(dataset):
    
    """
    Returns a list of column names with missing values in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input dataset.

    Returns:
    List[str]: List of column names with missing values.
    """
    
    return dataset.columns[dataset.isnull().any()].tolist()




def NaN_check(dataset, tag):
    
    """
    Analyzes a pandas DataFrame for missing (NaN) values and logs the findings.
    
    Parameters:
    -----------
    dataset : pandas.DataFrame
        The dataset to be analyzed for NaN values.
    tag : str
        A label or identifier used in the log to distinguish this NaN check (e.g., "train set", "test set").
    
    Functionality:
    --------------
    - Calculates the total number and percentage of NaN values in each column.
    - Logs a warning if any NaN values are found, including counts and percentages per column.
    - Logs an info message if no NaN values are found.
    """
    
    # Log the start of NaN analysis for the specified tag
    logging.info(f"NaN analysis {tag}")
    
    # Count NaNs in each column
    nan_counts = dataset.isnull().sum()
    
    # Calculate the percentage of NaNs in each column
    nan_percent = (nan_counts / len(dataset)) * 100
    
    # Create a DataFrame summarizing the NaN counts and percentages
    nan_report = pd.DataFrame({
        'NaN Count': nan_counts,
        'Percentage (%)': nan_percent
    })
    
    # Filter to only include columns with at least one NaN
    nan_report = nan_report[nan_report['NaN Count'] > 0]

    # Log warning with details if any NaNs are found
    if not nan_report.empty:
        logging.warning("NaN values found in the dataset:")
        for col, row in nan_report.iterrows():
            logging.warning(f"{col}: {row['NaN Count']} missing ({row['Percentage (%)']:.2f}%)")
    else:
        # Log that the dataset has no missing values
        logging.info("No NaN values found in the dataset.")



def db_impute(dataset):
    """
    Imputes missing numerical values in a dataset using the mean strategy.

    This function identifies columns with missing values (NaNs) by name and applies mean imputation
    to those columns. It checks for NaNs both before and after the imputation process for verification.

    Parameters:
    -----------
    dataset : pandas.DataFrame
        The input dataset containing potential missing (NaN) values.

    Returns:
    --------
    pandas.DataFrame
        A copy of the dataset with missing values imputed using the mean of each respective column.

    """

    # Check for NaN values in the raw dataset
    NaN_check(dataset, "Raw data")

    # Get the column names that contain missing values
    missingness_cols = get_missingness_cols(dataset)

    # Create an imputer object that replaces NaN with the column mean
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

    # Fit the imputer to the columns with missing values
    imputer.fit(dataset[missingness_cols])

    # Transform and replace the missing values with the computed means
    dataset[missingness_cols] = imputer.transform(dataset[missingness_cols])

    # Check for NaN values again after imputation
    NaN_check(dataset, "Imputed data")

    return dataset




def train_split(dataset, test_size = 0.3):

    
    # Splitting the dataset into the Training set and Test set
    X_full = dataset[PREDICTOR_VARS].values
    y_full = dataset[RESPONSE_VAR].values #Regenerability of MOFs
       
    # # Convert to DataFrame to retain original column names and values
    X_df = pd.DataFrame(X_full, columns = dataset[PREDICTOR_VARS].columns)  # Using original values

    # Split data into training and test sets (using original values)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_full, test_size=0.3, random_state=0)
    
    logging.info(f"Random Forest >> split dataset train samples: {X_train.shape[0]}")
    logging.info(f"Random Forest >> Split dataset test samples: {X_test.shape[0]}")

    return (X_full, y_full, X_train, X_test, y_train, y_test)



def run_RF_with_bayesian_tuning(X_train, X_test, y_train, y_test, X_full, y_full,
                                 model_output_path='final_model.joblib',
                                 random_state=42, n_splits=10, n_iter=32):
    """
    Trains a Random Forest Regressor with Bayesian hyperparameter tuning,
    evaluates performance before and after tuning, retrains on full dataset,
    and saves the final model.

    Returns:
        final_model: Trained RandomForestRegressor on full dataset
        best_params: Best hyperparameters from BayesSearchCV
        metrics: Evaluation results on test data
    """
    # -------------------------------
    # Baseline Model (Before Tuning)
    # -------------------------------
    baseline_model = RandomForestRegressor(random_state=random_state)
    baseline_model.fit(X_train, y_train.ravel())

    y_pred_base = baseline_model.predict(X_test)
    rmse_base = np.sqrt(mean_squared_error(y_test, y_pred_base))
    mae_base = mean_absolute_error(y_test, y_pred_base)
    r2_base = r2_score(y_test, y_pred_base)

    logging.info("Baseline Model Performance (Before Tuning):")
    logging.info(f"  RMSE: {rmse_base:.4f}")
    logging.info(f"  MAE:  {mae_base:.4f}")
    logging.info(f"  R²:   {r2_base:.4f}")

    # -------------------------------
    # Bayesian Hyperparameter Tuning
    # -------------------------------

    param_space = {
    'n_estimators': Integer(1, 100),          # Reduce upper bound: 1000 trees is overkill
    'max_depth': Integer(3, 15),                # Shallow trees reduce overfitting risk
    'min_samples_split': Integer(4, 10),        # Prevents over-splitting on small data
    'min_samples_leaf': Integer(2, 10),         # Larger leaf sizes force generalization
    'max_features': Categorical(['sqrt'])       # 'sqrt' generally works well for RF
    }
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    bayes_search = BayesSearchCV(estimator=RandomForestRegressor(random_state=random_state),
                                 search_spaces=param_space,
                                 n_iter=n_iter,
                                 cv=kf,
                                 scoring='r2',
                                 n_jobs=-1,
                                 random_state=random_state,
                                 verbose=1)

    bayes_search.fit(X_train, y_train.ravel())
    best_params = bayes_search.best_params_
    logging.info(f"Best Hyperparameters (Bayesian): {best_params}")

    
    # -------------------------------
    # Evaluation After Tuning
    # -------------------------------
    best_model = bayes_search.best_estimator_
    y_pred_tuned = best_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_tuned))
    mae = mean_absolute_error(y_test, y_pred_tuned)
    r2 = r2_score(y_test, y_pred_tuned)

    logging.info("Tuned Model Performance (After Bayesian Tuning):")
    logging.info(f"  RMSE: {rmse:.4f}")
    logging.info(f"  MAE:  {mae:.4f}")
    logging.info(f"  R²:   {r2:.4f}")

    # -------------------------------
    # Train Final Model on Full Data
    # -------------------------------
    final_model = RandomForestRegressor(**best_params, random_state=random_state)
    final_model.fit(X_full, y_full.ravel())
    joblib.dump(final_model, model_output_path)
    logging.info(f"Final model saved to {model_output_path}")

    if hasattr(X_full, 'columns'):
        feature_importances = pd.Series(final_model.feature_importances_, index=X_full.columns)
        feature_importances = feature_importances.sort_values(ascending=False)
    else:
        feature_importances = final_model.feature_importances_

    return final_model, best_params, {
        'rmse_before_tuning': rmse_base,
        'mae_before_tuning': mae_base,
        'r2_before_tuning': r2_base,
        'rmse_after_tuning': rmse,
        'mae_after_tuning': mae,
        'r2_after_tuning': r2,
        'feature_importances': final_model.feature_importances_
    }



def save_tuning_results(metrics: dict, best_params: dict, output_path='model_metrics.json'):
    """
    Saves model metrics and best hyperparameters to a JSON file.

    Args:
        metrics (dict): Dictionary containing performance metrics and feature importances.
        best_params (dict): Best hyperparameters from tuning.
        output_path (str): Path to save the JSON file.
    """
    output = {
        'best_params': best_params,
        'metrics': {
            'rmse_before_tuning': metrics.get('rmse_before_tuning'),
            'mae_before_tuning': metrics.get('mae_before_tuning'),
            'r2_before_tuning': metrics.get('r2_before_tuning'),
            'rmse_after_tuning': metrics.get('rmse_after_tuning'),
            'mae_after_tuning': metrics.get('mae_after_tuning'),
            'r2_after_tuning': metrics.get('r2_after_tuning'),
        },
        'feature_importances': metrics.get('feature_importances', []).tolist()  # Convert NumPy array to list if needed
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    logging.info(f"Results saved to {output_path}")






def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Read a filename from the command line")

    # Add the filename argument
    #parser.add_argument("-f","--filename", type=str, help="Path to the input file", default = "database_latest22032025.csv")
    parser.add_argument("-f","--filename", type=str, help="Path to the input file", default = "database_latest13042024expanded.csv")

    # Parse the arguments
    args = parser.parse_args()

    return args
    

    
if __name__ == "__main__":
    args = main()
    dataset = get_db(args.filename)
    imputed_dataset = db_impute(dataset)
    (X_full, y_full, X_train, X_test, y_train, y_test) = train_split(imputed_dataset)
    final_model, best_params, metrics = run_RF_with_bayesian_tuning(X_train, X_test, y_train, y_test, X_full, y_full)
    save_tuning_results(metrics, best_params)
    
        

