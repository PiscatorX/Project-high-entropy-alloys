#!/usr/bin/env python
# coding: utf-8

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
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFECV
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def get_db(db_filename):
    
    
    # Access the filename
    logging.info(f"Filename provided: {db_filename}")
    # Importing the dataset
    dataset = pd.read_csv(db_filename, encoding='latin-1')
    
    return dataset



def NaN_check(dataset, tag):

    # Check for NaNs and calculate percentage
    logging.info(f"NaN analysis {tag}")
    nan_counts = dataset.isnull().sum()
    nan_percent = (nan_counts / len(dataset)) * 100
    nan_report = pd.DataFrame({
        'NaN Count': nan_counts,
        'Percentage (%)': nan_percent })
    
    nan_report = nan_report[nan_report['NaN Count'] > 0]

    if not nan_report.empty:
        logging.warning("NaN values found in the dataset:")
        for col, row in nan_report.iterrows():
            logging.warning(f"{col}: {row['NaN Count']} missing ({row['Percentage (%)']:.2f}%)")
    else:
        logging.info("No NaN values found in the dataset.")




def db_impute(dataset):

    NaN_check(dataset, "Raw data")
    imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
    imputer.fit(dataset.iloc[1:, 28:36])
    dataset.iloc[1:, 28:36] = imputer.transform(dataset.iloc[1: , 28:36])
    NaN_check(dataset, "Imputed data")
    
    return dataset


def train_split(dataset, test_size = 0.3):

    
    # Splitting the dataset into the Training set and Test set
    X = dataset.iloc[:, 2:37].values
    y = dataset.iloc[:, -1].values #Regenerability of MOFs
    # # Convert to DataFrame to retain original column names and values
    X_df = pd.DataFrame(X, columns=dataset.columns[2:37])  # Using original values

    # Split data into training and test sets (using original values)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.3, random_state=0)
    
    logging.info(f"Random Forest >> split dataset train samples: {X_train.shape[0]}")
    logging.info(f"Random Forest >> Split dataset test samples: {X_test.shape[0]}")

    return (X_train, X_test, y_train, y_test)



def run_RF(X_train, X_test, y_train, y_test, random_state = 42):

    
    # Train Random Forest Regressor on original data
    regressor = RandomForestRegressor(n_estimators=100, random_state=0)
    regressor.fit(X_train, y_train)
    
    ## Predicting the Test set results
    y_pred = regressor.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    logging.info(f"Random Forest >> Test RMSE: {rmse:.4f}")
    logging.info(f"Random Forest >> Test R² Score: {r2:.4f}")
    


def run_rfecv_feature_sel(
    X, 
    y, 
    estimator=None, 
    cv_splits=5, 
    scoring='r2', 
    min_features=1, 
    step=1, 
    n_jobs=-1, 
    random_state=42,
    verbose=False):
    """
    Perform recursive feature elimination with cross-validation (RFECV)
    using a Random Forest (or custom) estimator.

    Parameters
    ----------
    X : pandas.DataFrame or numpy.ndarray
        Feature data.
    y : array-like
        Target vector.
    estimator : object, optional
        The base estimator to use for feature ranking. If None, a RandomForestClassifier
        with the provided random_state and n_jobs will be used.
    cv_splits : int, default=5
        Number of splits for stratified cross-validation.
    scoring : str, default='accuracy'
        Scoring metric used to evaluate model performance.
    min_features : int, default=1
        Minimum number of features to select.
    step : int, default=1
        Number of features to remove at each iteration.
    n_jobs : int, default=-1
        Number of jobs to run in parallel.
    random_state : int, default=42
        Random seed for reproducibility.
    verbose : bool, default=False
        If True, additional log messages will be recorded.

    Returns
    -------
    results : dict
        Dictionary with keys:
            'optimal_num_features': optimal number of features selected,
            'selected_features': list of selected feature names (if X is a DataFrame),
            'feature_ranking': DataFrame with features and their ranking,
            'cv_scores': DataFrame with number of features vs. cross-validation score,
            'rfecv_model': the trained RFECV model,
       
     'classification_report': classification report string on X and y.
    """

    
    estimator = RandomForestRegressor(random_state=random_state, n_jobs=n_jobs)
    logging.info("No estimator provided, using RandomForestClassifier with random_state=%s", random_state)

    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    logging.info("Using KFold with %d splits", cv_splits)
    
    # Set up RFECV

    rfecv = RFECV(
        estimator=estimator,
        step=step,
        cv=cv,
        scoring=scoring,
        min_features_to_select=min_features,
        n_jobs=n_jobs
    )
    
    logging.info("Starting RFECV fitting process...")
    
    rfecv.fit(X, y)
    
    optimal_features = rfecv.n_features_
    logging.info("Optimal number of features: %d", optimal_features)
    # Determine selected feature names (if X is a DataFrame) or indices.
    if isinstance(X, pd.DataFrame):
        selected_features = X.columns[rfecv.support_].tolist()
    else:
        selected_features = np.where(rfecv.support_)[0].tolist()
    logging.info("Selected features: %s", selected_features)
    
    # Prepare feature names for ranking.
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns
    else:
        feature_names = np.array([f"Feature_{i}" for i in range(X.shape[1])])
    
    ranking_df = pd.DataFrame({
        'Feature': feature_names,
        'Ranking': rfecv.ranking_
    }).sort_values(by='Ranking')
    logging.info("Feature Ranking:\n%s", ranking_df.to_string(index=False))
    
    # Extract cross-validation scores for each number of features.
    if hasattr(rfecv, 'grid_scores_'):
        cv_scores = rfecv.grid_scores_
    else:
        cv_scores = rfecv.cv_results_['mean_test_score']
    num_features = np.arange(min_features, len(cv_scores) + min_features)
    score_df = pd.DataFrame({
        'Num_features': num_features,
        'CV_Score': cv_scores
    })
    logging.info("CV Score for each number of features selected:\n%s", score_df.to_string(index=False))
    
    # Evaluate the final model on the entire dataset.
    y_pred = rfecv.predict(X)
    final_r2 = r2_score(y, y_pred)
    final_mse = mean_squared_error(y, y_pred)
    logging.info("Final model R² on entire dataset: %.4f", final_r2)
    logging.info("Final model Mean Squared Error on entire dataset: %.4f", final_mse)
    
    results = {
        'optimal_num_features': optimal_features,
        'selected_features': selected_features,
        'feature_ranking': ranking_df,
        'cv_scores': score_df,
        'rfecv_model': rfecv,
        'final_r2_score': final_r2,
        'final_mse': final_mse
    }
    
    return results
        
    
def main():
    
    # Create the parser
    parser = argparse.ArgumentParser(description="Read a filename from the command line")

    # Add the filename argument
    parser.add_argument("-f","--filename", type=str, help="Path to the input file", default = "database_latest22032025.csv")

    # Parse the arguments
    args = parser.parse_args()

    return args
    

    
if __name__ == "__main__":
    args = main()
    dataset = get_db(args.filename)
    imputed_dataset = db_impute(dataset)
    (X_train, X_test, y_train, y_test) = train_split(imputed_dataset)
    run_rfecv_feature_sel(X_train, y_train)
    
        




# from sklearn.inspection import PartialDependenceDisplay
# import matplotlib.pyplot as plt
# # Plot PDP for a selected feature (e.g., feature index 5)
# feature_idx = 20 
# fig, ax = plt.subplots(figsize=(8, 6))
# display = PartialDependenceDisplay.from_estimator(regressor, X_train, features=[feature_idx], ax=ax)
# plt.show()




# from sklearn.inspection import PartialDependenceDisplay
# import matplotlib.pyplot as plt
# # Plot PDP for a selected feature (e.g., feature index 5)
# feature_idx = 21 
# fig, ax = plt.subplots(figsize=(8, 6))
# display = PartialDependenceDisplay.from_estimator(regressor, X_train, features=[feature_idx], ax=ax)
# plt.show()




# from sklearn.inspection import PartialDependenceDisplay
# import matplotlib.pyplot as plt
# # Plot PDP for a selected feature (e.g., feature index 5)
# feature_idx = 22 
# fig, ax = plt.subplots(figsize=(8, 6))
# display = PartialDependenceDisplay.from_estimator(regressor, X_train, features=[feature_idx], ax=ax)
# plt.show()




# import shap
# # Convert X_train and X_test to DataFrame for SHAP analysis (Ensure feature names are included)
# feature_names = dataset.columns[2:37]  # Extract feature names
# X_test_df = pd.DataFrame(X_test, columns=feature_names)

# # Create SHAP Explainer
# explainer = shap.TreeExplainer(regressor)
# shap_values = explainer(X_test_df)

# # Select an individual instance (e.g., first test sample)
# instance_idx = 0
# shap.waterfall_plot(shap_values[instance_idx])

# # Show plot
# plt.show()




# # SHAP Summary Plot (Global feature importance)
# shap.summary_plot(shap_values, X_test_df)




# plot_data.to_csv("RF_model_performance.tsv", sep = "\t", index = False)

# ax = sns.regplot(y="y_pred",
#                  x="y_test", 
#                  color="g", 
#                  marker="+",
#                  line_kws={'label':'$r^2$ = {:.2f}'.format(r2)},
#                  data = plot_data)

# plt.ylabel('Predicted adsorptive ')
# plt.xlabel('Simulation adsorptive ')
# ax.legend(loc=9)
# plt.savefig('traning_r2.pdf', format='pdf', dpi=1200)
# plt.show()
# os.getcwd
# print('\n')




# #feature importance
# regressor.feature_importances_.shape
# print('1.2. Feature importance based on RF')
# print('\n')
# #print('regressor.feature_importances_.shape')

# xnum_cols = list(dataset.columns[2:37])
# #xnum_cols.remove('Regenerability of MOFs')


# feature_importance = pd.DataFrame(data = {"features": xnum_cols, "importance":regressor.feature_importances_} )
# print(feature_importance)
# feature_importance.to_csv("/Users/liberty.mguni/Downloads/HEA21_03_2025featureimportanceSelectivity.csv")
# feature_importance["importance"] = round(feature_importance["importance"],5)




# #pip install shap




# #Applying k-Fold Cross Validation
# from sklearn.model_selection import cross_val_score
# accuracies = cross_val_score(estimator = regressor, X = X_train, y = y_train, cv = 10)
# #print(X_train)
# print('\n')
# print('1.3. k-Fold Cross Validation Random Forest')
# print('\n')
# print("	metrics.r2_score: {:.2f} %".format(accuracies.mean()*100))
# #print("metrics.explained_variance_score: {:.2f} %".format(accuracies.std()*100))




# #Applying Grid Search to find the best model and the best parameters
# from sklearn.model_selection import GridSearchCV

# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 2, stop = 200, num = 2)]
# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']
# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(0, 50, num = 1)]
# max_depth.append(None)
# # Minimum number of samples required to split a node
# min_samples_split  = [int(x) for x in np.linspace(2, 50, num = 2)]
# #min_samples_split = [2, 5, 10]
# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]
# # Method of selecting samples for training each tree
# bootstrap = [True, False]
# # Create the random grid
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'max_depth': max_depth,
#                'min_samples_split': min_samples_split,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# #print(random_grid)
# grid_search = RandomizedSearchCV(estimator = regressor, param_distributions = random_grid, n_iter = 100, cv = 10, verbose=2, random_state=42, n_jobs = -1)
# grid_search.fit(X_train, y_train)
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_
# print('1.4. Best model parameters for RF')
# print('\n')
# print("Best Accuracy: {:.2f} %".format(best_accuracy*100))
# print("Best Parameters:", best_parameters)
# print('\n')




# from sklearn.metrics import r2_score
# regressor = RandomForestRegressor(**best_parameters, random_state=0)
# regressor.fit(X_train, y_train)

# # Predicting the Test set results
# y_pred = regressor.predict(X_test)

# # Calculate R2 score for test data
# test_r2 = r2_score(y_test, y_pred)

# # Calculate R2 score for train data
# y_train_pred = regressor.predict(X_train)
# train_r2 = r2_score(y_train, y_train_pred)

# print('1  Random forest')
# print('\n')
# print('1.1. R2 based on 80% training data for random forest')
# print(train_r2)
# print('\n')
# print('1.2. R2 based on test data for random forest')
# print(test_r2)
# print('\n')

# # Calculate percentage error
# percentage_error = np.abs((y_pred - y_test) / y_test) * 100

# print(y_train)
# print(y_test)




# # Saving model performance data
# plot_data = pd.DataFrame.from_dict({'y_pred': y_pred, 'y_test': y_test, 'errors': y_pred - y_test,
#                                     'abs_errors': abs(y_pred - y_test), 'percentage_error': percentage_error})
# plot_data.to_csv("RF_model_performance.tsv", sep="\t", index=False)

# # Plotting results _Test
# ax = sns.scatterplot(x="y_test", y="y_pred", hue="percentage_error", data=plot_data, palette="coolwarm", legend=False)
# plt
# plt.xlabel('Simulation adsorptive')
# plt.ylabel('Predicted adsorptive')

# percentage_error2 = np.abs((y_train_pred - y_train) / y_train) * 100

# plot_data2 = pd.DataFrame.from_dict({'y_train_pred': y_train_pred, 'y_train': y_train, 'errors': y_train_pred - y_train,
#                                     'abs_errors': abs(y_train_pred - y_train), 'percentage_error': percentage_error2})

# # Plotting results_ Training data: train_r2 = r2_score(y_train, y_train_pred)
# ax = sns.scatterplot(x=y_train, y=y_train_pred,marker='^' , hue= percentage_error2, data=plot_data2, palette="coolwarm")
# plt.xlabel('Simulation adsorptive')
# plt.ylabel('Predicted adsorptive')

# # Adding regression line
# sns.regplot(x=y_train, y=y_train_pred, scatter=False, color='gray', ax=ax)

# # Adding R2 values to the legend
# legend_labels = ['Test R^2: {:.2f}'.format(test_r2), 'Train R^2: {:.2f}'.format(train_r2)]
# plt.legend(legend_labels, loc='lower right')


# # Adding title and continuous scale to color bar
# #color_bar = ax.get_legend()
# #color_bar.set_title('Coeffiecents of Regression')

# # Normalize the percentage_error values
# norm = plt.Normalize(percentage_error.min(), percentage_error.max())
# norm = plt.Normalize(0, 100)
# sm = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
# sm.set_array([])

# # Create a new Axes for the color bar
# cax = plt.cm.ScalarMappable(cmap="coolwarm", norm=norm)
# cbar = plt.colorbar(cax, ax=ax, label='Percentage Error (%)')
# cbar.set_label('Percentage Error (%)', labelpad=-25, y=1.1, rotation=0)
# #plt.xlim([0, 1]) 
# #plt.ylim([0, 1]) 

# plt.savefig('model_performance.png', dpi=300)
# plt.show()

# print('\n')




# # Number of trees in random forest
# n_estimators = [int(x) for x in np.linspace(start = 200, stop = 1000, num = 2)]

# # Number of features to consider at every split
# max_features = ['auto', 'sqrt']

# # Maximum number of levels in tree
# max_depth = [int(x) for x in np.linspace(10, 110, num = 5)]
# max_depth.append(None)

# # Minimum number of samples required to split a node
# min_samples_split = [2, 5, 10]

# # Minimum number of samples required at each leaf node
# min_samples_leaf = [1, 2, 4]

# # Method of selecting samples for training each tree
# bootstrap = [True, False]

# # Create the random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'bootstrap': bootstrap
# }

# # Perform RandomizedSearchCV
# grid_search = RandomizedSearchCV(estimator = regressor, 
#                                  param_distributions = random_grid, 
#                                  n_iter = 100, 
#                                  cv = 5, 
#                                  verbose = 2, 
#                                  random_state = 42, 
#                                  n_jobs = -1)

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Get the best results
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_

# # Print the results
# print('Best model parameters for Random Forest:')
# print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
# print("Best Parameters:", best_parameters)




# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np

# # Define the parameter grid
# n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]  # More values
# max_features = ['sqrt', 'log2', None]  # Removed 'auto'
# max_depth = [int(x) for x in np.linspace(10, 110, num=5)]  # Removed None
# min_samples_split = [2, 5, 10]
# min_samples_leaf = [1, 2, 4]
# learning_rate = [0.01, 0.1, 0.2]  # Include learning rate

# # Create the random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'learning_rate': learning_rate
# }

# # Create the GradientBoostingRegressor model
# gb_reg = GradientBoostingRegressor(random_state=0)

# # Ensure X_train and y_train are defined
# try:
#     grid_search = RandomizedSearchCV(estimator=gb_reg,
#                                      param_distributions=random_grid,
#                                      n_iter=100,
#                                      cv=5,
#                                      verbose=2,
#                                      random_state=42,
#                                      n_jobs=-1)

#     # Fit the model
#     grid_search.fit(X_train, y_train)

#     # Get the best results
#     best_accuracy = grid_search.best_score_
#     best_parameters = grid_search.best_params_

#     # Print the results
#     print('Best model parameters for Gradient Boosting Regressor:')
#     print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
#     print("Best Parameters:", best_parameters)

# except NameError:
#     print("Error: Ensure X_train and y_train are properly defined before running the script.")




# from sklearn.ensemble import GradientBoostingRegressor
# from sklearn.model_selection import RandomizedSearchCV
# import numpy as np

# # Define the parameter grid
# n_estimators = [int(x) for x in np.linspace(start=1, stop=500, num=10)]  # Increase number of estimators
# max_features = ['auto', 'sqrt']
# max_depth = [int(x) for x in np.linspace(1, 500, num=11)]  # Increase granularity of max_depth
# max_depth.append(None)
# min_samples_split =[int(x) for x in np.linspace(1, 100, num=11)]  # Include larger values for min_samples_split
# min_samples_leaf = [1, 2, 4, 8, 10, 12, 14, 16, 20]  # Include larger values for min_samples_leaf
# learning_rate = [0.01, 0.1, 0.2, 0.5, 0.8, 1.0]  # Include a broader range for learning_rate

# # Create the random grid
# random_grid = {
#     'n_estimators': n_estimators,
#     'max_features': max_features,
#     'max_depth': max_depth,
#     'min_samples_split': min_samples_split,
#     'min_samples_leaf': min_samples_leaf,
#     'learning_rate': learning_rate
# }

# # Create the GradientBoostingRegressor model
# gb_reg = GradientBoostingRegressor(random_state=0)

# # Perform RandomizedSearchCV
# grid_search = RandomizedSearchCV(estimator=gb_reg,
#                                  param_distributions=random_grid,
#                                  n_iter=200,
#                                  cv=5,
#                                  verbose=2,
#                                  random_state=42,
#                                  n_jobs=-1)

# # Fit the model
# grid_search.fit(X_train, y_train)

# # Get the best results
# best_accuracy = grid_search.best_score_
# best_parameters = grid_search.best_params_

# # Print the results
# print('Best model parameters for Gradient Boosting Regressor:')
# print("Best Accuracy: {:.2f} %".format(best_accuracy * 100))
# print("Best Parameters:", best_parameters)




# from sklearn.ensemble import GradientBoostingRegressor
# import numpy as np
# import pandas as pd

# # Define the best parameters obtained from the randomized search
# #n_estimators': 1000, 'min_samples_split': 5, 'min_samples_leaf': 4, 'max_features': 'sqrt', 'max_depth': 35, 'learning_rate': 0.01
# best_params = {
#     'n_estimators':1000,
#     'min_samples_split': 5,
#     'min_samples_leaf': 4,
#     'max_features': 'sqrt',
#     'max_depth': 35,
#     'learning_rate': 0.01
# }

# # Create the GradientBoostingRegressor model with best parameters
# gb_reg = GradientBoostingRegressor(**best_params, random_state=0)

# # Fit the model on the entire dataset
# gb_reg.fit(X_train, y_train)

# # Get the feature importance scores
# feature_importance = gb_reg.feature_importances_

# # Assuming you have feature names stored separately
# feature_names = list(dataset.columns[2:37])  # Replace [...] with your actual feature names

# # Create a DataFrame to display feature importance scores
# feature_importance_df = pd.DataFrame({
#     'Feature': feature_names,
#     'Importance': feature_importance
# })

# # Sort the DataFrame by importance score
# #feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# # Print the top features

# print(feature_importance_df)
# feature_importance_df.to_csv("/Users/liberty.mguni/Downloads/cluster_drew/HEA22032025Feature_importanceSelectivity.csv")
# #print(feature_importance_df.head(10))




# print(feature_importance_df['Feature'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (25, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Fe loading (%)")
# plt.ylabel("Co loading (%)")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (25, 31)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Fe loading (%)")
# plt.ylabel("Zr loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (25, 30)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Fe loading (%) ")
# plt.ylabel("Zn loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (25, 31)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Fe loading (%)")
# plt.ylabel("K loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (25, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Fe loading (%)")
# plt.ylabel("Na loading (%)")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (5, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Space velocity (mL/h g_cat)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (7, 25)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Calcination temperature (oC)")
# plt.ylabel("Fe loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (7, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Calcination temperature (oC)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (9, 25)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Reduction temperature (oC)", fontsize=14)
# plt.ylabel("Fe loading (%) ", fontsize=14)




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (9, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Reduction temperature (oC)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (11, 25)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("BET surface area (m2/g)")
# plt.ylabel("Fe loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (11, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("BET surface area (m2/g)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (12, 25)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Pore volume (cm3/g)")
# plt.ylabel("Fe loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (12, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Pore volume (cm3/g)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (13, 25)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Pore diamiter (nm)", fontsize=14)
# plt.ylabel("Fe loading (%) ", fontsize=14)




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features = [ (13, 23)]
# PartialDependenceDisplay.from_estimator(clf, X, features)
# plt.xlabel("Pore diameter (nm)")
# plt.ylabel("Co loading (%) ")




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# pd_results1 = partial_dependence(
#     clf, X, features=4, kind="average", grid_resolution=5)
# pd_results2 = partial_dependence(
#     clf, X, features=5, kind="average", grid_resolution=5)




# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results1['grid_values'], pd_results1['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# #plt.plot(pd_results2['grid_values'], pd_results2['average'], marker='^', markersize=10, linestyle='--', color='r', linewidth=2)
# plt.xlabel("Temperature (oC)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()

# print(pd_results1['grid_values'])
# print(pd_results1['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# features, feature_names = [(20,)], [f"Features #{i}" for i in range(X.shape[1])]
# deciles = {0: np.linspace(0, 1, num=6)}
# pd_results = partial_dependence(
#     clf, X, features=3, kind="average", grid_resolution=7)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("Reactor Pressure (Bar)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()
# print(pd_results['grid_values'])
# print(pd_results['average'])





# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)

# pd_resultsSV = partial_dependence(
#     clf, X, features=23, kind="average", grid_resolution=6)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_resultsSV['grid_values'], pd_resultsSV['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("space velocity mL/h.g cat", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()

# print(pd_resultsSV['grid_values'])
# print(pd_resultsSV['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)

# pd_results = partial_dependence(
#     clf, X, features=7, kind="average", grid_resolution=7)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("calcination temperature (oC)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()

# print(pd_results['grid_values'])
# print(pd_results['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)

# pd_results = partial_dependence(
#     clf, X, features=9, kind="average", grid_resolution=7)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("Reduction temperature (oC)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()
# print(pd_results['grid_values'])
# print(pd_results['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# pd_results = partial_dependence(
#     clf, X, features=11, kind="average", grid_resolution=7)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("BET surface area (m2/g)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()

# print(pd_results['grid_values'])
# print(pd_results['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)

# pd_results = partial_dependence(
#     clf, X, features=12, kind="average", grid_resolution=7)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("Pore volume (cm3/g)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# plt.show()

# print(pd_results['grid_values'])
# print(pd_results['average'])




# from sklearn.inspection import PartialDependenceDisplay
# from sklearn.inspection import partial_dependence
# clf = GradientBoostingRegressor(**best_params, random_state=0).fit(X, y)
# pd_results = partial_dependence(
#     clf, X, features=13, kind="average", grid_resolution=8)
# # Extract feature values and partial dependence values
# # Plot the fancy scatter plot
# #plt.figure(figsize=(10, 6))
# #plt.plot(pd_results['grid_values'], pd_results['average'], c='blue', alpha=0.6, s=100)
# plt.plot(pd_results['grid_values'], pd_results['average'], marker='^', markersize=10, linestyle='--', color='b', linewidth=2)
# plt.xlabel("Pore daimeter (nm)", fontsize=14)
# plt.ylabel("Partial Dependence", fontsize=14)
# #plt.xlim([0, 40]) 
# plt.show()
# print(pd_results['grid_values'])
# print(pd_results['average'])

