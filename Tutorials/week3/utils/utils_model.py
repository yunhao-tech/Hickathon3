from sklearn.metrics import mean_absolute_error
from sklearn import tree
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all='warn')


def plot_importance(model, X_train):
    '''Plot features importance of the model
    rf_model : Random Forest or Decission Tree model fitted
    X_train : the dataframe on which the model was fitted
    '''

    max_features_to_show = min(20, len(X_train.columns))

    var_exp = X_train.columns
    importances = model.feature_importances_

    indices = np.argsort(importances)[- max_features_to_show:]

    # Plot the feature importances of the forest
    plt.figure(figsize=(14, 8))
    plt.title("Feature importances")
    plt.barh(range(max_features_to_show), importances[indices],
             color="r",  align="center")
    # xerr=std[indices],
    plt.yticks(range(max_features_to_show),
               X_train[var_exp].columns[indices])
    plt.ylim([-1, max_features_to_show])
    plt.show()


def plot_max_depth_influence(max_depth_ls, X_val, y_val, X_test, y_test):
    """Fit a Decision Tree with different maximum depths
    and plot learning curve for train and test sets.
    """

    mae_train = []
    mae_test = []

    # loop over different maximum depths
    # and compute MAE
    for max_depth in max_depth_ls:
        print(f"Fitting with max_depth = {max_depth}")
        reg = tree.DecisionTreeRegressor(max_depth=max_depth)
        reg = reg.fit(X_val, y_val)
        y_pred_val = reg.predict(X_val)
        y_pred_test = reg.predict(X_test)
        mae_train.append(mean_absolute_error(y_val, y_pred_val))
        mae_test.append(mean_absolute_error(y_test, y_pred_test))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    # Plot learning curves
    axes.plot(max_depth_ls, mae_train, color='red', label='Train')
    axes.plot(max_depth_ls, mae_test, color='blue', label='Validation')
    axes.set_title('MAE regarding max_depth')
    axes.legend()
    plt.xlabel("max_depth")
    plt.ylabel("MAE")
    plt.grid()
    fig.show()


def plot_n_estimators_influence(n_estimators_ls, X_train, y_train, X_test, y_test):
    """Fit a Random Forest with different number of trees
    and plot learning curve for train and test sets.
    """

    mae_train = []
    mae_test = []

    # loop over different number of trees
    # and compute MAE
    for n_estimators in n_estimators_ls:
        print(f"Fitting with n_estimators = {n_estimators}")
        reg = RandomForestRegressor(n_estimators=n_estimators,
                                    max_depth=16,
                                    # max_features="sqrt",
                                    random_state=0, n_jobs=-1)
        reg = reg.fit(X_train, y_train)
        y_pred_train = reg.predict(X_train)
        y_pred_test = reg.predict(X_test)
        mae_train.append(mean_absolute_error(y_train, y_pred_train))
        mae_test.append(mean_absolute_error(y_test, y_pred_test))

    # Plot learning curves
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    axes.plot(n_estimators_ls, mae_train, color='red', label='Train')
    axes.plot(n_estimators_ls, mae_test, color='blue', label='Validation')
    axes.set_title('MAE regarding n_estimators')
    axes.legend()
    plt.xlabel("n_estimators")
    plt.ylabel("MAE")
    plt.grid()
    fig.show()
    
    
def plot_alpha_influence_Lasso(alpha_values, X_val, y_val, X_test, y_test):
    """Fit a Decision Tree with different maximum depths
    and plot learning curve for train and test sets.
    """

    mae_train = []
    mae_test = []

    # loop over different maximum depths
    # and compute MAE
    for alpha in alpha_values:
        print(f"Fitting with max_depth = {alpha}")
        reg = linear_model.Lasso(alpha=alpha)
        reg = reg.fit(X_val, y_val)
        y_pred_val = reg.predict(X_val)
        y_pred_test = reg.predict(X_test)
        mae_train.append(mean_absolute_error(y_val, y_pred_val))
        mae_test.append(mean_absolute_error(y_test, y_pred_test))

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(16, 8))

    # Plot learning curves
    axes.plot(alpha_values, mae_train, color='red', label='Train')
    axes.plot(alpha_values, mae_test, color='blue', label='Validation')
    axes.set_title('MAE regarding alpha')
    axes.legend()
    plt.xlabel("alpha")
    plt.ylabel("MAE")
    plt.grid()
    fig.show()