from sklearn.feature_selection import VarianceThreshold
import statsmodels.api as sm
from sklearn.feature_selection import SelectFromModel, RFE
import numpy as np
import pandas as pd


def variance_threshold_selector(X, threshold=0.15):
    selector = VarianceThreshold(threshold)
    selector.fit(X)
    support_var_selec = selector.get_support()
    features_var_selec = X.iloc[:, support_var_selec].columns.to_list()
    return support_var_selec, features_var_selec


def feature_importance_selector(model, X, Y, threshold=0.01):

    rf_selector = SelectFromModel(model, threshold=threshold)
    rf_selector.fit(X, Y)

    support_rf = rf_selector.get_support()
    features_rf = X.iloc[:, support_rf].columns.to_list()

    return support_rf, features_rf


def recursive_selection(model, X, Y):

    rfe_selector = RFE(model, n_features_to_select=10, step=0.2, verbose=2)
    rfe_selector.fit(X, Y)

    support_rfe = rfe_selector.get_support()
    features_rfe = X.iloc[:, support_rfe].columns.to_list()

    return support_rfe, features_rfe


def F_test_selector(X, Y, p_value=0.05):
    '''
    Méthodes de selection de features basée sur le test de fisher.
    Pour un feature Xj, on fait une regression (linéaire ici) puis on regarde si la variable est significative
    via le test de fisher. 

    X : features
    Y : variables dépendante
    p_value : p_value à laquelle on regarde la significativité
    '''

    p_values = []
    feat_name = X.columns.to_list()
    # fisher test for each variable
    for var in X.columns.to_list():
        X_train = X[var]
        ols_model = sm.OLS(Y, X_train)
        ols_results = ols_model.fit()
        p_values.append(ols_results.f_pvalue)
    # replace nan
    p_values = [1 if np.isnan(i) else i for i in p_values]
    # features selected
    num_features = sum(i <= p_value for i in p_values)
    features = X.iloc[:, np.argsort(
        p_values)[-num_features:]].columns.to_list()
    # 0 for non selected 1 for selected
    support = [True if i in features else False for i in feat_name]
    return feat_name, support, features


def select_best_features(X, Y, p_value, var_threshold, model, feat_impor_threshold):
    '''
   Lance trois méthodes de selection de variables pertinentes et retourne un tableau pour dire si une variable
   est sélectionnée ou pas. 

    X : features
    Y : variables dépendantes
    threshold : seuil de selection pour les aggregation de modèle
    p_value : p_value à laquelle on regarde la significativité pour le test de fisher
    ...

    '''

    # Fisher
    print("::: Applying F_test_selector")
    feat_name, support_Ftest, features_Ftest = F_test_selector(X, Y, p_value)
    print("Selected features:")
    print(features_Ftest)
    print("\n")

    # Using feature importance of a model
    print("::: Applying feature_importance_selector")
    support_impor, features_impor = feature_importance_selector(
        model, X, Y, feat_impor_threshold
    )
    print("Selected features:")
    print(features_impor)
    print("\n")

    # Using variance threshold
    print("::: Applying variance_threshold_selector")
    support_var_selec, features_var_selec = variance_threshold_selector(
        X, var_threshold
    )
    print("Selected features:")
    print(features_var_selec)
    print("\n")

    # Using recursive feature elimination
    print("::: Applying recursive_selection")
    support_rfe, features_rfe = recursive_selection(model, X, Y)
    print("Selected features:")
    print(features_rfe)
    print("\n")

    # gathering
    print("::: Gathering results from all methods")

    feature_selected_df = pd.DataFrame({
        'Feature': feat_name,
        'Ftest': support_Ftest,
        'Importance': support_impor,
        'variance': support_var_selec,
        "RFE": support_rfe
    })

    feature_selected_df['Total_Selection'] = np.sum(
        feature_selected_df, axis=1
    )

    feature_selected_df = feature_selected_df.sort_values(
        'Total_Selection', ascending=False
    ).reset_index(drop=True)

    return feature_selected_df