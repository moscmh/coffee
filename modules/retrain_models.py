# Python Libraries
import pandas as pd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('seaborn-v0_8-colorblind')

from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Import Data
coffee = pd.read_csv('data/coffee_cleaned.csv')

# Separate by extraction method
def get_all_origins(df):
    origins = ['Ethiopia', 'Colombia', 'Brazil', 'Guatemala', 'Costa Rica', 'Kenya', 'Honduras', 'Mexico', 'Peru', 'Panama', 'Indonesia', 'Other']
    for o in origins:
        if 'Origin_' + o not in df.columns:
            df['Origin_' + o] = False
    return df
coffee = pd.get_dummies(coffee, columns=['Origin'])
coffee = get_all_origins(coffee)

espresso_coffee = coffee[coffee['Method'] == 'Espresso'].reset_index(drop=True)
filter_coffee = coffee[coffee['Method'] == 'Filter'].reset_index(drop=True)
espresso_coffee.drop(columns=['Date', 'Note', 'Method'], inplace=True)
filter_coffee.drop(columns=['Date', 'Note', 'Method'], inplace=True)

# Prepare the data for modelling
esp_X = espresso_coffee.drop(columns=['Score']).reset_index(drop=True)
fil_X = filter_coffee.drop(columns=['Score']).reset_index(drop=True)

esp_y = espresso_coffee['Score']
fil_y = filter_coffee['Score']

# Split
def stratified_split(X, y, test_size=.2, random_state=42):
    idx = []
    origins = X.iloc[0, 6:].index

    for o in origins:
        idx += X.loc[X[o] == True].sample(frac=test_size, random_state=random_state).index.to_list()
    return idx

esp_test_idx = stratified_split(esp_X, esp_y, test_size=.2, random_state=42)
esp_X_train = esp_X.drop(index=esp_test_idx).reset_index(drop=True)
esp_y_train = esp_y.drop(index=esp_test_idx).reset_index(drop=True)
esp_X_test = esp_X.loc[esp_test_idx].reset_index(drop=True)
esp_y_test = esp_y.loc[esp_test_idx].reset_index(drop=True)

fil_test_idx = stratified_split(fil_X, fil_y, test_size=.2, random_state=42)
fil_X_train = fil_X.drop(index=fil_test_idx).reset_index(drop=True)
fil_y_train = fil_y.drop(index=fil_test_idx).reset_index(drop=True)
fil_X_test = fil_X.loc[fil_test_idx].reset_index(drop=True)
fil_y_test = fil_y.loc[fil_test_idx].reset_index(drop=True)

# Model Training
esp_gb_params = fil_gb_params = {
    'n_estimators': [50, 100, 200],
    'criterion': ['squared_error', 'friedman_mse'],
    'learning_rate': [.001, .01, .05],
    'max_depth': [3, 5, 8],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

def optimise_gb_params(params, X_train, y_train):
    param_names = ['n_estimators', 'learning_rate', 'max_depth', 'min_samples_split', 'min_samples_leaf']
    optimisation = 0

    while optimisation != 5:
        optimisation = 0

        gb_grid = GridSearchCV(estimator=GradientBoostingRegressor(random_state=42), param_grid=params, cv=5, n_jobs=-1, verbose=0)
        gb_grid.fit(X_train, y_train)

        for param in param_names:
            if param == 'learning_rate':
                if gb_grid.best_params_[param] == params[param][1]:
                    optimisation += 1
                elif gb_grid.best_params_[param] == params[param][0]:
                    params[param] = [params[param][0] / 2, params[param][0], params[param][1]]
                else:
                    params[param] = [params[param][1], params[param][2], params[param][2] * 2]
            else:
                if gb_grid.best_params_[param] == params[param][1] or (param == 'min_samples_split' and params[param][0] == 2) or (param == 'max_depth' and params[param][0] == 1):
                    optimisation += 1
                elif gb_grid.best_params_[param] == params[param][0]:
                    params[param] = [int(np.round(params[param][0] / 2)), int(params[param][0]), int(params[param][1])]
                else:
                    params[param] = [int(np.round(params[param][1])), int(np.round(params[param][2])), int(np.round(params[param][2] * 2))]

    return gb_grid

esp_gb_grid = optimise_gb_params(esp_gb_params, esp_X_train, esp_y_train)
fil_gb_grid = optimise_gb_params(fil_gb_params, fil_X_train, fil_y_train)

print(esp_gb_grid.best_params_)
print(fil_gb_grid.best_params_)

# Evaluation
esp_y_pred = esp_gb_grid.best_estimator_.predict(esp_X_test)
esp_r2 = r2_score(esp_y_test, esp_y_pred)
esp_mse = mean_squared_error(esp_y_test, esp_y_pred)
print(f"ESP R2: {esp_r2:.3f}")
print(f"ESP MSE: {esp_mse:.3f}")

fil_y_pred = fil_gb_grid.best_estimator_.predict(fil_X_test)
fil_r2 = r2_score(fil_y_test, fil_y_pred)
fil_mse = mean_squared_error(fil_y_test, fil_y_pred)
print(f"FIL R2: {fil_r2:.3f}")
print(f"FIL MSE: {fil_mse:.3f}")