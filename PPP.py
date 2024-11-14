import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import time
import numpy as np
import matplotlib.pyplot as plt

def load_data():
    print("Loading data...")
    start_time = time.time()
    indicator_data = pd.read_excel('housing_data.xlsx', sheet_name='data')
    ppp_data = pd.read_excel('housing_data.xlsx', sheet_name='Ark1')
    print(f"Data loaded in {round(time.time() - start_time, 2)} seconds.")
    return indicator_data, ppp_data

def improvement_data(indicator_data, ppp_data):
    print("Processing data...")
    country_mapping = {
        'AT': 'Austria', 'BE': 'Belgium', 'BG': 'Bulgaria', 'CH': 'Switzerland',
        'CY': 'Cyprus', 'CZ': 'Czechia', 'DE': 'Germany', 'DK': 'Denmark',
        'EE': 'Estonia', 'EL': 'Greece', 'ES': 'Spain', 'FI': 'Finland',
        'FR': 'France', 'HR': 'Croatia', 'HU': 'Hungary', 'IE': 'Ireland',
        'IS': 'Iceland', 'IT': 'Italy', 'LT': 'Lithuania', 'LU': 'Luxembourg',
        'LV': 'Latvia', 'MT': 'Malta', 'NL': 'Netherlands', 'NO': 'Norway',
        'PL': 'Poland', 'PT': 'Portugal', 'RO': 'Romania', 'SE': 'Sweden',
        'SI': 'Slovenia', 'SK': 'Slovakia', 'UK': 'United Kingdom'
    }

    indicator_data['country_name'] = indicator_data['geo'].map(country_mapping)
    indicator_data = indicator_data[indicator_data['country_name'] != 'United Kingdom']

    ppp_data_melted = ppp_data.melt(id_vars=['TIME'], var_name='TIME_PERIOD', value_name='PPP_VALUE')
    ppp_data_melted = ppp_data_melted[ppp_data_melted['TIME_PERIOD'] != 2021]

    indicator_data['TIME_PERIOD'] = indicator_data['TIME_PERIOD'].astype(int)
    ppp_data_melted['TIME_PERIOD'] = ppp_data_melted['TIME_PERIOD'].astype(int)

    print(f"Unique 'geo' in indicator_data: {indicator_data['geo'].unique()}")
    print(f"Unique 'TIME_PERIOD' in indicator_data: {indicator_data['TIME_PERIOD'].unique()}")
    print(f"Unique 'TIME' in ppp_data_melted: {ppp_data_melted['TIME'].unique()}")
    print(f"Unique 'TIME_PERIOD' in ppp_data_melted: {ppp_data_melted['TIME_PERIOD'].unique()}")

    combined_data = pd.merge(indicator_data, ppp_data_melted, left_on=['country_name', 'TIME_PERIOD'], right_on=['TIME', 'TIME_PERIOD'], how='inner')
    if combined_data.empty:
        print("Combined data is empty after merging.")
        return None, None

    train_data = combined_data[combined_data['TIME_PERIOD'] < 2019]
    test_data = combined_data[combined_data['TIME_PERIOD'] >= 2019]

    print(f"Training data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")

    return train_data, test_data

def train_model(train_data, test_data):
    # Prepare training and test data
    X_train = train_data.drop(columns=['PPP_VALUE', 'geo', 'STRUCTURE', 'STRUCTURE_ID', 'na_item', 'ppp_cat', 'OBS_FLAG', 'country_name'])
    y_train = train_data['PPP_VALUE']

    X_test = test_data.drop(columns=['PPP_VALUE', 'geo', 'STRUCTURE', 'STRUCTURE_ID', 'na_item', 'ppp_cat', 'OBS_FLAG', 'country_name'])
    y_test = test_data['PPP_VALUE']

    X_train = X_train.select_dtypes(include=['number'])
    X_test = X_test.select_dtypes(include=['number'])

    # Split training data into training and validation data
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialize and tune model with hyperparameter optimization
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_

    print("Best parameters found: ", grid_search.best_params_)

    # Train the best model
    best_model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = best_model.predict(X_val)
    mse = mean_squared_error(y_val, y_pred)
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_val, y_pred)

    # Print the metrics
    print(f"Mean Squared Error (MSE) on validation data: {mse}")
    print(f"Mean Absolute Error (MAE) on validation data: {mae}")
    print(f"Root Mean Squared Error (RMSE) on validation data: {rmse}")
    print(f"R² (R-squared) on validation data: {r2}")

    # Cross-validation
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
    print("Cross-validation MSE scores: ", cv_scores)

    # Plotting predictions vs. actual values
    plt.scatter(y_val, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Actual vs Predicted values")
    plt.show()

    return best_model, mse, mae, rmse, r2, cv_scores

# Main execution flow
indicator_data, ppp_data = load_data()
train_data, test_data = improvement_data(indicator_data, ppp_data)

if train_data is None or test_data is None:
    print("Error in data processing. Cannot proceed with model training.")
else:
    best_model, mse, mae, rmse, r2, cv_scores = train_model(train_data, test_data)
