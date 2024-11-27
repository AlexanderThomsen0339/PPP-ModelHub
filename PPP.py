import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import time
import matplotlib.pyplot as plt
import lightgbm as lgb
import numpy as np
import seaborn as sns

# Funktion til at indlæse data
def load_data():
    print("Loading data...")  # Udskrivning, der angiver, at data bliver indlæst
    start_time = time.time()  # Starttidspunkt for at måle, hvor lang tid dataindlæsningen tager
    # Læs data fra Excel-filerne
    indicator_data = pd.read_excel('housing_data.xlsx', sheet_name='data')
    ppp_data = pd.read_excel('housing_data.xlsx', sheet_name='Ark1')
    # Beregn og udskriv hvor lang tid dataindlæsningen tog
    print(f"Data loaded in {round(time.time() - start_time, 2)} seconds.")
    return indicator_data, ppp_data  # Returnér de indlæste data

# Funktion til at forbehandle og kombinere data
def improvement_data(indicator_data, ppp_data):
    print("Processing data...")  # Udskrivning, der indikerer, at data bliver behandlet
    # Kortlægning af lande-koder til lande-navne
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
    
    # Tilføj en ny kolonne 'country_name' baseret på 'geo' kolonnen
    indicator_data['country_name'] = indicator_data['geo'].map(country_mapping)
    # Fjern 'United Kingdom' fra datasættet, da det ikke er relevant
    indicator_data = indicator_data[indicator_data['country_name'] != 'United Kingdom']

    # Smelt PPP data (transformere data fra bred format til lang format)
    ppp_data_melted = ppp_data.melt(id_vars=['TIME'], var_name='TIME_PERIOD', value_name='PPP_VALUE')
    ppp_data_melted = ppp_data_melted[ppp_data_melted['TIME_PERIOD'] != 2021]  # Fjern data for 2021

    # Sørg for, at 'TIME_PERIOD' er af type int i begge datasæt
    indicator_data['TIME_PERIOD'] = indicator_data['TIME_PERIOD'].astype(int)
    ppp_data_melted['TIME_PERIOD'] = ppp_data_melted['TIME_PERIOD'].astype(int)

    # Udskriv unikke værdier for at tjekke datasættet
    print(f"Unique 'geo' in indicator_data: {indicator_data['geo'].unique()}")
    print(f"Unique 'TIME_PERIOD' in indicator_data: {indicator_data['TIME_PERIOD'].unique()}")
    print(f"Unique 'TIME' in ppp_data_melted: {ppp_data_melted['TIME'].unique()}")
    print(f"Unique 'TIME_PERIOD' in ppp_data_melted: {ppp_data_melted['TIME_PERIOD'].unique()}")

    # Kombiner de to datasæt på 'country_name' og 'TIME_PERIOD' kolonnerne
    combined_data = pd.merge(indicator_data, ppp_data_melted, left_on=['country_name', 'TIME_PERIOD'], right_on=['TIME', 'TIME_PERIOD'], how='inner')
    
    # Tjek, om det kombinerede datasæt er tomt
    if combined_data.empty:
        print("Combined data is empty after merging.")
        return None, None

    # Opdel i trænings- og testdata baseret på år
    train_data = combined_data[combined_data['TIME_PERIOD'] < 2019]
    test_data = combined_data[combined_data['TIME_PERIOD'] >= 2019]

    print(f"Training data: {len(train_data)} rows")
    print(f"Test data: {len(test_data)} rows")

    return train_data, test_data  # Returnér trænings- og testdatasættene

# Beregn MAPE (Mean Absolute Percentage Error)
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100  # MAPE-formel

# Evaluer modellen og vis resultaterne
def evaluate_model(y_true, y_pred, country='Denmark'):
    mse = mean_squared_error(y_true, y_pred)  # Mean Squared Error
    mae = mean_absolute_error(y_true, y_pred)  # Mean Absolute Error
    rmse = np.sqrt(mse)  # Root Mean Squared Error
    r2 = r2_score(y_true, y_pred)  # R²-score
    mape = mean_absolute_percentage_error(y_true, y_pred)  # MAPE

    print(f"MSE: {mse}, MAE: {mae}, RMSE: {rmse}, R²: {r2}, MAPE: {mape}")

    # Hvis 'y_true' indeholder landene, filtrer kun for det ønskede land
    if isinstance(y_true, pd.Series) and 'country_name' in y_true.index.names:
        print(f"Filtering results for {country}")
        country_data = y_true.index.get_level_values('country_name') == country
        y_true = y_true[country_data]
        y_pred = y_pred[country_data]

    # Plot faktiske vs forudsigede værdier
    plt.figure(figsize=(8, 6))
    plt.plot(y_true.values, label='Actual', color='blue', alpha=0.7)
    plt.plot(y_pred, label='Predicted', color='orange', alpha=0.7)
    plt.xlabel("Index")
    plt.ylabel("PPP Value")
    title = f"Actual vs Predicted Values for {country}"
    plt.title(title)
    plt.legend()
    plt.show()

    # Plot residualer
    residuals = y_true - y_pred
    plt.figure(figsize=(8, 6))
    sns.histplot(residuals, kde=True, color='red', bins=30)
    plt.xlabel("Residuals")
    plt.title(f"Residuals Distribution for {country}")
    plt.show()

    return mse, mae, rmse, r2, mape  # Returnér evalueringsmålene

# Træn Random Forest-model
def train_RFG_model(train_data, test_data):
    # Forbered trænings- og testdata
    X_train = train_data.drop(columns=['PPP_VALUE', 'geo', 'STRUCTURE', 'STRUCTURE_ID', 'na_item', 'ppp_cat', 'OBS_FLAG', 'country_name'])
    y_train = train_data['PPP_VALUE']

    X_test = test_data.drop(columns=['PPP_VALUE', 'geo', 'STRUCTURE', 'STRUCTURE_ID', 'na_item', 'ppp_cat', 'OBS_FLAG', 'country_name'])
    y_test = test_data['PPP_VALUE']

    X_train = X_train.select_dtypes(include=['number'])  # Vælg kun numeriske kolonner
    X_test = X_test.select_dtypes(include=['number'])

    # Split træningsdata i træning og validation
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    # Initialiser og tun model med hyperparameteroptimering
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Definer parametergrid for GridSearch
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # GridSearch for at finde de bedste hyperparametre
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_  # Vælg bedste model

    print("Best parameters found: ", grid_search.best_params_)

    # Træn den bedste model
    best_model.fit(X_train, y_train)

    # Forudsig på testdata
    y_pred = best_model.predict(X_test)
    
    # Evaluer modellen
    return evaluate_model(y_test, y_pred)
