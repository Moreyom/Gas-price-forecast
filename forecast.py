# forecast.py
# Prosty miesięczny forecast TTF (horyzont 12 miesięcy)
# Użycie:
# pip install -r requirements.txt
# python forecast.py

import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt

# Spróbuj załadować xgboost, jeśli nie będzie dostępny — użyj RandomForest
try:
    from xgboost import XGBRegressor
    USE_XGB = True
except Exception:
    from sklearn.ensemble import RandomForestRegressor
    USE_XGB = False

DATA_FILE = 'ttf_prices.csv'
FORECAST_HORIZON = 12


def load_data(path=DATA_FILE):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} nie znaleziony. Wgraj ttf_prices.csv do repo.")
    df = pd.read_csv(path)
    if 'date' not in df.columns or 'price' not in df.columns:
        raise ValueError('CSV musi mieć kolumny: date,price')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date').sort_index()
    df = df['price'].resample('MS').mean().to_frame()
    return df


def create_features(df):
    d = df.copy()
    d['lag_1'] = d['price'].shift(1)
    d['lag_3'] = d['price'].shift(3)
    d['lag_12'] = d['price'].shift(12)
    d['rmean_3'] = d['price'].rolling(3).mean().shift(1)
    d['rmean_12'] = d['price'].rolling(12).mean().shift(1)
    d['month'] = d.index.month
    d = d.dropna()
    return d


def train_and_evaluate(df_features, test_months=12):
    X = df_features.drop(columns=['price'])
    y = df_features['price']
    if len(X) <= test_months:
        raise ValueError('Za mało danych, aby wydzielić test.')
    X_train, X_test = X[:-test_months], X[-test_months:]
    y_train, y_test = y[:-test_months], y[-test_months:]
    if USE_XGB:
        model = XGBRegressor(n_estimators=100, learning_rate=0.05, max_depth=4, random_state=42)
    else:
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    print(f"Ewaluacja na ostatnich {test_months} miesiącach: RMSE={rmse:.3f}, MAE={mae:.3f}")
    return model, X_test, y_test, preds


def recursive_forecast(model, df, horizon=FORECAST_HORIZON):
    history = df.copy()
    preds = []
    for i in range(horizon):
        next_date = history.index[-1] + pd.offsets.MonthBegin(1)
        lag_1 = history['price'].iloc[-1]
        lag_3 = history['price'].iloc[-3] if len(history) >= 3 else lag_1
        lag_12 = history['price'].iloc[-12] if len(history) >= 12 else lag_1
        rmean_3 = history['price'].rolling(3).mean().iloc[-1]
        rmean_12 = history['price'].rolling(12).mean().iloc[-1]
        month = next_date.month
        X_next = pd.DataFrame({'lag_1':[lag_1],'lag_3':[lag_3],'lag_12':[lag_12],
                               'rmean_3':[rmean_3],'rmean_12':[rmean_12],'month':[month]}, index=[next_date])
        yhat = model.predict(X_next)[0]
        preds.append((next_date, yhat))
        history.loc[next_date] = yhat
    forecast_df = pd.DataFrame(preds, columns=['date','forecast']).set_index('date')
    return forecast_df


def plot_and_save(history, X_test_index, y_test, test_preds, forecast_df):
    plt.figure(figsize=(10,6))
    plt.plot(history.index, history['price'], label='history')
    if len(X_test_index)>0:
        plt.plot(X_test_index, y_test, label='test actual', marker='o')
        plt.plot(X_test_index, test_preds, label='test preds', marker='x')
    plt.plot(forecast_df.index, forecast_df['forecast'], label='forecast', marker='s')
    plt.legend()
    plt.title('TTF price - history, test, forecast')
    plt.tight_layout()
    plt.savefig('forecast_plot.png')
    print('Zapisano forecast_plot.png')


def main():
    print('Wczytuję dane...')
    df = load_data()
    print(f'Dane: {len(df)} miesięcy, od {df.index.min().date()} do {df.index.max().date()}')
    df_features = create_features(df)
    print(f'Z cechami: {len(df_features)} rekordów')
    try:
        model, X_test, y_test, preds = train_and_evaluate(df_features, test_months=12)
    except Exception as e:
        print('Błąd podczas trenowania:', str(e))
        return
    joblib.dump(model, 'xgb_ttf_model.joblib')
    print('Zapisano model xgb_ttf_model.joblib')
    forecast_df = recursive_forecast(model, df)
    forecast_df.to_csv('forecast_output.csv')
    print('Zapisano forecast_output.csv')
    plot_and_save(df, X_test.index if X_test is not None else [], y_test if y_test is not None else [], preds if preds is not None else [], forecast_df)
    print('Gotowe.')

if __name__ == '__main__':
    main()
