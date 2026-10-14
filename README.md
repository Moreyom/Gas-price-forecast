# TTF Gas Price Forecast

To jest notebook do prognozowania cen gazu TTF na horyzont 1–12 miesięcy.

## Co zawiera
- wczytanie danych z pliku CSV
- agregację do miesięcy
- utworzenie cech (lagi, rolling mean, sezonowość)
- trenowanie modelu XGBoost
- walidację na ostatnich 12 miesiącach
- prognozę na kolejne 12 miesięcy
- wykresy i zapis wyników do pliku

## Wymagania
- Python 3.10+
- notebook Jupyter / VS Code / Codespaces

## Instalacja
```bash
pip install -r requirements.txt
```

## Jak używać
1. Wgraj plik z danymi do katalogu repozytorium.
   - nazwa domyślna: `ttf_prices.csv`
   - wymagane kolumny: `date, price`
   - format daty: `YYYY-MM-DD`
2. Otwórz notebook `forecast.ipynb`.
3. Uruchom komórki po kolei.
4. Wyniki pojawią się w notebooku, a dodatkowo zostaną zapisane do plików:
   - `forecast_output.csv`
   - `forecast_plot.png`
   - `xgb_ttf_model.joblib`

## Przykład danych CSV
```csv
date,price
2020-01-01,20.15
2020-02-01,21.40
2020-03-01,22.10
```

## Uwaga
Jeśli nie masz danych TTF, możesz wgrać dowolny plik z cenami gazu w tej samej strukturze. Model będzie działał na tej samej zasadzie.
