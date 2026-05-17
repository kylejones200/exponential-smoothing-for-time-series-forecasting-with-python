"""Generated from Jupyter notebook: 2025-04-04 Time Series ERCOT exponential smoothingTime

Magics and shell lines are commented out. Run with a normal Python interpreter."""
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_ercot_data() -> None:
    df = pd.read_csv('ercot_load_data.csv')

    df['date'] = pd.to_datetime(df['date'])

    df.set_index('date', inplace=True)

    df.sort_index(inplace=True)

    df = df[df['values'] >= 60]

    values = df['values']

    dates = df.index

    scaler_minmax = MinMaxScaler()

    scaler_std = StandardScaler()

    values_scaled_minmax = scaler_minmax.fit_transform(values.values.reshape(-1, 1))

    values_scaled_std = scaler_std.fit_transform(values.values.reshape(-1, 1))

    print('Min-Max Scaled Values:\n', values_scaled_minmax.flatten())

    print('Standardized Values:\n', values_scaled_std.flatten())

    values_diff = values.diff().dropna()

    print('First Difference:\n', values_diff.head())

    values_gradient = np.gradient(values)

    values_acceleration = np.gradient(values_gradient)

    print('First Derivative (Rate of Change):\n', values_gradient[:5])

    print('Second Derivative (Acceleration):\n', values_acceleration[:5])

    rolling_mean = values.rolling(window=3).mean()

    print('Rolling Mean (window=3):\n', rolling_mean.head())

    seasonal_decomposition = seasonal_decompose(values, period=12, model='additive')

    print('Seasonal Component Head:\n', seasonal_decomposition.seasonal.head())

    df_features = pd.DataFrame({'value': values, 'lag_1': values.shift(1), 'lag_2': values.shift(2), 'rate_of_change': values.diff()}).dropna()

    df_features['month'] = df_features.index.month

    df_features['year'] = df_features.index.year

    print('Time-Based Features:\n', df_features[['month', 'year']].head())

    plt.figure(figsize=(10, 6))

    plt.plot(values, label='Original Values', color='Blue')

    plt.plot(rolling_mean, label='Rolling Mean (window=3)', color='Red')

    plt.title('Original Values and Rolling Mean')

    plt.xlabel('Date')

    plt.ylabel('Values')

    plt.legend()

    plt.savefig('rolling_mean_ercot.png')

    plt.show()


def load_the_ercot_data() -> None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from darts import TimeSeries
    from darts.metrics import mape
    from darts.models import ExponentialSmoothing

    # Load the ERCOT data
    df = pd.read_csv("ercot_load_data.csv")
    df["date"] = pd.to_datetime(df["date"])  # Ensure 'date' is in datetime format
    df["values"] = pd.to_numeric(
        df["values"], errors="coerce"
    )  # Convert 'values' to numeric
    df = df.sort_values("date")  # Sort by date

    # Drop rows with missing or NaN values
    df = df.dropna()

    # Resample the data to hourly frequency (mean aggregation)
    df = (
        df.set_index("date").resample("H").mean().reset_index()
    )  # Resample to hourly frequency

    # Define hold-out period (e.g., last 24 hours)
    hold_out_hours = 24  # Hold-out size (24 hours = 1 day)
    train = df.iloc[:-hold_out_hours]
    hold_out = df.iloc[-hold_out_hours:]

    # Create TimeSeries for training and hold-out data
    series_train = TimeSeries.from_dataframe(
        train, "date", "values", freq="H", fill_missing_dates=True
    )
    series_hold_out = TimeSeries.from_dataframe(hold_out, "date", "values", freq="H")

    # Fit the Exponential Smoothing model on training data
    model = ExponentialSmoothing()
    model.fit(series_train)

    # Forecast the hold-out period
    forecast = model.predict(len(series_hold_out))

    # Calculate MAPE (Mean Absolute Percentage Error)
    mape_result = mape(series_hold_out, forecast)

    # Plot the results
    plt.figure(figsize=(12, 6))

    # Plot training data
    plt.plot(
        series_train.time_index, series_train.values(), label="Training Data", color="blue"
    )

    # Plot hold-out data
    plt.plot(
        series_hold_out.time_index,
        series_hold_out.values(),
        label="Hold-Out Data (Actual)",
        color="green",
    )

    # Plot forecasted data
    plt.plot(forecast.time_index, forecast.values(), label="Forecast", color="red")

    # Add title and labels
    plt.title(f"ERCOT Hourly Load Forecast with Hold-Out Data \n MAPE: {mape_result:.2f}%")
    plt.xlabel("Date")
    plt.ylabel("Load Values")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot
    plt.savefig("ERCOT_Hourly_HoldOut_Forecast.png")
    plt.show()


def ensure_train_and_hold_out_datasets_are_already_d() -> None:
    scaler_minmax = MinMaxScaler()

    scaler_std = StandardScaler()

    train_values = train['values'].values.reshape(-1, 1)

    values_scaled_minmax = scaler_minmax.fit_transform(train_values)

    values_scaled_std = scaler_std.fit_transform(train_values)

    print('Min-Max Scaled Train Values:\n', values_scaled_minmax.flatten())

    print('Standardized Train Values:\n', values_scaled_std.flatten())

    values_diff = train['values'].diff().dropna()

    print('First Difference:\n', values_diff.head())

    values_gradient = np.gradient(train['values'].values)

    values_acceleration = np.gradient(values_gradient)

    print('First Derivative (Rate of Change):\n', values_gradient[:5])

    print('Second Derivative (Acceleration):\n', values_acceleration[:5])

    rolling_mean = train['values'].rolling(window=3).mean()

    print('Rolling Mean (window=3):\n', rolling_mean.head())

    seasonal_decomposition = seasonal_decompose(train['values'], period=12, model='additive')

    print('Seasonal Component Head:\n', seasonal_decomposition.seasonal.head())

    df_features_train = pd.DataFrame({'value': train['values'], 'lag_1': train['values'].shift(1), 'lag_2': train['values'].shift(2), 'rate_of_change': train['values'].diff()}).dropna()

    df_features_train['month'] = train['date'].dt.month

    df_features_train['year'] = train['date'].dt.year

    print('Time-Based Features (Train Data):\n', df_features_train[['month', 'year']].head())

    plt.figure(figsize=(10, 6))

    plt.plot(train['date'], train['values'], label='Train Values', color='Blue')

    plt.plot(train['date'], rolling_mean, label='Rolling Mean (window=3)', color='Red')

    plt.plot(hold_out['date'], hold_out['values'], label='Hold-Out Values', color='Green')

    plt.title('Hold-Out Values')

    plt.xlabel('Date')

    plt.ylabel('Values')

    plt.legend()

    plt.grid(True)

    plt.savefig('holdout_values_ercot.png')

    plt.show()


def fit_an_exponential_smoothing_model_on_the_traini() -> None:
    model = ExponentialSmoothing(train['values'], trend='additive', seasonal='additive', seasonal_periods=24)

    fitted_model = model.fit()

    forecast_values = fitted_model.forecast(steps=len(hold_out))

    hold_out['forecast'] = forecast_values.values

    mape_value = mean_absolute_percentage_error(hold_out['values'], hold_out['forecast'])

    print(f'MAPE: {mape_value * 100:.2f}%')

    plt.figure(figsize=(12, 6))

    plt.plot(train['date'], train['values'], label='Training Data', color='blue')

    plt.plot(hold_out['date'], hold_out['values'], label='Hold-Out Data (Actual)', color='green')

    plt.plot(hold_out['date'], hold_out['forecast'], label='Forecast', color='red', linestyle='--')

    plt.title(f'Forecasting Hold-Out Data \n MAPE: {mape_value * 100:.2f}%')

    plt.xlabel('Date')

    plt.ylabel('Load Values')

    plt.legend()

    plt.grid(True)

    plt.tight_layout()

    plt.savefig('forecast_holdout_ercot.png')

    plt.show()


def main() -> None:
    load_ercot_data()
    load_the_ercot_data()
    ensure_train_and_hold_out_datasets_are_already_d()
    fit_an_exponential_smoothing_model_on_the_traini()

if __name__ == "__main__":
    main()
