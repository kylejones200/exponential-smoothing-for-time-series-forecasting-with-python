"""Generated from Jupyter notebook: Exponential Smoothing for Time Series Forecasting with Python

Magics and shell lines are commented out. Run with a normal Python interpreter."""

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing


def generate_synthetic_data_constant_with_slight_noi() -> None:
    np.random.seed(42)

    time = np.arange(30)

    data = 10 + np.random.normal(scale=0.5, size=len(time))

    model = SimpleExpSmoothing(data)

    fit = model.fit(smoothing_level=0.5, optimized=False)

    smoothed = fit.fittedvalues

    plt.figure(figsize=(10, 6))

    plt.plot(time, data, label="Original Data", marker="o")

    plt.plot(time, smoothed, label="SES Smoothed", color="red")

    plt.title("Simple Exponential Smoothing")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("SES_example.png")

    plt.show()


def generate_synthetic_data_linear_trend_with_noise() -> None:
    data = 10 + 0.5 * time + np.random.normal(scale=1.0, size=len(time))

    model = ExponentialSmoothing(data, trend="add", seasonal=None)

    fit = model.fit(smoothing_level=0.5, smoothing_trend=0.5, optimized=False)

    smoothed = fit.fittedvalues

    plt.figure(figsize=(10, 6))

    plt.plot(time, data, label="Original Data", marker="o")

    plt.plot(time, smoothed, label="DES Smoothed", color="red")

    plt.title("Double Exponential Smoothing")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("DES_example.png")

    plt.show()


def generate_synthetic_seasonal_data() -> None:
    data = (
        10
        + 0.5 * time
        + 2 * np.sin(2 * np.pi * time / 12)
        + np.random.normal(scale=1.0, size=len(time))
    )

    model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=12)

    fit = model.fit(
        smoothing_level=0.5,
        smoothing_trend=0.5,
        smoothing_seasonal=0.5,
        optimized=False,
    )

    smoothed = fit.fittedvalues

    plt.figure(figsize=(10, 6))

    plt.plot(time, data, label="Original Data", marker="o")

    plt.plot(time, smoothed, label="Holt-Winters Smoothed", color="red")

    plt.title("Triple Exponential Smoothing (Holt-Winters)")

    plt.xlabel("Time")

    plt.ylabel("Value")

    plt.legend()

    plt.savefig("Holt_Winters_example.png")

    plt.show()


def main() -> None:
    generate_synthetic_data_constant_with_slight_noi()
    generate_synthetic_data_linear_trend_with_noise()
    generate_synthetic_seasonal_data()


if __name__ == "__main__":
    main()
