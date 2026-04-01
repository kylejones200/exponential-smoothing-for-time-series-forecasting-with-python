# Description: Short example for Exponential Smoothing for Time Series Forecasting with Python.



from data_io import read_csv
from dataclasses import dataclass
from pathlib import Path
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



# Generate synthetic data (constant with slight noise)
np.random.seed(42)
time = np.arange(30)
data = 10 + np.random.normal(scale=0.5, size=len(time))

# Apply Simple Exponential Smoothing
model = SimpleExpSmoothing(data)
fit = model.fit(smoothing_level=0.5, optimized=False)
smoothed = fit.fittedvalues

# Plot the data and smoothed values
plt.figure(figsize=(10, 6))
plt.plot(time, data, label="Original Data", marker="o")
plt.plot(time, smoothed, label="SES Smoothed", color="red")
plt.title("Simple Exponential Smoothing")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig("SES_example.png")
plt.show()


# Generate synthetic data (linear trend with noise)
data = 10 + 0.5 * time + np.random.normal(scale=1.0, size=len(time))

# Apply Double Exponential Smoothing
model = ExponentialSmoothing(data, trend="add", seasonal=None)
fit = model.fit(smoothing_level=0.5, smoothing_trend=0.5, optimized=False)
smoothed = fit.fittedvalues

# Plot the data and smoothed values
plt.figure(figsize=(10, 6))
plt.plot(time, data, label="Original Data", marker="o")
plt.plot(time, smoothed, label="DES Smoothed", color="red")
plt.title("Double Exponential Smoothing")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig("DES_example.png")
plt.show()

# Generate synthetic seasonal data
data = 10 + 0.5 * time + 2 * np.sin(2 * np.pi * time / 12) + np.random.normal(scale=1.0, size=len(time))

# Apply Triple Exponential Smoothing (Holt-Winters)
model = ExponentialSmoothing(data, trend="add", seasonal="add", seasonal_periods=12)
fit = model.fit(smoothing_level=0.5, smoothing_slope=0.5, smoothing_seasonal=0.5, optimized=False)
smoothed = fit.fittedvalues

# Plot the data and smoothed values
plt.figure(figsize=(10, 6))
plt.plot(time, data, label="Original Data", marker="o")
plt.plot(time, smoothed, label="Holt-Winters Smoothed", color="red")
plt.title("Triple Exponential Smoothing (Holt-Winters)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.savefig("Holt_Winters_example.png")
plt.show()


np.random.seed(42)
plt.rcParams.update({
    'axes.grid': False,'font.family': 'serif','axes.spines.top': False,'axes.spines.right': False,'axes.linewidth': 0.8})

def save_fig(path: str):
    plt.tight_layout(); plt.savefig(path, bbox_inches='tight'); plt.close()

@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12


def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = read_csv(p, header=None, usecols=[0,1], names=["date","value"], sep=",")
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d", errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    s = df.dropna().sort_values("date").set_index("date")["value"].asfreq(cfg.freq)
    return s.astype(float)


def rolling_origin_ets(y: pd.Series, cfg: Config):
    idx = np.arange(len(y))
    tscv = TimeSeriesSplit(n_splits=cfg.n_splits)
    maes = []
    last_true, last_pred = None, None
    for tr, te in tscv.split(idx):
        end = tr[-1]
        y_tr = y.iloc[: end + 1]
        y_te = y.iloc[end + 1 : end + 1 + cfg.horizon]
        if len(y_te) == 0:
            continue
        ets = ExponentialSmoothing(y_tr, trend='add', seasonal='add', seasonal_periods=cfg.season).fit(optimized=True)
        f = ets.forecast(len(y_te)).to_numpy()
        m = mean_absolute_error(y_te.values, f)
        logger.info(f"Fold MAE: {m:.3f}")
        maes.append(m)
        last_true, last_pred = y_te, pd.Series(f, index=y_te.index)
    return float(np.mean(maes)), last_true, last_pred


def main():
    cfg = Config()
    y = load_series(cfg)
    mean_mae, _, _ = rolling_origin_ets(y, cfg)
    logger.info(f"ETS mean MAE: {mean_mae}")

    # Tufte-style figure focused on 2024 history and Jan-Aug 2025 forecast vs actuals
    start_2024 = pd.Period('2024-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period('2024-12', freq='M').start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period('2025-01', freq='M').start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period('2025-08', freq='M').start_time + pd.offsets.MonthBegin(0)

    y_hist = y.loc[start_2024:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    # Fit ETS on data through Dec 2024, forecast Jan-Aug 2025
    y_train = y.loc[:end_2024]
    ets = ExponentialSmoothing(y_train, trend='add', seasonal='add', seasonal_periods=cfg.season).fit(optimized=True)
    fcast = ets.forecast(len(pd.period_range('2025-01', '2025-08', freq='M')))
    # Simple uncertainty band using residual std
    resid = y_train - ets.fittedvalues.reindex(y_train.index)
    sigma = float(resid.std(ddof=1)) if resid.std(ddof=1) else 0.0
    upper = fcast + 1.96 * sigma
    lower = fcast - 1.96 * sigma

    fig, ax = plt.subplots(figsize=(10,5))
    # History 2024
    ax.plot(y_hist.index, y_hist.values, color='#555555', lw=1.5)
    # Vertical line at Jan 2025
    ax.axvline(jan_2025, color='#777777', linestyle='--', lw=1)
    # Actuals 2025 Jan-Aug
    ax.plot(y_act.index, y_act.values, color='#1f77b4', lw=1.8)
    # Forecast in red with band
    ax.fill_between(fcast.index, lower.values, upper.values, color='red', alpha=0.08, linewidth=0)
    ax.plot(fcast.index, fcast.values, color='red', lw=2.0)

    # Minimal y-axis
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Direct end labels
    if len(y_hist):
        ax.annotate('History (2024)', xy=(y_hist.index[-1], y_hist.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#555555')
    if len(y_act):
        ax.annotate('Actual (Jan-Aug 2025)', xy=(y_act.index[-1], y_act.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='#1f77b4')
    ax.annotate('Forecast', xy=(fcast.index[-1], fcast.values[-1]), xytext=(6,0), textcoords='offset points', fontsize=9, va='center', ha='left', color='red')

    ax.set_title('EIA Net Generation — ETS forecast from Jan-Aug 2025 (history: 2024)')
    ax.set_xlabel('')
    save_fig('eia_expsmooth_last_fold.png')

if __name__ == '__main__':
    main()
