import signalplot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
np.random.seed(42)
signalplot.apply(font_family='serif')




@dataclass
class Config:
    csv_path: str = "2001-2025 Net_generation_United_States_all_sectors_monthly.csv"
    freq: str = "MS"
    horizon: int = 12
    n_splits: int = 5
    season: int = 12

def load_config(config_path=None) -> 'Config':
    """Build Config from config.yaml, falling back to dataclass defaults."""
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    if not config_path.exists():
        return Config()
    with open(config_path) as _f:
        import yaml as _yaml
        raw = _yaml.safe_load(_f) or {}
    _d = raw.get('data', {})
    _m = raw.get('model', {})
    _o = raw.get('output', {})
    return Config(
        csv_path=_d.get('input_file', '2001-2025 Net_generation_United_States_all_sectors_monthly.csv'),
        freq=_d.get('freq', 'MS'),
        horizon=_m.get('horizon', 12),
        n_splits=_d.get('n_splits', 5),
        season=_m.get('season', 12),
    )



def load_series(cfg: Config) -> pd.Series:
    p = Path(cfg.csv_path)
    df = pd.read_csv(p, header=None, usecols=[0, 1], names=["date", "value"], sep=",")
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
        ets = ExponentialSmoothing(
            y_tr, trend="add", seasonal="add", seasonal_periods=cfg.season
        ).fit(optimized=True)
        f = ets.forecast(len(y_te)).to_numpy()
        m = mean_absolute_error(y_te.values, f)
        logger.info(f"Fold MAE: {m:.3f}")
        maes.append(m)
        last_true, last_pred = y_te, pd.Series(f, index=y_te.index)
    return float(np.mean(maes)), last_true, last_pred


def main(plot: bool = False):
    cfg = load_config()
    y = load_series(cfg)
    mean_mae, _, _ = rolling_origin_ets(y, cfg)
    logger.info(f"ETS mean MAE: {mean_mae}")

    # Tufte-style figure focused on 2024 history and Jan–Aug 2025 forecast vs actuals
    start_2024 = pd.Period("2024-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    end_2024 = pd.Period("2024-12", freq="M").start_time + pd.offsets.MonthBegin(0)
    jan_2025 = pd.Period("2025-01", freq="M").start_time + pd.offsets.MonthBegin(0)
    aug_2025 = pd.Period("2025-08", freq="M").start_time + pd.offsets.MonthBegin(0)

    y_hist = y.loc[start_2024:end_2024]
    y_act = y.loc[jan_2025:aug_2025]

    # Fit ETS on data through Dec 2024, forecast Jan–Aug 2025
    y_train = y.loc[:end_2024]
    ets = ExponentialSmoothing(
        y_train, trend="add", seasonal="add", seasonal_periods=cfg.season
    ).fit(optimized=True)
    fcast = ets.forecast(len(pd.period_range("2025-01", "2025-08", freq="M")))
    # Simple uncertainty band using residual std
    resid = y_train - ets.fittedvalues.reindex(y_train.index)
    sigma = float(resid.std(ddof=1)) if resid.std(ddof=1) else 0.0
    upper = fcast + 1.96 * sigma
    lower = fcast - 1.96 * sigma

    if plot:
        fig, ax = plt.subplots(figsize=(10, 5))
    # History 2024
        ax.plot(y_hist.index, y_hist.values, color="#555555", lw=1.5)
    # Vertical line at Jan 2025
        ax.axvline(jan_2025, color="#777777", linestyle="--", lw=1)
    # Actuals 2025 Jan–Aug
        ax.plot(y_act.index, y_act.values, color="#1f77b4", lw=1.8)
    # Forecast in red with band
        ax.fill_between(
            fcast.index, lower.values, upper.values, color="red", alpha=0.08, linewidth=0
        )
        ax.plot(fcast.index, fcast.values, color="red", lw=2.0)

    # Minimal y-axis
        from matplotlib.ticker import MaxNLocator, StrMethodFormatter

        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Direct end labels
        if len(y_hist):
            ax.annotate(
                "History (2024)",
                xy=(y_hist.index[-1], y_hist.values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                ha="left",
                color="#555555",
            )
        if len(y_act):
            ax.annotate(
                "Actual (Jan–Aug 2025)",
                xy=(y_act.index[-1], y_act.values[-1]),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=9,
                va="center",
                ha="left",
                color="#1f77b4",
            )
        ax.annotate(
            "Forecast",
            xy=(fcast.index[-1], fcast.values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            fontsize=9,
            va="center",
            ha="left",
            color="red",
        )

        ax.set_title("EIA Net Generation — ETS forecast from Jan–Aug 2025 (history: 2024)")
        ax.set_xlabel("")
        ax.grid(False)
        signalplot.save("eia_expsmooth_last_fold.png")


if __name__ == "__main__":
    main()
