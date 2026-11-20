"""
Shared helpers used by every step script: settings, data pipeline, scaling, metrics, charts.
The LSTM itself lives in lstm.py.

Dataset: daily weather in Stockholm, Sweden, 2000-2009 (one row per day).
  - Source: "Weather prediction dataset" (Huber et al.), built from ECA&D station data
    https://github.com/florian-huber/weather_prediction_dataset
  - Licence: CC BY 4.0 (dataset on Zenodo), MIT (GitHub repo). Free to use if you credit the source.
Task: predict TOMORROW's minimum temperature (°C) from a SEQUENCE of the days before it.
Same data, same split and same test years as Week 5, so the two models compare directly.
"""
import re                      # turns a chart title into a safe file name
import sys                     # lets us fix how text prints on Windows
from pathlib import Path       # file paths that work on Windows and Mac/Linux

import matplotlib
matplotlib.use("Agg")          # "Agg" = only write PNG files, never open a chart window
from matplotlib import pyplot as mpl   # PNG charts (called mpl because plt is the terminal one)
import numpy as np             # fast maths on whole arrays of numbers
import pandas as pd            # tables: DataFrame = whole table, Series = one labelled column
import plotext as plt          # charts drawn as text, right inside the terminal

sys.stdout.reconfigure(encoding="utf-8")  # so °C, σ and the terminal charts print on Windows too

# ----------------------------------------------------------------------------- settings
ROOT = Path(__file__).parent                              # this project's folder
DATA_PATH = ROOT / "data" / "weather_prediction_dataset.csv"
MODEL_PATH = ROOT / "models" / "lstm.npz"                 # step 4 saves the trained LSTM here
RESULTS_DIR = ROOT / "results"                            # every chart is also saved here as PNG

CITY = "STOCKHOLM"             # the dataset has 18 European cities; we use one
LAST_DAY = "2009-12-31"        # the file ends on 2010-01-01; stop there to keep 10 full years

# Time-series split: cut by the DATE OF THE ANSWER, never randomly (the future must stay hidden).
#   2000-2005 -> train       (the LSTM learns its weights from this)
#   2006-2007 -> validation  (watched every epoch: early stopping + choosing settings)
#   2008-2009 -> test        (used at the very end, like real "future" days)
TRAIN_END = "2005-12-31"
VAL_END = "2007-12-31"

# What the LSTM reads: a SEQUENCE of WINDOW days, each day described by FEATURES.
WINDOW = 14                                   # two weeks of history, read one day at a time
FEATURES = ["temp_min", "doy_sin", "doy_cos"]  # per day: min temperature + position in the year
TARGET = "temp_min"                           # what we predict (for the day AFTER the window)
WEATHER = ["temp_mean", "temp_max", "pressure", "cloud_cover", "sunshine", "precipitation"]

SEED = 42                      # fixes the random numbers, so every run gives the same result

# LSTM settings ("hyperparameters" = settings WE choose; the LSTM learns the weights)
HIDDEN = 16                    # size of the memory: h and c are vectors of 16 numbers
LEARNING_RATE = 0.005          # Adam's step size
BATCH_SIZE = 32                # sequences per weight update
EPOCHS = 80                    # maximum passes over the training data
PATIENCE = 10                  # early stopping: give up after 10 epochs without val improvement
CLIP = 1.0                     # gradient clipping: never let one update be longer than this

# Earlier weeks, same test days (2008-2009), for comparison
WEEK2_GB_TEST_MAE = 1.63       # gradient boosting
WEEK5_FFNN_TEST_MAE = 1.620    # feedforward network, temperatures + season
WEEK5_BEST_TEST_MAE = 1.312    # ensemble of 5 feedforward networks + yesterday's weather


# ----------------------------------------------------------------------------- data
def load_raw():
    """Read the CSV and keep only Stockholm's columns, with short names."""
    full = pd.read_csv(DATA_PATH, dtype={"DATE": str})
    city_cols = [c for c in full.columns if c.startswith(CITY + "_")]
    # "STOCKHOLM_temp_min" -> "temp_min" (shorter and easier to read)
    data = full[city_cols].rename(columns=lambda c: c.removeprefix(CITY + "_"))
    data.insert(0, "Date", pd.to_datetime(full["DATE"], format="%Y%m%d"))   # "20000101" -> a real date
    return data[data["Date"] <= LAST_DAY].reset_index(drop=True)


# Physically possible ranges. The file has no empty cells, but it hides missing readings as
# impossible numbers: cloud cover -99, pressure -0.099 (= -99 hPa), sunshine -1.7 hours, and
# cloud cover 9 ("sky not visible"). These "sentinel values" must become real gaps (NaN).
VALID_RANGES = {"cloud_cover": (0, 8), "pressure": (0.9, 1.1), "sunshine": (0, 24),
                "precipitation": (0, 50), "temp_min": (-50, 50), "temp_mean": (-50, 50),
                "temp_max": (-50, 50)}


def find_invalid(df):
    """True where a value lies outside its physically possible range."""
    return pd.DataFrame({c: ~df[c].between(*VALID_RANGES[c]) for c in df.columns if c in VALID_RANGES})


def clean(raw):
    """Raw table -> one row per calendar day, no gaps, no impossible values (indexed by date)."""
    df = raw.set_index("Date").astype(float)
    df = df.mask(find_invalid(df))   # impossible values -> NaN (a real "missing" marker)
    df = df.asfreq("D")              # a row for every calendar day; missing days become NaN
    return df.interpolate()          # fill each gap with a straight line between its neighbours


def add_season(df):
    """Add the day of the year as a point on a circle, so Dec 31 sits next to Jan 1."""
    angle = 2 * np.pi * df.index.day_of_year / 365.25
    return df.assign(doy_sin=np.sin(angle), doy_cos=np.cos(angle))


class Standardizer:
    """Rescale each column to mean 0 and standard deviation 1:   x_scaled = (x - mean) / std

    Why? The LSTM multiplies inputs by weights and squashes them with sigmoid/tanh. Inputs of
    very different sizes (pressure ~1.0, temperature -20..+30) make some gates saturate and
    learning slow. IMPORTANT: mean and std come from the TRAINING period only.
    """

    def fit(self, table):
        self.mean, self.std = table.mean(), table.std(ddof=0)   # one number per column
        return self

    def transform(self, table):
        return (table - self.mean) / self.std

    def to_celsius(self, y_scaled):
        """Undo the scaling of the target: scaled prediction -> °C."""
        return np.ravel(y_scaled) * self.std[TARGET] + self.mean[TARGET]


def make_sequences(table, window=WINDOW):
    """Sliding window: daily table -> 3-D input array X, answers y, and the answer dates.

    For every day t (from day `window` on):
      X[t] = the table rows of days t-window ... t-1   shape (window, features)  <- the sequence
      y[t] = temp_min on day t                          (the day AFTER the sequence)
    Result: X (samples, window, features), y (samples, 1). The inputs only use days BEFORE the
    answer, so the LSTM can never peek at it ("data leakage").
    """
    values = table.to_numpy()
    target = table[TARGET].to_numpy()
    X = np.stack([values[t - window:t] for t in range(window, len(table))])
    y = target[window:, None]                      # [:, None] = make it a column, shape (n, 1)
    return X, y, table.index[window:]


def load_data(window=WINDOW, features=FEATURES):
    """The whole pipeline in one call: raw -> clean -> season -> scale -> sequences -> split.

    Returns ({"train": (X, y, dates), "val": ..., "test": ...}, scaler).
    """
    df = add_season(clean(load_raw()))[features]
    scaler = Standardizer().fit(df.loc[:TRAIN_END])          # statistics of the TRAIN period only
    X, y, dates = make_sequences(scaler.transform(df), window)
    parts = {"train": dates <= TRAIN_END,
             "val": (dates > TRAIN_END) & (dates <= VAL_END),
             "test": dates > VAL_END}
    return {name: (X[m], y[m], dates[m]) for name, m in parts.items()}, scaler


# ----------------------------------------------------------------------------- metrics
def scores(y_true, y_pred):
    """How good are the predictions? All in °C except R².
    MAE  = average size of the miss                       mean(|y - ŷ|)
    RMSE = like MAE, but big misses count extra           sqrt(mean((y - ŷ)²))
    R²   = share of the ups and downs explained           1 - Σ(y - ŷ)² / Σ(y - mean(y))²
    """
    y_true, y_pred = np.ravel(y_true), np.ravel(y_pred)
    err = y_true - y_pred
    return {"MAE": np.mean(np.abs(err)),
            "RMSE": np.sqrt(np.mean(err ** 2)),
            "R2": 1 - np.sum(err ** 2) / np.sum((y_true - y_true.mean()) ** 2)}


def print_scores(rows):
    """rows = {"name": scores_dict}. Prints a small aligned table."""
    print(f"{'':<40}{'MAE °C':>9}{'RMSE °C':>10}{'R²':>8}")
    for name, s in rows.items():
        print(f"{name:<40}{s['MAE']:>9.3f}{s['RMSE']:>10.3f}{s['R2']:>8.3f}")


# ----------------------------------------------------------------------------- terminal output
def banner(text):
    """A section heading, so each part of a step's output is easy to find."""
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]    # fixed order, distinct for colour-blind readers
SURFACE = "#fcfcfb"                                   # PNG background colour
mpl.rcParams.update({                                 # shared PNG style: quiet axes, data stands out
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.dpi": 150,
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"], "text.color": "#0b0b0b",
    "axes.titlelocation": "left", "axes.titlesize": 13, "axes.titlepad": 12,
    "axes.labelcolor": "#52514e", "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "axes.axisbelow": True,
    "grid.color": "#e1e0d9", "grid.linewidth": 0.6, "legend.frameon": False,
    "xtick.color": "#c3c2b7", "xtick.labelcolor": "#52514e",
    "ytick.color": "#c3c2b7", "ytick.labelcolor": "#52514e",
    "lines.linewidth": 1.0, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})


def save_png(fig, title):
    """Save a chart to results/, named after the running step and the title."""
    name = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")    # "MAE by month (°C)" -> "mae-by-month-c"
    path = RESULTS_DIR / f"{Path(sys.argv[0]).stem}_{name}.png"
    RESULTS_DIR.mkdir(exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    mpl.close(fig)
    print(f"Saved -> results/{path.name}")


def plot_lines(lines, title, xlabel="", ylabel="", dates=None, start=1):
    """Line chart(s) in the terminal + PNG. lines = {"legend label": list_of_y_values}.
    With `dates`, the x-axis shows years (or months) instead of row numbers."""
    plt.clear_figure()                 # plotext remembers the last chart; wipe it
    plt.theme("clear")                 # use the terminal's own background (dark or light)
    plt.plotsize(100, 22)              # 100 characters wide, 22 rows tall
    x = list(range(start, start + len(next(iter(lines.values())))))
    for label, y in lines.items():
        plt.plot(x, list(y), label=label, marker="braille")    # braille dots = smoother line
    ticks = None
    if dates is not None:
        ticks = [i for i, d in enumerate(dates) if d.day_of_year == 1]    # every January 1st
        if not ticks:                                                     # short range: month starts
            ticks = [i for i, d in enumerate(dates) if d.day == 1]
        fmt = "%Y" if dates[ticks[0]].day_of_year == 1 else "%b %Y"
        plt.xticks([x[i] for i in ticks], [dates[i].strftime(fmt) for i in ticks])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

    # ----- the same chart as a PNG -----
    fig, ax = mpl.subplots(figsize=(10, 4.4))
    for (label, y), color in zip(lines.items(), SERIES_COLORS):
        ax.plot(x, list(y), label=label, color=color)
    if ticks is not None:
        ax.set_xticks([x[i] for i in ticks], [dates[i].strftime(fmt) for i in ticks])
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    ax.grid(axis="y")
    if len(lines) > 1:
        ax.legend()
    save_png(fig, title)


def plot_bars(labels, values, title):
    """Horizontal bar chart in the terminal + PNG."""
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(100, len(labels) + 4)       # +4 rows for title and axis -> one row per bar
    # plotext draws the first bar at the bottom, so reverse to read top-down
    plt.bar([str(l) for l in labels][::-1], list(values)[::-1], orientation="horizontal", width=0.5)
    plt.title(title)
    plt.show()

    fig, ax = mpl.subplots(figsize=(10, 0.32 * len(labels) + 1))
    bars = ax.barh([str(l) for l in labels], list(values), height=0.5, color=SERIES_COLORS[0])
    ax.invert_yaxis()                                     # first label on top
    ax.bar_label(bars, fmt="%g", padding=4, color="#52514e")
    ax.axvline(0, color="#c3c2b7", linewidth=0.6)
    ax.margins(x=0.08)
    ax.xaxis.set_visible(False)
    ax.spines[["left", "bottom"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title)
    save_png(fig, title)


def plot_hist(values, title, bins=30, xlabel=""):
    """Histogram (how many values fall in each range) in the terminal + PNG."""
    plt.clear_figure()
    plt.theme("clear")
    plt.plotsize(100, 20)
    plt.hist(list(values), bins)
    plt.title(title)
    plt.show()

    fig, ax = mpl.subplots(figsize=(10, 4.4))
    ax.hist(list(values), bins=bins, color=SERIES_COLORS[0], edgecolor=SURFACE, linewidth=1)
    ax.set(title=title, xlabel=xlabel, ylabel="count")
    ax.grid(axis="y")
    save_png(fig, title)
