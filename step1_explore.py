"""
STEP 1 - Data analysis: look at the data before touching any model.

Questions we answer here:
  1. What does the dataset look like?           4. How far back does the past matter?
  2. Are any days missing or broken?               (autocorrelation -> how long a sequence?)
  3. Is there a trend or a yearly cycle?         5. Do other measurements help?
                                                 6. What range do the numbers have? (scaling)
Run:  python step1_explore.py
"""
import pandas as pd

from common import (banner, clean, find_invalid, load_raw, plot_bars, plot_hist, plot_lines,
                    CITY, DATA_PATH, VALID_RANGES)

# ----- 1. The dataset -----
banner("1. The dataset: one row = one day, columns = weather measurements")
header = pd.read_csv(DATA_PATH, nrows=0).columns                  # read only the column names
cities = sorted({c.split("_")[0] for c in header if c not in ("DATE", "MONTH")})
print(f"Full file: {len(header)} columns, {len(cities)} cities: {', '.join(cities)}")
raw = load_raw()                                                  # keep only Stockholm's columns
print(f"We keep {CITY}: {len(raw)} rows (days) x {raw.shape[1]} columns\n")
print(raw.head(), "\n")
print(f"From {raw['Date'].min().date()} to {raw['Date'].max().date()}")
print("Units: temperatures °C, pressure in 1000 hPa (1.02 = 1020 hPa), precipitation cm,")
print("       sunshine hours, cloud cover in eighths of the sky (0 = clear, 8 = overcast)\n")
print("Our TARGET (what we predict) = temp_min, the day's minimum temperature:")
print(raw["temp_min"].describe().round(2))                       # count, mean, std, min, quartiles, max

# ----- 2. Data quality -----
banner("2. Data quality: missing days, empty values, impossible values?")
df = clean(raw)                                                   # one row per day, gaps filled
missing_days = df.index.difference(raw["Date"])
print(f"Calendar days: {len(df)}   Rows in file: {len(raw)}   Missing days: {len(missing_days)}")
print(f"Empty values (NaN) in the file: {int(raw.isna().sum().sum())}")
bad = find_invalid(raw.set_index("Date"))                         # values outside a possible range
print("Impossible values hiding as numbers (sentinel codes for 'no reading'):")
for col in bad.columns[bad.any()]:
    codes = ", ".join(f"{v:g}" for v in sorted(raw.set_index("Date")[col][bad[col]].unique()))
    print(f"  {col:<12} {int(bad[col].sum()):>3} values: {codes:<10} allowed range {VALID_RANGES[col]}")
print("-> clean() turns them into NaN, then fills each gap with a straight line between its")
print("   neighbours (linear interpolation). temp_min, our target, has none: it is clean.")
print("Why this matters MORE for an LSTM: there must be exactly one value per day, every day,")
print("because the LSTM reads the days in order. A missing day would silently shift the sequence.")
print(f"Empty or impossible values left after cleaning: {int(df.isna().sum().sum() + find_invalid(df).sum().sum())}")

# ----- 3. Trend and seasons -----
s = df["temp_min"]
banner("3. The whole series: a strong yearly cycle (warm summers, cold winters), no big trend")
plot_lines({"min temp °C": s}, "Stockholm daily minimum temperature", "year", "°C", dates=s.index)
monthly = s.groupby(s.index.month).mean()                         # average of all Januaries, ...
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plot_bars(months, monthly.round(1), "Average minimum temperature by month (°C)")
print("(Bars start at 0: below-zero months grow to the LEFT.)")
yearly = s.groupby(s.index.year).mean()
print("Yearly averages (looking for a trend):", "  ".join(f"{y}: {v:.1f}" for y, v in yearly.items()))
print("-> They wobble between 4.7 and 5.8 °C with no steady rise or fall: no clear trend.")

# ----- 4. Autocorrelation: how long should the sequence be? -----
banner("4. Autocorrelation: how similar is today to N days ago? (+1 identical, 0 unrelated)")
lags = list(range(1, 61))
corr = [s.autocorr(lag) for lag in lags]                          # correlation of s with s shifted
for lag in [1, 2, 3, 7, 14, 30, 60]:
    print(f"  today vs {lag:>2} days ago: {corr[lag - 1]:+.2f}")
plot_lines({"correlation": corr}, "Correlation of today with N days ago (1-60)", "days ago", "corr")
print(f"Yesterday is the strongest clue ({corr[0]:+.2f}); the clue fades slowly with distance.")
print("Part of what stays high for weeks is just the SEASON (a cold week sits inside a cold month),")
print("which we give the LSTM directly as doy_sin/doy_cos. So 14 days is our starting window;")
print("step 6 tests 7 and 30 days. An LSTM reads any length with the SAME number of weights.")
deseason = s - s.groupby(s.index.day_of_year).transform("mean")   # remove the average yearly curve
print("\nWithout the season (each day minus the average for that calendar day):")
print("  " + "  ".join(f"{lag}d: {deseason.autocorr(lag):+.2f}" for lag in [1, 2, 3, 7, 14, 30]))
print("-> The real day-to-day 'memory' of the weather lasts about 1-2 weeks.")

# ----- 5. Other measurements -----
banner("5. Do YESTERDAY's other measurements relate to TODAY's minimum temperature?")
others = [c for c in df.columns if c != "temp_min"]
corr_other = {c: df[c].shift(1).corr(s) for c in others}          # shift(1) = yesterday's value
for c, v in sorted(corr_other.items(), key=lambda kv: -abs(kv[1])):
    print(f"  yesterday's {c:<14} vs today's temp_min: {v:+.2f}")
print("temp_mean/temp_max mostly repeat what the temperature sequence already says; pressure,")
print("clouds and sunshine carry weather information it doesn't. Step 6 adds them to every day.")

# ----- 6. Ranges -----
banner("6. Value ranges: very different sizes -> we must scale inputs for a neural network")
print(df.describe().loc[["mean", "std", "min", "max"]].round(3).T)
plot_hist(s, "Distribution of daily minimum temperature (°C)", xlabel="°C")
print(f"temp_min goes from {s.min():.1f} to {s.max():.1f} °C; pressure stays near 1.0 (std 0.01);")
print("sunshine 0-18 hours. The LSTM squashes values with sigmoid and tanh, which go flat for big")
print("inputs, so step 2 rescales every column to mean 0, std 1.")
