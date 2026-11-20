"""
STEP 2 - Data processing & transformation: from a calendar of temperatures to the 3-D
sequences an LSTM reads.

  2.1 Clean           -> one row per day, no gaps, no impossible values
  2.2 Season          -> the day of the year as a point on a circle (sin, cos)
  2.3 Standardize     -> every column to mean 0, std 1 (statistics of the TRAIN years only)
  2.4 Sliding window  -> each sample = 14 days IN ORDER (inputs) -> the next day (answer)
  2.5 Split by time   -> train 2000-05 | validation 2006-07 | test 2008-09
  2.6 Arrays          -> X (samples, timesteps, features) and y (samples, 1)
Run:  python step2_sequences.py
"""
import numpy as np
import pandas as pd

from common import (add_season, banner, clean, load_data, load_raw, make_sequences, Standardizer,
                    FEATURES, TRAIN_END, WINDOW)

pd.set_option("display.width", 140)

# ----- 2.1 Clean -----
banner("2.1 Clean: one value per calendar day")
df = clean(load_raw())
print(f"{len(df)} days, {int(df.isna().sum().sum())} gaps left. One row per day, no day skipped:")
print("the LSTM treats 'next row' as 'next day', so this must be exactly true.")

# ----- 2.2 Season -----
banner("2.2 Season on a circle: why sin and cos instead of 'day 1..365'?")
df = add_season(df)[FEATURES]                         # keep the 3 columns the LSTM reads per day
print("As a plain number, Dec 31 (365) and Jan 1 (1) look 364 apart, but they are neighbours.")
print("On a circle (like a clock hand) they sit side by side:  angle = 2π · day / 365.25\n")
for date in ["2001-01-01", "2001-04-01", "2001-07-01", "2001-10-01", "2001-12-31"]:
    row = df.loc[date]
    print(f"  {date}   doy_sin {row['doy_sin']:+.2f}   doy_cos {row['doy_cos']:+.2f}")
print(f"\nEvery day is now described by {len(FEATURES)} numbers (its 'feature vector'): {FEATURES}")

# ----- 2.3 Standardize -----
banner("2.3 Standardize: x_scaled = (x - mean) / std, with mean and std from TRAIN years only")
scaler = Standardizer().fit(df.loc[:TRAIN_END])       # 2000-2005 only; the future stays unseen
print(pd.DataFrame({"train mean": scaler.mean, "train std": scaler.std}).round(3), "\n")
scaled = scaler.transform(df)
x = df["temp_min"].iloc[0]
print(f"Example: 2000-01-01 temp_min = {x} °C  ->  ({x} - {scaler.mean['temp_min']:.3f}) / "
      f"{scaler.std['temp_min']:.3f} = {scaled['temp_min'].iloc[0]:+.3f}")
print(f"0 = an average day, {scaled['temp_min'].iloc[0]:+.2f} = about two typical steps colder than average.")
print("Why scale? Sigmoid and tanh are flat for big inputs (sigmoid(10) ≈ sigmoid(20) ≈ 1), so")
print("gates fed with raw numbers would get stuck and learn nothing. Scaled inputs stay in the")
print("sensitive middle part of those curves.")
print("The target (tomorrow's temp_min) uses the SAME temp_min mean/std, so predictions convert")
print(f"back with  °C = ŷ × {scaler.std['temp_min']:.3f} + {scaler.mean['temp_min']:.3f}")

# ----- 2.4 Sliding window -----
banner(f"2.4 Sliding window: {WINDOW} days in order -> the next day")
print("Week 5's feedforward network needed the past FLATTENED into one row (lag_1 ... lag_7).")
print("An LSTM instead receives the days as a SEQUENCE and reads them one by one, keeping a memory.")
print("Each sample is a small table: rows = days (time steps), columns = features.\n")
X, y, dates = make_sequences(scaled)
print(f"Sample 0 = the {WINDOW} days before {dates[0].date()} (scaled values):")
sample = pd.DataFrame(X[0], columns=FEATURES, index=scaled.index[:WINDOW].date)
print(sample.round(2))
print(f"  answer y[0] = temp_min on {dates[0].date()} = {y[0, 0]:+.3f} "
      f"(= {scaler.to_celsius(y[0])[0]:.1f} °C)\n")
print("The next sample is the same window moved ONE day later:")
print(f"  sample 1 = days {scaled.index[1].date()} .. {scaled.index[WINDOW].date()} -> answer {dates[1].date()}")
print(f"  sample 2 = days {scaled.index[2].date()} .. {scaled.index[WINDOW + 1].date()} -> answer {dates[2].date()}")
print(f"\nThe first {WINDOW} days can't be answers (no full window before them): "
      f"{len(scaled)} days -> {len(X)} samples.")

# ----- 2.5 Split -----
banner("2.5 Split by the DATE OF THE ANSWER, never randomly")
splits, _ = load_data()                               # the same steps 2.1-2.4, then split
for name, use in [("train", "learn the weights"),
                  ("val", "watch every epoch: early stopping, choosing settings"),
                  ("test", "final honest score, used at the very end")]:
    Xp, yp, dp = splits[name]
    share = len(Xp) / len(X)
    print(f"  {name:<6} {dp[0].date()} -> {dp[-1].date()}  {len(Xp):>5} samples ({share:4.0%})  "
          f"{'█' * round(share * 40):<26} {use}")
first_val = splits["val"][2][0]
print(f"\nThe first validation sample ({first_val.date()}) reads 14 days of December 2005, which")
print("are training days. That's fine: on 1 Jan 2006 you really would know the last two weeks.")
print("What must never happen is the ANSWER of a validation/test day being used for training.")
print("A random split would break this: the LSTM would train on 5 March 2008 and be 'tested' on")
print("4 March 2008, whose window it has mostly already seen. Cutting by date copies real life.")

# ----- 2.6 Arrays -----
banner("2.6 Final arrays: what the LSTM actually receives")
for name, (Xp, yp, _) in splits.items():
    print(f"  X_{name:<5} shape {Xp.shape}  = (samples, days per sample, features per day)   "
          f"y_{name:<5} shape {yp.shape}")
Xt = splits["train"][0]
print(f"\nTraining inputs after scaling: mean {Xt[:, :, 0].mean():+.3f}, std {Xt[:, :, 0].std():.3f} "
      "(temp_min column; ~0 and ~1 as intended)")
print("Picture X_train as a stack of 2178 small tables, each 14 rows (days) × 3 columns:")
print("""
         features →
       ┌──────────────┐
  day  │ t-14 ...     │┐
   ↓   │ ...          ││┐       2178 samples (stacked)
       │ t-1  ...     │││
       └──────────────┘││
        └──────────────┘│
         └──────────────┘""")
print(f"The LSTM reads one row (one day, {Xt.shape[2]} numbers) per step, {Xt.shape[1]} steps per sample.")
print(f"Number check: {np.prod(Xt.shape)} input numbers in X_train = 2178 × 14 × 3.")
