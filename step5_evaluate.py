"""
STEP 5 - Evaluate on the TEST years (2008-2009), which nothing has touched so far.

  5.1 Metrics: MAE, RMSE, R² - what each one means
  5.2 Compare with baselines: is the LSTM better than simple rules and Week 5?
  5.3 Look at the predictions: actual vs predicted
  5.4 Error analysis: how are the errors spread, and in which months is it worse?
  5.5 Look inside: which of the 14 days does the LSTM actually use?
Run:  python step5_evaluate.py      (needs step 4 first: it loads models/lstm.npz)
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from common import (banner, load_data, plot_bars, plot_hist, plot_lines, print_scores, scores,
                    MODEL_PATH, WEEK2_GB_TEST_MAE, WEEK5_FFNN_TEST_MAE, WINDOW)
from lstm import LSTM

# ----- Load the trained LSTM and the test data -----
net, saved = LSTM.load(MODEL_PATH)
splits, scaler = load_data()          # same pipeline + same train-only scaler as in training
(X_tr, y_tr, _), (X_te, y_te, d_te) = splits["train"], splits["test"]
y_test = scaler.to_celsius(y_te)                          # real answers in °C
pred = scaler.to_celsius(net.predict(X_te))               # LSTM output -> °C
assert np.allclose(saved["mean"], scaler.mean), "scaler differs from training: rerun step 4"

# ----- 5.1 Metrics -----
banner(f"5.1 Metrics on {len(y_test)} unseen test days (2008-2009)")
s = scores(y_test, pred)
print(f"MAE  = mean(|y - ŷ|)             = {s['MAE']:.3f} °C   the typical miss")
print(f"RMSE = sqrt(mean((y - ŷ)²))      = {s['RMSE']:.3f} °C   punishes big misses more")
print(f"R²   = 1 - SSE / SST             = {s['R2']:.3f}      share of the day-to-day variation explained")
print("RMSE > MAE tells us a few days have large misses (sudden weather changes).")

# ----- 5.2 Baselines -----
banner("5.2 Baselines: a model is only useful if it beats simple rules")
temps = scaler.to_celsius(X_te[:, :, 0]).reshape(X_te.shape[:2])   # the 14 input temps in °C
# Linear regression on the SAME 14-day window, flattened into one row of 14 × 3 = 42 numbers.
flat = lambda X: X.reshape(len(X), -1)
lin = LinearRegression().fit(flat(X_tr), y_tr.ravel())
rows = {
    "Tomorrow = today (persistence)": scores(y_test, temps[:, -1]),
    f"Average of the last {WINDOW} days": scores(y_test, temps.mean(axis=1)),
    "Linear regression (same 14 days, flat)": scores(y_test, scaler.to_celsius(lin.predict(flat(X_te)))),
    "LSTM (16 memory cells)": s,
}
print_scores(rows)
print(f"\nWeek 2 gradient boosting on the same test days:  MAE {WEEK2_GB_TEST_MAE:.2f} °C")
print(f"Week 5 feedforward network on the same test days: MAE {WEEK5_FFNN_TEST_MAE:.3f} °C")
base = rows["Tomorrow = today (persistence)"]["MAE"]
lin_mae = rows["Linear regression (same 14 days, flat)"]["MAE"]
print(f"\nLSTM vs persistence:       {100 * (base - s['MAE']) / base:+.1f}% smaller MAE  -> it learned something real")
print(f"LSTM vs linear regression: {s['MAE'] - lin_mae:+.3f} °C   (same 14 days as input)")
print(f"LSTM vs Week 5 network:    {s['MAE'] - WEEK5_FFNN_TEST_MAE:+.3f} °C   (7 days, flat row)")
print("Honest reading: the LSTM is ahead, but only by a few hundredths of a degree (~2%). Step 6")
print("shows that just changing the random start of training moves the score by ~0.03 °C, so this")
print("is a SMALL edge, not a breakthrough. Tomorrow's minimum is close to a straight-line mix of")
print("the last few days; the LSTM's strength (long, complex sequences) has little extra to find.")

# ----- 5.3 Predictions -----
banner("5.3 Actual vs predicted: the last 120 test days (Sep-Dec 2009)")
last = slice(-120, None)
plot_lines({"actual": y_test[last], "predicted": pred[last]},
           "Actual vs predicted - Sep-Dec 2009", "date", "°C", dates=d_te[last])
print("The prediction follows the level and the autumn cooling, but it is smoother than reality:")
print("it reacts to a sudden cold or warm day one day LATE, because it only knows the past.")

# ----- 5.4 Errors -----
banner("5.4 Error analysis")
err = y_test - pred                                        # positive = real day was warmer
plot_hist(err, "Histogram of errors (actual - predicted, °C)", bins=40, xlabel="°C")
print(f"Mean error {err.mean():+.2f} °C (close to 0 = no systematic bias)")
print(f"{np.mean(np.abs(err) < 1):.0%} of days within ±1 °C, {np.mean(np.abs(err) < 3):.0%} within ±3 °C, "
      f"worst miss {np.abs(err).max():.1f} °C on {d_te[np.abs(err).argmax()].date()}")
monthly = pd.Series(np.abs(err), index=d_te).groupby(d_te.month).mean()
months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
plot_bars(months, monthly.round(2), "MAE by month (°C)")
print("Values:", "  ".join(f"{m}: {v:.2f}" for m, v in zip(months, monthly)))
worst, best = monthly.idxmax(), monthly.idxmin()
print(f"Hardest month: {months[worst - 1]} ({monthly.max():.2f} °C). Easiest: {months[best - 1]} ({monthly.min():.2f} °C).")
print("Winter is hardest: cold snaps and mild spells can swing the minimum by ~10 °C overnight.")

# ----- 5.5 Inside the LSTM -----
banner("5.5 Which of the 14 days does the LSTM listen to?")
print("Sensitivity = |dŷ/dx|: how much the prediction moves if one input day's temperature moves.")
print("We get it for free from backpropagation: run backward() with dL/dŷ = 1 and read dX.\n")
net.forward(X_te)
_, dX = net.backward(np.ones((len(X_te), 1)))              # dŷ/dX for every test sequence
influence = np.abs(dX[:, :, 0]).mean(axis=0)[::-1]         # temp_min column; [::-1] = yesterday first
labels = [f"{k} day{'s' if k > 1 else ''} ago" for k in range(1, WINDOW + 1)]
plot_bars(labels, influence.round(3), "Influence of each past day on the prediction")
share = influence / influence.sum()
print(f"Yesterday alone: {share[0]:.0%} of the total influence; the last 3 days: {share[:3].sum():.0%}; "
      f"days 8-14: {share[7:].sum():.0%}.")
print("The LSTM learned by itself what step 1's autocorrelation showed: the most recent days")
print("matter most. Its forget gates let older days fade instead of keeping them all equally.")
