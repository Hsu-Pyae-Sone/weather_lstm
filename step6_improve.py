"""
STEP 6 - How to improve the model. Every decision is made on VALIDATION data;
the test set is used once, at the very end, for the winner only.

  6.1 Experiments: memory size, learning rate, sequence length, more weather inputs per day
  6.2 Ensemble: average 5 LSTMs trained from different random starts
  6.3 The library way: the same LSTM in PyTorch (what you'd use in a real project)
  6.4 Pick the winner on validation, then test it once
Run:  python step6_improve.py      (~2 minutes)
"""
import numpy as np
import torch
from sklearn.linear_model import LinearRegression

from common import (banner, load_data, plot_bars, print_scores, scores, BATCH_SIZE, CLIP, EPOCHS,
                    FEATURES, HIDDEN, LEARNING_RATE, PATIENCE, SEED, WEATHER, WEEK5_BEST_TEST_MAE,
                    WEEK5_FFNN_TEST_MAE)
from lstm import LSTM, train


def fit_lstm(data, hidden=HIDDEN, lr=LEARNING_RATE, seed=SEED):
    """Train one from-scratch LSTM; return its predictions in °C for val and test."""
    splits, scaler = data
    (X_tr, y_tr, _), (X_va, y_va, _) = splits["train"], splits["val"]
    net = LSTM(X_tr.shape[2], hidden, seed=seed)
    train(net, X_tr, y_tr, X_va, y_va, epochs=EPOCHS, lr=lr, batch_size=BATCH_SIZE,
          patience=PATIENCE, clip=CLIP, seed=seed)
    return {name: scaler.to_celsius(net.predict(splits[name][0])) for name in ("val", "test")}


def val_mae(data, preds):
    splits, scaler = data
    return scores(scaler.to_celsius(splits["val"][1]), preds["val"])["MAE"]


# ----- 6.1 Experiments -----
banner("6.1 Experiments: change ONE thing at a time, compare validation MAE")
data = {"base": load_data(), "window 7": load_data(window=7), "window 30": load_data(window=30),
        "weather": load_data(features=FEATURES + WEATHER)}
experiments = {   # name: (data version, memory cells, learning rate)
    "baseline: 16 cells, 14 days, lr 0.005": ("base", 16, 0.005),
    "smaller memory: 4 cells": ("base", 4, 0.005),
    "bigger memory: 64 cells": ("base", 64, 0.005),
    "slower learning rate 0.001": ("base", 16, 0.001),
    "faster learning rate 0.02": ("base", 16, 0.02),
    "shorter sequence: 7 days": ("window 7", 16, 0.005),
    "longer sequence: 30 days": ("window 30", 16, 0.005),
    "+ weather on every day (9 features)": ("weather", 16, 0.005),
}
results = {}
for name, (version, hidden, lr) in experiments.items():
    preds = fit_lstm(data[version], hidden, lr)
    results[name] = {"version": version, "preds": preds, "val": val_mae(data[version], preds)}
    print(f"  {name:<40} validation MAE {results[name]['val']:.3f} °C")
print("\nWhat we learn:")
print("  * memory size barely matters: the pattern is simple, so more cells add nothing")
print("  * learning rate changes speed more than the final result (early stopping catches it)")
print("  * sequence length barely matters: step 5.5 showed the LSTM mostly uses the last 3 days")
print("  * NEW INFORMATION helps most: each day's pressure, clouds and sunshine describe the")
print("    weather situation that temperatures alone cannot show")

# ----- 6.2 Ensemble -----
banner("6.2 Ensemble: average 5 LSTMs that started from different random weights")
members = [fit_lstm(data["weather"], seed=s) for s in range(5)]
for i, m in enumerate(members):
    print(f"  LSTM {i + 1} (seed {i})   validation MAE {val_mae(data['weather'], m):.3f} °C")
ensemble = {k: np.mean([m[k] for m in members], axis=0) for k in ("val", "test")}
results["ensemble of 5 (+ weather)"] = {"version": "weather", "preds": ensemble,
                                        "val": val_mae(data["weather"], ensemble)}
spread = [val_mae(data["weather"], m) for m in members]
print(f"  average of all 5        validation MAE {results['ensemble of 5 (+ weather)']['val']:.3f} °C")
print(f"Just changing the random start moves a single LSTM by {max(spread) - min(spread):.3f} °C: that is")
print("the 'noise level' of one training run. Averaging cancels part of each model's own mistakes.")


# ----- 6.3 PyTorch -----
banner("6.3 The library way: the same LSTM in PyTorch (+ weather)")


class TorchLSTM(torch.nn.Module):
    """nn.LSTM reads the sequence (the loop over days is inside it); Linear makes ŷ from h_T."""

    def __init__(self, n_in, hidden):
        super().__init__()
        self.lstm = torch.nn.LSTM(n_in, hidden, batch_first=True)   # input (batch, days, features)
        self.out = torch.nn.Linear(hidden, 1)

    def forward(self, X):
        h_all, _ = self.lstm(X)            # h for every day: (batch, days, hidden)
        return self.out(h_all[:, -1])      # the last day's h -> prediction


torch.manual_seed(SEED)
splits_w, scaler_w = data["weather"]
X_tr, y_tr, X_va, y_va = (torch.tensor(a, dtype=torch.float32) for a in
                          (splits_w["train"][0], splits_w["train"][1], splits_w["val"][0], splits_w["val"][1]))
model = TorchLSTM(X_tr.shape[2], HIDDEN)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = torch.nn.MSELoss()
best, best_state, wait = np.inf, None, 0
for epoch in range(1, EPOCHS + 1):
    model.train()
    for batch in torch.randperm(len(X_tr)).split(BATCH_SIZE):
        optimizer.zero_grad()                                      # forget last batch's gradients
        loss = loss_fn(model(X_tr[batch]), y_tr[batch])            # forward + loss
        loss.backward()                                            # BPTT, done for us (autograd)
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)   # gradient clipping
        optimizer.step()                                           # Adam update
    model.eval()
    with torch.no_grad():
        val_loss = loss_fn(model(X_va), y_va).item()
    if val_loss < best:                                            # early stopping, as before
        best, best_state, wait = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
    else:
        wait += 1
        if wait >= PATIENCE:
            break
model.load_state_dict(best_state)
with torch.no_grad():
    pt = {name: scaler_w.to_celsius(model(torch.tensor(splits_w[name][0], dtype=torch.float32)).numpy())
          for name in ("val", "test")}
results["PyTorch nn.LSTM (+ weather)"] = {"version": "weather", "preds": pt, "val": val_mae(data["weather"], pt)}
print(f"  validation MAE {results['PyTorch nn.LSTM (+ weather)']['val']:.3f} °C after {epoch} epochs")
print("Same model, same settings, ~25 lines. PyTorch's autograd does the backward pass for us:")
print("loss.backward() replaces our whole backward() method. Small differences come from its")
print("different random start and its float32 numbers.")

plot_bars(list(results), [round(r["val"], 3) for r in results.values()],
          "Validation MAE per experiment (°C, lower is better)")

# ----- 6.4 Winner on the test set -----
banner("6.4 The winner (chosen on validation) meets the test set - once")
winner = min(results, key=lambda n: results[n]["val"])
print(f"Lowest validation MAE: {winner}\n")
flat = lambda X: X.reshape(len(X), -1)                    # 14 days × features -> one long row


def linear_test(version):
    """Linear regression on the same (flattened) sequences, scored on the test set."""
    splits, scaler = data[version]
    lin = LinearRegression().fit(flat(splits["train"][0]), splits["train"][1].ravel())
    return scores(scaler.to_celsius(splits["test"][1]), scaler.to_celsius(lin.predict(flat(splits["test"][0]))))


splits_b, scaler_b = data["base"]
y_test = scaler_b.to_celsius(splits_b["test"][1])         # same test days in every data version
rows = {
    "Persistence (tomorrow = today)": scores(y_test, scaler_b.to_celsius(splits_b["test"][0][:, -1, 0])),
    "Linear regression (step 5 inputs)": linear_test("base"),
    "LSTM baseline (step 5 settings)": scores(y_test, results["baseline: 16 cells, 14 days, lr 0.005"]["preds"]["test"]),
    "Linear regression (+ weather)": linear_test("weather"),
    "WINNER (see name above)": scores(y_test, results[winner]["preds"]["test"]),
}
print_scores(rows)
print(f"\nWeek 5 feedforward network: {WEEK5_FFNN_TEST_MAE:.3f} °C (temperatures)   "
      f"{WEEK5_BEST_TEST_MAE:.3f} °C (ensemble + weather)")
start, end, lin = rows["LSTM baseline (step 5 settings)"], rows["WINNER (see name above)"], rows["Linear regression (+ weather)"]
print(f"\nLSTM MAE {start['MAE']:.2f} -> {end['MAE']:.2f} °C "
      f"({100 * (start['MAE'] - end['MAE']) / start['MAE']:.0f}% better), almost all of it from better INPUTS.")
print(f"With the same weather inputs, linear regression gets {lin['MAE']:.2f} °C "
      f"(LSTM ahead by {lin['MAE'] - end['MAE']:.3f} °C MAE, {lin['RMSE'] - end['RMSE']:.3f} °C RMSE).")
print(f"Against Week 5's best ({WEEK5_BEST_TEST_MAE:.3f} °C): {end['MAE'] - WEEK5_BEST_TEST_MAE:+.3f} °C.")
print("Lessons: (1) data and features first, model type later; (2) always keep a simple baseline;")
print("(3) an LSTM earns its complexity on long sequences with patterns a straight line can't catch.")

banner("More ideas to try (see studyreport.md, section 8)")
for idea in ["Neighbouring cities' weather on every day (Oslo, Malmö): weather moves west -> east",
             "Predict the CHANGE (today - yesterday) instead of the temperature itself",
             "Retrain on train + validation (2000-2007) with the chosen settings before the final test",
             "Stack 2 LSTM layers, or try a GRU (a simpler gated cell with 3 gates' worth of weights)",
             "Forecast several days ahead (sequence -> sequence), where memory matters more",
             "Predict a range (e.g. 10th-90th percentile), not just one number"]:
    print("  -", idea)
