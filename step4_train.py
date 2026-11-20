"""
STEP 4 - Train the LSTM on 6 years of real data, watching the validation set.

  4.1 Build the LSTM with random weights and look at its (bad) first guesses
  4.2 Train: mini-batch BPTT + Adam + gradient clipping + early stopping on validation loss
  4.3 Learning curves: training vs validation loss, epoch by epoch
  4.4 Save the trained weights (and the scaler) to models/lstm.npz
Run:  python step4_train.py      (~10 seconds)
"""
import numpy as np

from common import (banner, load_data, plot_lines, scores, BATCH_SIZE, CLIP, EPOCHS, FEATURES,
                    HIDDEN, LEARNING_RATE, MODEL_PATH, PATIENCE, SEED, WINDOW)
from lstm import LSTM, train

# ----- 4.1 Data and an untrained LSTM -----
banner("4.1 An untrained LSTM: random weights give useless guesses")
splits, scaler = load_data()
(X_tr, y_tr, _), (X_va, y_va, d_va) = splits["train"], splits["val"]
y_va_c = scaler.to_celsius(y_va)                                  # real validation answers in °C

net = LSTM(X_tr.shape[2], HIDDEN, seed=SEED)
guess = scaler.to_celsius(net.predict(X_va))                      # scaled output -> °C
print(f"Validation MAE before training: {scores(y_va_c, guess)['MAE']:.2f} °C")
print("First 5 validation days, real vs guessed (°C):")
for day, real, g in zip(d_va[:5], y_va_c[:5], guess[:5]):
    print(f"  {day.date()}   real {real:6.1f}   guess {g:6.1f}")
print("Untrained, it guesses roughly the training average (~5 °C) for every day.")

# ----- 4.2 Train -----
banner("4.2 Training: read 14 days -> guess -> loss -> BPTT -> Adam step, batch after batch")
updates = int(np.ceil(len(X_tr) / BATCH_SIZE))
print(f"{len(X_tr)} training sequences / batch size {BATCH_SIZE} = {updates} weight updates per epoch")
print(f"each update: forward through {WINDOW} days, backward through {WINDOW} days (BPTT)")
print(f"Adam learning rate {LEARNING_RATE}, gradient clipping at length {CLIP}, "
      f"at most {EPOCHS} epochs, early-stopping patience {PATIENCE}")
print("loss = mean squared error on SCALED values (1.0 = as bad as always guessing the average)\n")
history, best_epoch = train(net, X_tr, y_tr, X_va, y_va, epochs=EPOCHS, lr=LEARNING_RATE,
                            batch_size=BATCH_SIZE, patience=PATIENCE, clip=CLIP, seed=SEED,
                            log_every=2)
print(f"\nBest validation loss at epoch {best_epoch}: {history['val'][best_epoch - 1]:.4f}"
      f"  -> the weights from that epoch are kept")

# ----- 4.3 Learning curves -----
banner("4.3 Learning curves: is the LSTM learning, underfitting or overfitting?")
plot_lines({"train loss": history["train"], "validation loss": history["val"]},
           "Learning curve - loss per epoch (scaled MSE)", "epoch", "loss")
tr_best, va_best = history["train"][best_epoch - 1], history["val"][best_epoch - 1]
print("How to read it:")
print("  * both lines drop fast in the first epochs -> the LSTM learns the main pattern quickly")
print("    (Adam takes big, well-sized steps at the start)")
print("  * then both flatten out, with small wiggles from the random batch order")
print(f"  * at the best epoch: train {tr_best:.4f} vs validation {va_best:.4f} (gap {va_best - tr_best:+.4f})")
print("    -> the lines stay together: NOT overfitting. ~1300 weights, ~2200 noisy sequences.")
print("  * overfitting would look like: validation loss rising while training loss keeps falling;")
print("    early stopping guards against it by keeping the best-validation weights.")
print("  * the floor (~0.09) is weather that 14 days of temperatures can't explain (noise).")
rmse_c = np.sqrt(va_best) * scaler.std["temp_min"]                  # undo the scaling
print(f"\nIn °C: validation RMSE = sqrt(loss) × temp_min std = {rmse_c:.2f} °C")
pred_va = scaler.to_celsius(net.predict(X_va))
print(f"Validation MAE after training: {scores(y_va_c, pred_va)['MAE']:.2f} °C")

# ----- 4.4 Save -----
banner("4.4 Save the trained LSTM")
net.save(MODEL_PATH, mean=scaler.mean.to_numpy(), std=scaler.std.to_numpy(),
         features=np.array(FEATURES), best_epoch=best_epoch)
print(f"Saved {net.count_params()} parameters + scaler statistics -> models/{MODEL_PATH.name}")
print("A 'model file' is just these numbers. Step 5 loads them to predict the test years.")
