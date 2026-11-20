"""
An LSTM (Long Short-Term Memory) network written from scratch with NumPy only (no ML library).
Every line maps to one formula in studyreport.md, section 5.

The LSTM reads a sequence ONE DAY AT A TIME and carries two memory vectors from day to day:
  c = cell state   (long-term memory, the "conveyor belt")
  h = hidden state (short-term memory / the output of the cell at that step)
After the last day, one linear layer turns h into the prediction ŷ.

Shapes (n = sequences in a batch, T = days per sequence, F = features per day, H = HIDDEN):
  X       (n, T, F)        the input sequences
  W       (F + H, 4H)      ONE big weight matrix for all 4 gates, applied to [x_t, h_{t-1}]
  b       (4H,)            the 4 gates' biases
  Wy, by  (H, 1), (1,)     the output layer:  ŷ = h_T · Wy + by
Gate order inside the 4H columns: i (input), f (forget), g (candidate), o (output),
the same order PyTorch uses, so step 3 can copy these weights into torch.nn.LSTM.
"""
import numpy as np


def sigmoid(z):
    """Squash any number into (0, 1): a 'how open is this gate' value."""
    return 1 / (1 + np.exp(-z))


def mse(y_pred, y_true):
    """Mean squared error = the LOSS we minimise:  L = mean((ŷ - y)²)"""
    return float(np.mean((y_pred - y_true) ** 2))


class LSTM:
    """One LSTM layer (n_in features per day -> n_hidden memory size) + a linear output layer."""

    def __init__(self, n_in, n_hidden, seed=42):
        rng = np.random.default_rng(seed)
        H = n_hidden
        # Small random weights (std 1/sqrt(inputs to a gate)), so the gates start near "half open".
        self.p = {"W": rng.normal(0, 1 / np.sqrt(n_in + H), size=(n_in + H, 4 * H)),
                  "b": np.zeros(4 * H),
                  "Wy": rng.normal(0, 1 / np.sqrt(H), size=(H, 1)),
                  "by": np.zeros(1)}
        self.p["b"][H:2 * H] = 1.0    # forget-gate bias 1 -> starts by REMEMBERING (a classic trick)
        self.adam = None              # optimiser memory, created on the first update

    # ------------------------------------------------------------------ forward pass
    def forward(self, X):
        """Read every sequence day by day; return ŷ (n, 1). Keeps every step's values in
        self.cache, because backward() needs them (backpropagation through time)."""
        W, b, H = self.p["W"], self.p["b"], self.p["Wy"].shape[0]
        n, T, _ = X.shape
        h, c = np.zeros((n, H)), np.zeros((n, H))      # memory starts empty for every sequence
        self.cache = []
        for t in range(T):                             # one day at a time, oldest first
            xh = np.concatenate([X[:, t], h], axis=1)  # today's input next to yesterday's memory
            z = xh @ W + b                             # all 4 gates in one matrix multiplication
            i = sigmoid(z[:, :H])                      # input gate:  how much new info to write
            f = sigmoid(z[:, H:2 * H])                 # forget gate: how much old memory to keep
            g = np.tanh(z[:, 2 * H:3 * H])             # candidate:   the new info itself (-1..1)
            o = sigmoid(z[:, 3 * H:])                  # output gate: how much memory to show
            c_prev = c
            c = f * c_prev + i * g                     # update the long-term memory
            h = o * np.tanh(c)                         # what the cell outputs at this step
            self.cache.append((xh, i, f, g, o, c_prev, c))
        self.h_last = h
        return h @ self.p["Wy"] + self.p["by"]         # only the LAST step makes the prediction

    predict = forward                                  # using a trained LSTM = just the forward pass

    # ------------------------------------------------------------------ backward pass
    def backward(self, d_yhat):
        """Backpropagation THROUGH TIME (BPTT): walk the steps backwards with the chain rule.

        d_yhat = dL/dŷ, shape (n, 1). For MSE that is 2(ŷ - y)/n.
        Returns (grads: same keys as self.p, dX: dL/dX with the shape of X).
        The two error signals that flow back in time are dh (into h_{t-1}) and dc (into c_{t-1}).
        """
        W, H = self.p["W"], self.p["Wy"].shape[0]
        F = W.shape[0] - H
        grads = {"W": np.zeros_like(W), "b": np.zeros_like(self.p["b"]),
                 "Wy": self.h_last.T @ d_yhat,          # output layer: (what came in) × (error)
                 "by": d_yhat.sum(axis=0)}
        dh = d_yhat @ self.p["Wy"].T                   # blame arriving at the last hidden state
        dc = np.zeros_like(dh)                         # no blame on the cell after the last step
        dX = np.zeros((len(d_yhat), len(self.cache), F))
        for t in reversed(range(len(self.cache))):     # last day first
            xh, i, f, g, o, c_prev, c = self.cache[t]
            tanh_c = np.tanh(c)
            do = dh * tanh_c                           # from h = o · tanh(c)
            dc = dc + dh * o * (1 - tanh_c ** 2)       # c gets blame via h AND from step t+1
            di, df, dg = dc * g, dc * c_prev, dc * i   # from c = f · c_prev + i · g
            dz = np.concatenate([di * i * (1 - i),     # back through each gate's squashing:
                                 df * f * (1 - f),     #   sigmoid' = s(1 - s)
                                 dg * (1 - g ** 2),    #   tanh'    = 1 - tanh²
                                 do * o * (1 - o)], axis=1)
            grads["W"] += xh.T @ dz                    # the SAME W is used at every step,
            grads["b"] += dz.sum(axis=0)               # so its gradients are ADDED over time
            dxh = dz @ W.T                             # blame for [x_t, h_{t-1}]
            dX[:, t] = dxh[:, :F]
            dh = dxh[:, F:]                            # -> flows to the previous day's h
            dc = dc * f                                # -> flows to the previous day's c
        return grads, dX                               # (multiplied by the forget gate only!)

    # ------------------------------------------------------------------ update (Adam)
    def step(self, grads, lr, clip=1.0, beta1=0.9, beta2=0.999, eps=1e-8):
        """One Adam update with gradient clipping.

        Clipping: if the whole gradient vector is longer than `clip`, shrink it to length `clip`
                  (protects against the 'exploding gradient' jumps recurrent nets are known for).
        Adam:     m = running average of the gradient       (direction, smoothed)
                  v = running average of the gradient²      (typical size)
                  w ← w - lr · m̂ / (√v̂ + eps)              (every weight gets its own step size)
        """
        norm = np.sqrt(sum(np.sum(g ** 2) for g in grads.values()))
        if norm > clip:
            grads = {k: g * clip / norm for k, g in grads.items()}
        if self.adam is None:
            self.adam = {"t": 0, "m": {k: np.zeros_like(v) for k, v in self.p.items()},
                         "v": {k: np.zeros_like(v) for k, v in self.p.items()}}
        a = self.adam
        a["t"] += 1
        for k in self.p:
            a["m"][k] = beta1 * a["m"][k] + (1 - beta1) * grads[k]
            a["v"][k] = beta2 * a["v"][k] + (1 - beta2) * grads[k] ** 2
            m_hat = a["m"][k] / (1 - beta1 ** a["t"])  # bias correction: m and v start at 0,
            v_hat = a["v"][k] / (1 - beta2 ** a["t"])  # so early averages are scaled up
            self.p[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ------------------------------------------------------------------ helpers
    def get_params(self):
        return {k: v.copy() for k, v in self.p.items()}

    def set_params(self, params):
        self.p = {k: v.copy() for k, v in params.items()}

    def count_params(self):
        return sum(v.size for v in self.p.values())

    def save(self, path, **extra):
        """Save the weights (and anything else, e.g. the scaler) into one .npz file."""
        path.parent.mkdir(exist_ok=True)
        np.savez(path, **self.p, **extra)

    @classmethod
    def load(cls, path):
        """Rebuild a saved LSTM. Returns (lstm, the whole saved file)."""
        data = np.load(path, allow_pickle=True)
        net = cls.__new__(cls)                 # skip __init__: we load the weights instead
        net.p = {k: data[k] for k in ("W", "b", "Wy", "by")}
        net.adam = None
        return net, data


def train(net, X_tr, y_tr, X_va, y_va, epochs=80, lr=0.005, batch_size=32, patience=10,
          clip=1.0, seed=42, log_every=0):
    """The training loop: mini-batch Adam with early stopping on validation loss.

    One EPOCH = one pass over all training sequences, in small BATCHES:
        1. guess    ŷ = forward(X_batch)                    read 14 days, predict day 15
        2. score    dL/dŷ = 2(ŷ - y)/n                      (gradient of the MSE loss)
        3. blame    gradients = backward(dL/dŷ)             backpropagation through time
        4. nudge    step(gradients)                         Adam + clipping
    After each epoch we measure the loss on TRAIN and VALIDATION data. Early stopping: if
    validation loss hasn't improved for `patience` epochs, stop and restore the best weights.
    """
    rng = np.random.default_rng(seed)
    history = {"train": [], "val": []}
    best_val, best_params, best_epoch, wait = np.inf, net.get_params(), 0, 0
    for epoch in range(1, epochs + 1):
        # Shuffle the ORDER of the sequences. Safe: each sequence carries its own 14 days of
        # past inside it, and the validation sequences are still later in time.
        order = rng.permutation(len(X_tr))
        for start in range(0, len(order), batch_size):
            batch = order[start:start + batch_size]
            y_hat = net.forward(X_tr[batch])
            grads, _ = net.backward(2 * (y_hat - y_tr[batch]) / len(batch))
            net.step(grads, lr, clip)

        history["train"].append(mse(net.predict(X_tr), y_tr))
        history["val"].append(mse(net.predict(X_va), y_va))
        if log_every and (epoch == 1 or epoch % log_every == 0):
            print(f"  epoch {epoch:>3}   train loss {history['train'][-1]:.4f}   "
                  f"val loss {history['val'][-1]:.4f}")

        if history["val"][-1] < best_val:              # new best on validation -> remember it
            best_val, best_params, best_epoch, wait = history["val"][-1], net.get_params(), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                if log_every:
                    print(f"  early stop at epoch {epoch}: no val improvement for {patience} epochs")
                break
    net.set_params(best_params)                        # roll back to the best epoch
    return history, best_epoch
