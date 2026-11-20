"""
STEP 3 - The mathematics of an LSTM, small enough to follow with a calculator.

  Part A  One LSTM with 1 memory cell reads 3 days: forward pass -> loss -> backpropagation
          through time (BPTT) -> update. Every gate value is printed.
  Part B  Proof that our BPTT is right: compare it with a brute-force "wiggle" test
  Part C  Proof that our LSTM is the real thing: PyTorch's nn.LSTM gives the same output
  Part D  WHY the LSTM exists: the vanishing gradient of a simple RNN vs the LSTM's cell path
  Part E  The real LSTM for this project: shapes and parameter count
Run:  python step3_lstm_from_scratch.py
"""
import numpy as np
import torch

from common import banner, load_data, plot_lines, HIDDEN
from lstm import LSTM, mse, sigmoid

np.set_printoptions(precision=4, suppress=True)

# =============================================================================== Part A
banner("Part A - an LSTM with ONE memory cell reads 3 days, by hand")
xs = [1.0, 0.5, -0.5]          # three scaled temperatures, oldest first (a warm spell cooling down)
y = -0.8                       # the real (scaled) temperature of day 4: colder still
# Hand-picked weights so every number is easy to follow.  For each gate:  z = w_x·x + w_h·h + b
#            i     f     g     o
w_x = np.array([0.5, 0.0, 1.0, 0.5])    # how strongly today's input drives each gate
w_h = np.array([0.0, 0.5, 0.5, 0.0])    # how strongly yesterday's h drives each gate
bias = np.array([0.0, 1.0, 0.0, 0.0])   # forget bias 1: start by remembering
w_y, b_y = 2.0, 0.0                     # output layer: ŷ = 2·h₃
lr = 0.3

print("FORWARD PASS  (h and c start at 0)")
print("  t    x    h_prev |  i=σ(.)  f=σ(.)  g=tanh(.)  o=σ(.) |  c = f·c_prev + i·g   h = o·tanh(c)")
h, c, steps = 0.0, 0.0, []
for t, x in enumerate(xs, start=1):
    z = w_x * x + w_h * h + bias                    # the 4 gate pre-activations
    i, f, g, o = sigmoid(z[0]), sigmoid(z[1]), np.tanh(z[2]), sigmoid(z[3])
    c_prev, h_prev = c, h
    c = f * c_prev + i * g
    h = o * np.tanh(c)
    steps.append((x, h_prev, c_prev, i, f, g, o, c))
    print(f"  {t}  {x:+.1f}  {h_prev:+.4f} | {i:.4f}  {f:.4f}  {g:+.4f}    {o:.4f} | "
          f"c{t} = {f:.3f}·{c_prev:+.3f} + {i:.3f}·{g:+.3f} = {c:+.4f}   h{t} = {h:+.4f}")
y_hat = w_y * h + b_y
loss = (y_hat - y) ** 2
print(f"\n  ŷ = w_y·h₃ + b_y = {w_y}·{h:+.4f} = {y_hat:+.4f}      true y = {y}")
print(f"  loss = (ŷ - y)² = ({y_hat:+.4f} - ({y}))² = {loss:.4f}")
print("  Read the gates: the forget gate f ≈ 0.73-0.76 keeps most of the old memory each day; the")
print("  input gate i lets in less when x is negative; on day 3 the candidate g turns negative,")
print("  so the memory c drops: the cell 'notices' the cooling.")

print("\nBACKWARD PASS THROUGH TIME  (chain rule, from the loss back to day 1)")
d_yhat = 2 * (y_hat - y)
print(f"  dL/dŷ = 2(ŷ - y) = {d_yhat:+.4f}   (positive -> the prediction is too HIGH)")
g_wy, g_by = d_yhat * h, d_yhat
dh, dc = d_yhat * w_y, 0.0               # blame arriving at h₃; nothing comes from 'day 4'
g_wx, g_wh, g_b = np.zeros(4), np.zeros(4), np.zeros(4)
for t in reversed(range(len(xs))):
    x, h_prev, c_prev, i, f, g, o, c = steps[t]
    do = dh * np.tanh(c)                              # h = o·tanh(c)
    dc = dc + dh * o * (1 - np.tanh(c) ** 2)          # c gets blame from h AND from step t+1
    dz = np.array([dc * g * i * (1 - i),              # i: c = ... + i·g      σ' = σ(1-σ)
                   dc * c_prev * f * (1 - f),         # f: c = f·c_prev + ...
                   dc * i * (1 - g ** 2),             # g: c = ... + i·g      tanh' = 1 - tanh²
                   do * o * (1 - o)])                 # o: h = o·tanh(c)
    g_wx += dz * x                                    # same weights every day -> ADD the blame
    g_wh += dz * h_prev
    g_b += dz
    print(f"  day {t + 1}: dL/dh = {dh:+.4f}  dL/dc = {dc:+.4f}  -> dL/dz[i,f,g,o] = {dz}")
    dh = np.dot(dz, w_h)                              # blame passed to the previous day's h
    dc = dc * f                                       # ... and to the previous day's c (×f only!)
print(f"\n  dL/dw_x = {g_wx}     dL/dw_h = {g_wh}")
print(f"  dL/db   = {g_b}     dL/dw_y = {g_wy:+.4f}   dL/db_y = {g_by:+.4f}")
print("  Note dL/dz[f] = 0 on day 1: c_prev was 0, so there was nothing to forget yet -> no blame.")
print("  dL/dc shrinks as it travels back (1.70 -> 1.46 -> 1.28) but does not vanish: on the cell")
print("  path it is only multiplied by f each day (plus new blame arriving through h).")

print(f"\nUPDATE (plain gradient descent, learning rate {lr}):  new = old - lr × gradient")
w_x2, w_h2, b2 = w_x - lr * g_wx, w_h - lr * g_wh, bias - lr * g_b
w_y2, b_y2 = w_y - lr * g_wy, b_y - lr * g_by
print(f"  w_x: {w_x} -> {w_x2}")
print(f"  w_h: {w_h} -> {w_h2}")
print(f"  b:   {bias} -> {b2}     w_y: {w_y} -> {w_y2:.4f}   b_y: {b_y} -> {b_y2:.4f}")

# Same numbers from lstm.py: one sample, 3 days, 1 feature, 1 memory cell.
net = LSTM(n_in=1, n_hidden=1)
net.p = {"W": np.vstack([w_x, w_h]), "b": bias.copy(), "Wy": np.array([[w_y]]), "by": np.array([b_y])}
X = np.array(xs).reshape(1, 3, 1)                     # (1 sample, 3 days, 1 feature)
y_hat_net = net.forward(X)
grads, _ = net.backward(2 * (y_hat_net - y))
print(f"\nlstm.py gives the same ŷ ({y_hat_net[0, 0]:+.4f}) and the same gradients:",
      np.allclose(grads["W"], np.vstack([g_wx, g_wh])) and np.allclose(grads["b"], g_b))
net.p = {"W": np.vstack([w_x2, w_h2]), "b": b2, "Wy": np.array([[w_y2]]), "by": np.array([b_y2])}
loss_new = mse(net.forward(X), np.array([[y]]))
print(f"AFTER ONE STEP: ŷ = {net.forward(X)[0, 0]:+.4f}, loss = {loss_new:.6f} (was {loss:.4f})")
print("One sample is easy to fit in one step. Real training (step 4) averages the gradient over")
print("32 sequences per step, and needs thousands of small steps to fit 2178 of them at once.")

# =============================================================================== Part B
banner("Part B - gradient check: is backpropagation through time correct?")
print("Brute force: nudge ONE weight by ±ε, re-run the forward pass, and measure the slope:")
print("    slope ≈ (loss(w + ε) - loss(w - ε)) / 2ε        (no calculus, just two forward passes)")
print("BPTT gives every slope in one backward pass. If the maths is right, they must match.\n")
splits, _ = load_data()
X, Y = splits["train"][0][:64], splits["train"][1][:64]     # a small batch of real sequences
net = LSTM(X.shape[2], HIDDEN, seed=0)
grads, _ = net.backward(2 * (net.forward(X) - Y) / len(Y))
eps = 1e-5
for name, idx, label in [("W", (0, 3), "W[x=temp_min -> input gate 4]"),
                         ("W", (5, 20), "W[h2 -> forget gate 5]"),
                         ("b", (40,), "b[candidate 9]"),
                         ("Wy", (3, 0), "Wy[h4 -> output]")]:
    original = net.p[name][idx]
    net.p[name][idx] = original + eps
    loss_plus = mse(net.forward(X), Y)
    net.p[name][idx] = original - eps
    loss_minus = mse(net.forward(X), Y)
    net.p[name][idx] = original                                    # put it back
    numeric = (loss_plus - loss_minus) / (2 * eps)
    print(f"  {label:<32} brute force {numeric:+.8f}   BPTT {grads[name][idx]:+.8f}")
print("\nThe slopes agree to ~8 decimals: BPTT is just a fast way to get them all.")

# =============================================================================== Part C
banner("Part C - is our LSTM the 'real' LSTM? Copy its weights into PyTorch's nn.LSTM")
torch_lstm = torch.nn.LSTM(input_size=X.shape[2], hidden_size=HIDDEN, batch_first=True)
torch_out = torch.nn.Linear(HIDDEN, 1)
F = X.shape[2]
with torch.no_grad():                                   # PyTorch stores W split in two, transposed
    torch_lstm.weight_ih_l0.copy_(torch.tensor(net.p["W"][:F].T))    # the x part of W
    torch_lstm.weight_hh_l0.copy_(torch.tensor(net.p["W"][F:].T))    # the h part of W
    torch_lstm.bias_ih_l0.copy_(torch.tensor(net.p["b"]))
    torch_lstm.bias_hh_l0.zero_()                                    # PyTorch has 2 biases; we use 1
    torch_out.weight.copy_(torch.tensor(net.p["Wy"].T))
    torch_out.bias.copy_(torch.tensor(net.p["by"]))
    seq_out, _ = torch_lstm(torch.tensor(X, dtype=torch.float32))   # h at every day
    torch_pred = torch_out(seq_out[:, -1]).numpy()                  # use the LAST day's h
ours = net.predict(X)
print("First 5 predictions (scaled):")
print("  ours   :", ours[:5, 0])
print("  PyTorch:", torch_pred[:5, 0])
print(f"Largest difference over 64 samples: {np.abs(ours - torch_pred).max():.1e}  "
      "(float32 rounding only)")
print("Same gate order (i, f, g, o), same formulas: our 100 lines of NumPy ARE an LSTM.")

# =============================================================================== Part D
banner("Part D - why LSTMs exist: how much error signal survives k days back in time?")
print("Simple RNN (1980s):  h_t = tanh(w·h_{t-1} + u·x_t). Going back one day multiplies the")
print("  error signal by  w · tanh'(z).  Typical: 0.8 × 0.6 = 0.48 per day  -> it VANISHES.")
print("LSTM cell path:      c_t = f·c_{t-1} + i·g.  Going back one day along c multiplies the")
print("  signal by just f, the forget gate, which the LSTM can learn to keep near 1 (e.g. 0.95).\n")
k = np.arange(1, 31)
rnn, lstm_path = 0.48 ** k, 0.95 ** k
for days in [1, 3, 7, 14, 30]:
    print(f"  {days:>2} days back:  simple RNN {rnn[days - 1]:.1e}   LSTM cell path {lstm_path[days - 1]:.3f}")
plot_lines({"simple RNN (×0.48 per day)": rnn, "LSTM cell path (×0.95 per day)": lstm_path},
           "Error signal left after k days back in time", "days back", "share left")
print("After 14 days the simple RNN keeps 0.00003 of the signal: it cannot learn from 2 weeks ago.")
print("The LSTM keeps 0.49. That 'constant error carousel' is the whole idea of the LSTM (1997).")
print("(If w were > 1 instead, the signal would EXPLODE: that's why we also clip gradients.)")

# =============================================================================== Part E
banner("Part E - the LSTM we train in step 4")
net = LSTM(X.shape[2], HIDDEN)
F, H = X.shape[2], HIDDEN
print(f"{F} features per day -> LSTM with {H} memory cells, read over {X.shape[1]} days -> 1 output\n")
print(f"  W   shape {net.p['W'].shape}  = ({F} inputs + {H} from h) × (4 gates × {H}) = {net.p['W'].size} weights")
print(f"  b   shape {net.p['b'].shape}      = 4 gates × {H}                         = {net.p['b'].size} biases")
print(f"  Wy  shape {net.p['Wy'].shape}    output layer                          = {net.p['Wy'].size} weights")
print(f"  by  shape {net.p['by'].shape}      output bias                           = {net.p['by'].size}")
print(f"\n  total learnable parameters: {net.count_params()}")
print(f"The SAME {net.count_params()} numbers are reused on every one of the 14 days (weight sharing):")
print("a 30-day window would need no extra weights. Week 5's network needed a new weight per lag.")
print("Training (step 4) = finding values of these parameters that make the loss small.")
