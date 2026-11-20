# Week 6 — Forecasting Tomorrow's Temperature with an LSTM

A small, beginner-friendly **time-series** project. It uses an **LSTM (Long Short-Term Memory) network written from scratch in NumPy**, so you can see every formula: the four gates, the cell memory, backpropagation through time and the Adam update. PyTorch is used only to check the maths and to show the "library way".
It predicts tomorrow's minimum temperature in **Stockholm, Sweden** from the 14 days before it. The data, split and test years are the same as in Weeks 2 and 5, so all models can be compared directly.
Data: the [Weather prediction dataset](https://github.com/florian-huber/weather_prediction_dataset) (Huber et al., built from ECA&D station data; CC BY 4.0 on Zenodo, MIT in the repo).
Every chart prints in the terminal and is also saved as a PNG in `results/`. The full explanation is in **[studyreport.md](studyreport.md)**.

```
weather_lstm/
├── data/weather_prediction_dataset.csv   18 European cities, 2000-2010, 165 columns
├── docs/                      architecture diagrams (LSTM cell, unrolled LSTM)
├── models/                    the trained LSTM (created by step 4)
├── results/                   PNG charts (created by steps 1, 3, 4, 5, 6)
├── common.py                  settings, data pipeline, scaling, metrics, terminal charts
├── lstm.py                    THE MODEL: LSTM + BPTT + Adam + training loop, NumPy only
├── step1_explore.py           data analysis (quality, seasons, how far back the past matters)
├── step2_sequences.py         data processing & transformation into 3-D sequences
├── step3_lstm_from_scratch.py the maths by hand, gradient check, PyTorch check, vanishing gradients
├── step4_train.py             training with validation, early stopping, learning curves
├── step5_evaluate.py          test metrics, baselines, error analysis, which days matter
├── step6_improve.py           experiments, ensemble, PyTorch, final test
├── requirements.txt           pinned library versions
└── studyreport.md             the week's assignment report
```

## Setup (separate virtual environment for this project)

```powershell
cd weather_lstm
python -m venv .venv
.venv\Scripts\activate            # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Run the steps in order

| Step | Command | What you learn |
|---|---|---|
| 1 | `python step1_explore.py` | The dataset, hidden missing values, yearly cycle, how long the "memory" of weather lasts |
| 2 | `python step2_sequences.py` | Season on a circle, train-only scaling, sliding window → `(samples, days, features)`, time split |
| 3 | `python step3_lstm_from_scratch.py` | Gates, cell state, BPTT by hand; gradient check; same output as PyTorch; why LSTMs exist |
| 4 | `python step4_train.py` | Mini-batch BPTT + Adam + clipping, validation, early stopping, learning curve (~10 s) |
| 5 | `python step5_evaluate.py` | MAE / RMSE / R², baselines, error analysis, which past days the LSTM uses |
| 6 | `python step6_improve.py` | What helps (and what doesn't), ensemble, PyTorch, final test (~75 s) |

## Results (test years 2008–2009, 731 days)

| Model | MAE °C |
|---|---|
| Tomorrow = today (persistence) | 1.680 |
| Linear regression, same 14 days | 1.624 |
| Week 5 feedforward network | 1.620 |
| **LSTM, 14 days of temperature + season** (step 5) | **1.587** |
| Linear regression + daily weather | 1.334 |
| Week 5 best: ensemble of 5 networks + weather | 1.312 |
| **Ensemble of 5 LSTMs + daily weather** (step 6) | **1.297** |

Main lessons: better **inputs** helped far more than the type or size of the model, and the LSTM's lead over a simple linear model is small (~0.04 °C). Always compare against a baseline.

Charts look best in a terminal at least 100 characters wide (VS Code terminal, Windows Terminal).
