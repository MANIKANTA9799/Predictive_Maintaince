[![Live Demo](https://img.shields.io/badge/Live%20Demo-Open%20Streamlit%20App-brightgreen?style=for-the-badge&logo=streamlit)](https://predictivemaintaince-appcfspnqqbnxskrsrcvhro.streamlit.app/)
# Aircraft Engine Remaining Useful Life (RUL) Prediction using LSTM

Predicting when an aircraft engine will fail is a critical problem in **predictive maintenance**.
This project builds a deep learning model to estimate the **Remaining Useful Life (RUL)** of turbofan engines using **multivariate time-series sensor data**.

The model leverages a **Long Short-Term Memory (LSTM)** network to learn degradation patterns over time, while **Optuna** is used for automated hyperparameter optimization.

**Dataset:** NASA C-MAPSS FD001

---

# Project Overview

Aircraft engines degrade gradually during operation. Instead of waiting for failure, predictive models estimate how long an engine can continue operating safely.

This project develops a **data-driven predictive maintenance pipeline** that:

* Learns degradation patterns from sensor signals
* Models temporal dependencies using LSTM
* Tunes model hyperparameters using Optuna
* Predicts the remaining cycles before failure

---

# Dataset: NASA C-MAPSS FD001

The **Commercial Modular Aero-Propulsion System Simulation (C-MAPSS)** dataset was created by NASA to simulate aircraft engine degradation.

## Dataset Files

| File              | Purpose                            |
| ----------------- | ---------------------------------- |
| `train_FD001.txt` | Complete engine life until failure |
| `test_FD001.txt`  | Partial engine trajectories        |
| `RUL_FD001.txt`   | True RUL for test engines          |

Each record includes:

* `engine_id`
* `cycle`
* `op1, op2, op3` (operating conditions)
* `s1 – s21` (sensor measurements)

---

# Data Preprocessing & Feature Engineering

## RUL Computation

For the training dataset:

RUL = max_cycle_of_engine − current_cycle

This produces a **regression label for every timestep**.

---

## Removing Non-Informative Sensors

Sensors with **near-zero variance** were removed:

s1, s5, s6, s10, s12, s13, s16, s18, s19

Dropped columns:

engine_id → identifier only
cycle → relative time index
op3 → constant value

---

## Final Feature Set

The model uses:

* Operating conditions: `op1`, `op2`
* 14 meaningful sensors

These features capture **engine operating states and degradation signals**.

---

## Feature Scaling

To ensure stable model training:

* Standardization is applied
* Scaler is fitted on **training data only**
* Same scaler applied to **validation and test sets**

This prevents **data leakage**.

---

# Sequence Construction (Sliding Window)

Since degradation is **temporal**, the model receives sequences instead of individual rows.

Key design decisions:

* Fixed window size (optimized via Optuna)
* Sequences created **engine-wise**
* No mixing of timestamps across engines

Final LSTM input format:

(batch_size, window_size, num_features)

### Test Data Handling

For each engine in the test set:

* Only the **last window** is used
* Target labels come from `RUL_FD001.txt`

---

# Model Architecture

The model follows a **Many-to-One LSTM architecture**:

Input Sequence (window_size × features)
↓
LSTM (hidden_size, num_layers)
↓
Last Time-Step Output
↓
Fully Connected Layer
↓
Predicted RUL

### Model Characteristics

* Temporal sequence modeling
* Regression output
* Implemented in **PyTorch**

---

# Hyperparameter Optimization (Optuna)

Instead of manually tuning parameters, **Optuna** performs automated hyperparameter search.

Parameters optimized:

* window_size
* hidden_size
* num_layers
* dropout
* learning_rate
* batch_size

### Optimization Strategy

Objective: **Minimize Validation RMSE**

During Optuna trials:

* Training uses fewer epochs for faster search
* The test set is **never used during tuning**

---

# Evaluation Metric

### Primary Metric: Root Mean Squared Error (RMSE)

Used for:

* Training monitoring
* Validation during hyperparameter tuning
* Final test evaluation

### Why RMSE?

* Standard metric in RUL prediction literature
* Penalizes large prediction errors
* Interpretable in **engine cycles**

---

# Streamlit Deployment

A **Streamlit dashboard** is included to allow users to interactively predict the Remaining Useful Life of an engine using the trained LSTM model.

The dashboard allows users to:

* Upload engine sensor data
* Run the trained model
* View predicted Remaining Useful Life

---

# How to Run

## 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/Predictive_Maintenance.git
cd Predictive_Maintenance
```

## 2. Install Dependencies

Install all required libraries:

```bash
pip install -r requirements.txt
```

or manually install:

```bash
pip install numpy pandas torch optuna scikit-learn streamlit
```

## 3. Run the Streamlit App

Start the interactive dashboard:

```bash
streamlit run app/app.py
```

After running this command, Streamlit will start a local server and display a URL such as:

```
Local URL: http://localhost:8501
```

Open this URL in your browser to use the **Aircraft Engine RUL Prediction dashboard**.

---

# Saved Model

After training, the best model is saved as:

best_model/best_rul_lstm.pth

This file contains the **trained PyTorch LSTM model with optimized hyperparameters**.

---

# Final Outputs

The project produces:

* Best hyperparameters discovered by Optuna
* Validation RMSE per trial
* Final Test RMSE on FD001
* Saved trained PyTorch model
* Interactive Streamlit dashboard for predictions

---

# Key Learnings

* RUL prediction is fundamentally a **time-series problem**
* Engine degradation follows **temporal trajectories**
* LSTM effectively captures **sequential degradation patterns**
* Engine-wise windowing prevents **data leakage**
* Hyperparameter optimization significantly improves performance
* Streamlit enables easy **deployment of ML models as web apps**
