This project focuses on Remaining Useful Life (RUL) prediction for aircraft engines using the NASA C-MAPSS FD001 dataset.
The goal is to predict how many cycles remain before engine failure using multivariate sensor time-series data.

A Long Short-Term Memory (LSTM) network is used to model temporal degradation patterns, and Optuna is used for systematic hyperparameter optimization.

Dataset Description (C-MAPSS FD001)

The dataset consists of three files:

train_FD001.txt

Complete run-to-failure data

Used for training and validation

test_FD001.txt

Partial engine life (failure not observed)

RUL_FD001.txt

One RUL value per test engine

Corresponds to the last observed cycle only

Each row contains:

engine_id

cycle

3 operating conditions (op1, op2, op3)

21 sensor readings (s1 to s21)

 Data Processing & Feature Engineering
✔ Key preprocessing steps

Computed RUL for training data:

RUL = max_cycle_per_engine − current_cycle


Dropped non-informative sensors (near-zero variance):

s1, s5, s6, s10, s12, s13, s16, s18, s19


Dropped:

engine_id (identifier only)

cycle (relative time index)

op3 (constant)

Retained:

op1, op2

14 meaningful sensor features

Applied standardization:

Scaler fitted on training split only

Applied to validation and test

 Sequence Construction (Sliding Window)

Fixed window size (optimized using Optuna)

Sequences are created engine-wise

No mixing of timestamps across engines

Final LSTM input shape:

(batch_size, window_size, num_features)


For test data:

Only the last window per engine is used

Labels come from RUL_FD001.txt

 Model Architecture (LSTM)
Input (window_size × features)
        ↓
LSTM (hidden_size, num_layers)
        ↓
Last timestep output
        ↓
Fully Connected Layer
        ↓
Predicted RUL

Model characteristics

Many-to-one sequence modeling

Regression output

Implemented using PyTorch

⚙️ Hyperparameter Optimization (Optuna)

Optuna is used to tune:

window_size

hidden_size

num_layers

dropout

learning_rate

batch_size

Optimization strategy

Objective: minimize validation RMSE

Training inside Optuna uses fewer epochs (fast search)

Test set is never used during tuning

Evaluation Metrics
Primary metric: RMSE

Used for:

Training monitoring

Validation (Optuna objective)

Final test evaluation

Why RMSE?

Standard metric in RUL literature

Penalizes large errors

Interpretable in units of cycles

(NASA asymmetric score can be added later as an extension)

 How to Run
1️⃣ Install dependencies
pip install numpy pandas torch optuna scikit-learn

2️⃣ Run the notebook / script

Data loading & preprocessing

Sequence construction

Optuna hyperparameter search

Final model training

Test evaluation

3️⃣Saved artifact
best_rul_lstm.pth


Contains the trained model with best hyperparameters.

 Final Outputs

Best hyperparameters found by Optuna

Validation RMSE per trial

Final Test RMSE on FD001

Saved PyTorch model

 Key Learnings

RUL prediction is inherently a temporal problem

LSTM captures degradation trajectories better than static models

Engine-wise windowing is critical to avoid data leakage

Hyperparameter tuning significantly improves performance

 Future Improvements

Add NASA asymmetric scoring function

Try GRU or Transformer models

Include RUL capping

Use early stopping + Optuna pruning

Cross-dataset evaluation (FD002–FD004)
