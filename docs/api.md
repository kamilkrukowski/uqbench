# API Documentation

## Models

### BayesianMLP

Bayesian Multi-Layer Perceptron using Bayes by Backprop.

### FFN

Feedforward Neural Network with Dropout.

## Evaluation

### Calibration Metrics

- `expected_calibration_error`: Calculate ECE
- `brier_score`: Calculate Brier score
- `calibration_curve`: Generate calibration curve data

### Out-of-Distribution Detection

- `predict_with_uncertainty`: Predict with uncertainty estimation
- `detect_ood`: Detect out-of-distribution samples

