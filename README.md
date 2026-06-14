# BayesCal: Bayesian Neural Network Calibration Research

## Project Intent (TL;DR)

This is a practical evaluation of uncertainty-aware neural network methods from an applied ML engineering perspective.

It explores trade-offs between accuracy, calibration, compute, and operational complexity under a fixed task and architecture.  
It is not a benchmark and does not claim universal rankings.

### Scope, Intent, and Non-Goals

This repository is not a benchmark and does not claim to establish a global ranking of Bayesian neural network techniques. The goal is a practical, regime-scoped evaluation of common uncertainty-aware neural methods under a fixed architecture, dataset family, and compute regime, intended to surface engineering trade-offs rather than winners.

All results should be interpreted as conditional on: 
 - the synthetic task family used
 - the chosen model architecture and optimizer
 - the stated training and inference budgets
 - and the specific hyperparameter configurations explored.

## Operational Considerations and Failure Modes

This project evaluates uncertainty-aware neural network techniques with an emphasis on **practical trade-offs**, not just metric performance. The methods implemented here differ significantly in their computational cost, stability, and operational risk. The following notes summarize common failure modes and situations where each approach may be a poor fit in practice.

### Deterministic Feedforward Networks (with or without Dropout)
- **Primary failure mode**: Systematic overconfidence, especially under distribution shift.
- **Operational risk**: Confidence estimates are not meaningful for downstream risk-sensitive decisions.
- **Why not use**: When uncertainty is required for rejection, fallback logic, or human-in-the-loop review.

---

### MC Dropout
- **Primary failure mode**: Conflates epistemic and aleatoric uncertainty; calibration quality is sensitive to dropout rate and training dynamics.
- **Operational risk**: Requires multiple stochastic forward passes at inference time, increasing latency and variance.
- **Why not use**: When calibrated uncertainty under distribution shift is critical or inference latency is tightly constrained.

---

### Bayesian Neural Networks (Bayes by Backprop)
- **Primary failure mode**: Sensitivity to prior scale and KL weighting; risk of posterior collapse or under-dispersed uncertainty.
- **Operational risk**: More difficult to tune and debug than deterministic models; results can vary significantly across configurations.
- **Why not use**: When iteration speed, reproducibility, or ease of debugging is a priority.

---

### Deep Ensembles
- **Primary failure mode**: High training and memory cost due to multiple independent models.
- **Operational risk**: Increased deployment complexity and inference cost proportional to ensemble size.
- **Why not use**: When compute, memory, or model management overhead is constrained.

---

### Laplace Approximation
- **Primary failure mode**: Assumes local Gaussianity around a single mode; unreliable when the loss landscape is poorly conditioned or multi-modal.
- **Operational risk**: Quality depends strongly on the MAP solution and curvature approximation.
- **Why not use**: When training is unstable or when uncertainty far from the training manifold is important.

### MCMC-based Bayesian Neural Networks
- **Primary failure mode**: Extremely high computational cost and sensitivity to sampler configuration. Does not scale well to larger architectures.
- **Operational risk**: Requires careful convergence diagnostics (e.g., effective sample size, divergences, chain mixing); failures can be silent and difficult to detect without expertise.
- **Why not use**: Most online or real-time systems; scenarios where iteration speed, reproducibility, or operational simplicity are required.

## Method Selection Guidance

The following heuristics summarize when each class of method is typically a good fit in practice. These are **engineering guidelines**, not hard rules.

- **Tight latency or throughput budgets**  
  → Deterministic models with post-hoc calibration (e.g., temperature scaling) or Laplace approximations  
  *Rationale: Minimal inference overhead with reasonable calibration.*

- **Strong baseline performance with minimal complexity**  
  → Deterministic + post-hoc calibration  
  *Rationale: Often sufficient when data is i.i.d. and uncertainty is used primarily for confidence estimation.*

- **Best overall probabilistic performance under moderate compute budgets**  
  → Deep Ensembles  
  *Rationale: Frequently achieve strong NLL and robustness at the cost of increased training and inference compute.*

- **Exploration of epistemic uncertainty or model uncertainty analysis**  
  → Bayesian Neural Networks (e.g., Bayes by Backprop)  
  *Rationale: Explicitly model parameter uncertainty but require careful tuning.*

- **High-fidelity uncertainty reference or offline analysis**  
  → MCMC-based Bayesian Neural Networks  
  *Rationale: Rich posterior estimates, but impractical for most production inference.*

## Project Structure

```
bayescal/
├── README.md                 # Project overview and setup instructions
├── GOALS.md                  # Project goals and objectives
├── pyproject.toml            # Modern Python project configuration
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore patterns
├── .env.example            # Example environment variables
│
├── bayescal/                # Main package
│   ├── __init__.py
│   ├── config.py            # Configuration management
│   │
│   ├── data/                # Data handling
│   │   ├── __init__.py
│   │   ├── loaders.py       # Data loaders (CIFAR-10, CIFAR-100, toy dataset)
│   │   └── preprocessing.py # Data preprocessing utilities
│   │
│   ├── models/              # Neural network models
│   │   ├── __init__.py
│   │   ├── bayesian.py      # Bayes by Backprop implementation
│   │   ├── feedforward.py   # Feedforward NN with dropout
│   │   └── layers/          # Custom JAX layers
│   │       ├── __init__.py
│   │       ├── bayesiandense.py  # Bayesian Dense layer
│   │       └── bayesianconv2d.py  # Bayesian Conv2D layer
│   │
│   ├── training/            # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py       # Training loop
│   │   └── optimizers.py    # Optimizer configurations
│   │
│   ├── evaluation/          # Evaluation metrics
│   │   ├── __init__.py
│   │   ├── calibration.py   # ECE, Brier score, calibration curves
│   │   └── ood.py           # Out-of-distribution detection
│   │
│   ├── utils/               # Utility functions
│   │   ├── __init__.py
│   │   ├── logging.py       # Logging configuration
│   │   ├── visualization.py # Plotting utilities
│   │   └── toy_dataset.py  # Toy dataset analysis utilities
│   │
│   └── api/                 # FastAPI application (if needed)
│       ├── __init__.py
│       ├── main.py          # FastAPI app
│       └── endpoints.py     # API endpoints
│
├── scripts/                 # Executable scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   └── compare.py           # Comparison script
│
├── notebooks/               # Jupyter notebooks for exploration
│   └── toy.ipynb            # Toy dataset demonstration and analysis
│
├── tests/                   # Test suite
│   ├── __init__.py
│   ├── test_models.py
│   ├── test_training.py
│   ├── test_evaluation.py
│   └── test_data.py
│
├── experiments/             # Experiment configurations and results
│   ├── configs/             # Experiment configs (YAML/JSON)
│   └── results/             # Saved results and checkpoints
│
└── docs/                    # Documentation
    └── api.md               # API documentation
```

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Or using pip with pyproject.toml
pip install -e .
```

## Usage

```bash
# Train a model
python scripts/train.py --config experiments/configs/bayesian_config.yaml

# Evaluate calibration
python scripts/evaluate.py --model-path experiments/results/bayesian_model.pkl

# Compare models
python scripts/compare.py
```

## Features

- **Bayesian Neural Networks**: Implementation of Bayes by Backprop
- **Custom JAX Layers**: Registered custom layers for Bayesian inference
- **Calibration Metrics**: ECE, Brier score, and calibration curves
- **Comparison Framework**: Systematic comparison with feedforward dropout-based models

