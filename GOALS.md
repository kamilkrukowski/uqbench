
Make clean project to show ML Research Scientist experience

## Current Focus: Toy Dataset Analysis

**Primary Notebook**: `notebooks/train_analyze_posterior.ipynb`

### Objectives
- Demonstrate **calibration** and **predictive uncertainty quantification** on a controlled 2D toy dataset
- Compare five uncertainty quantification methods:

  **Training-time Methods:**
  - **FNN**: Traditional feedforward network (deterministic baseline)
  - **DropoutFNN**: FNN with Monte Carlo Dropout for uncertainty estimation
  - **BayesianFNN**: Bayesian FNN using variational inference (Bayes by Backprop)

  **Post-hoc Methods:**
  - **LaplaceFNN**: Laplace approximation fitted on trained FNN (last-layer)
  - **MCMCFNN**: MCMC sampling using NUTS for exact Bayesian inference

### Key Features
- **2D Toy Dataset**: Overlapping Gaussian distributions for binary classification
  - Allows visualization of decision boundaries in 2D space
  - Comparison against analytical Bayes optimal boundary
  - Controlled difficulty via overlap parameter

- **Calibration Analysis**:
  - ECE (Expected Calibration Error)
  - MCE (Maximum Calibration Error)
  - Brier Score
  - Calibration curves comparison

- **Performance Metrics**:
  - Training time (wall-clock time)
  - Inference time (per sample, with/without MC sampling)
  - Accuracy
  - Computational cost trade-offs between methods

- **Visualizations**:
  - Predictive posterior distributions (decision boundaries)
  - Uncertainty quantification (entropy-based)
  - Calibration curves
  - Comparison to analytical Bayes optimal boundary

### Technical Stack
- **JAX/Flax**: Neural network implementation
- **Custom Bayesian Layers**: Registered custom layers for Bayesian inference
- **BlackJAX**: MCMC sampling (HMC/NUTS) for exact Bayesian inference
- **Orbax**: Model checkpointing and export
- **Python**: Core implementation

### Future Goals
- Extend to CIFAR-10/100 for OOD detection
- FastAPI integration (where applicable)