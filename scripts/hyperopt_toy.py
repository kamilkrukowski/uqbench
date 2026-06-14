"""Hyperparameter optimization for toy dataset using Optuna with TPE sampler.

Install Optuna: pip install optuna
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from sklearn.model_selection import train_test_split

try:
    import optuna
except ImportError:
    raise ImportError(
        "Optuna is required for hyperparameter optimization. "
        "Install it with: pip install optuna"
    )

from bayescal.models import BayesianFNN, DropoutFNN, FNN
from bayescal.training import optimizers, trainer
from bayescal.evaluation import calibration


def generate_toy_dataset(n_samples=2000, overlap=0.5, seed=42):
    """Generate 2D toy dataset with two overlapping Gaussian classes."""
    np.random.seed(seed)
    n_per_class = n_samples // 2
    
    # Class 0: centered at (-1, -1)
    mean_0 = np.array([-1.0, -1.0])
    cov_0 = np.array([[1.0, 0.0], [0.0, 1.0]])
    X_0 = np.random.multivariate_normal(mean_0, cov_0, n_per_class)
    y_0 = np.zeros(n_per_class, dtype=int)
    
    # Class 1: centered at (1, 1) with controlled overlap
    mean_1 = np.array([1.0, 1.0])
    cov_1 = np.array([[1.0 + overlap, 0.0], [0.0, 1.0 + overlap]])
    X_1 = np.random.multivariate_normal(mean_1, cov_1, n_per_class)
    y_1 = np.ones(n_per_class, dtype=int)
    
    # Combine
    X = np.vstack([X_0, X_1])
    y = np.hstack([y_0, y_1])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # One-hot encode
    y_onehot = np.zeros((len(y), 2))
    y_onehot[np.arange(len(y)), y] = 1
    
    return X, y, y_onehot


def train_and_evaluate_model(
    model,
    X_train,
    y_train_onehot,
    X_val,
    y_val_onehot,
    epochs=200,
    batch_size=64,
    lr=0.01,
    max_grad_norm=1.0,
    seed=42,
):
    """Train a model and return validation metrics."""
    rng = jax.random.PRNGKey(seed)
    
    # Initialize model
    input_shape = (X_train.shape[1],)
    rng, init_rng = jax.random.split(rng)
    params = model.init_params(init_rng, input_shape=input_shape)
    
    # Create optimizer with gradient clipping
    optimizer = optimizers.get_optimizer(
        learning_rate=lr, optimizer_type="adam", max_grad_norm=max_grad_norm
    )
    opt_state = optimizer.init(params)
    
    # Create batches
    n_batches = (len(X_train) + batch_size - 1) // batch_size
    batches = []
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        batch_X = jnp.array(X_train[start_idx:end_idx])
        batch_y = jnp.array(y_train_onehot[start_idx:end_idx])
        batches.append((batch_X, batch_y))
    
    # Training loop
    rng, train_rng = jax.random.split(rng)
    best_val_loss = float('inf')
    patience = 50
    patience_counter = 0
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_X, batch_y in batches:
            rng, batch_rng = jax.random.split(rng)
            
            # Determine n_vi_samples for Bayesian models
            is_bayesian = hasattr(model, "compute_kl_divergence") and hasattr(model, "beta")
            n_vi_samples = 1 if is_bayesian else 1
            beta = getattr(model, "beta", 1.0) if is_bayesian else 1.0
            
            params, opt_state, metrics = trainer.train_step(
                model, params, opt_state, (batch_X, batch_y), batch_rng, 
                optimizer, beta=beta, n_vi_samples=n_vi_samples
            )
            
            epoch_loss += metrics['loss']
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        
        # Early stopping on validation set
        if (epoch + 1) % 10 == 0:
            rng, eval_rng = jax.random.split(rng)
            val_loss, _ = model.get_loss(
                params, inputs=jnp.array(X_val), labels=jnp.array(y_val_onehot), 
                rng=eval_rng, n_vi_samples=1
            )
            val_loss = float(val_loss)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience // 10:
                    break
    
    # Final evaluation on validation set
    rng, eval_rng = jax.random.split(rng)
    val_probs = model.apply(
        params, inputs=jnp.array(X_val), rng=eval_rng, training=False, 
        n_samples=100 if is_bayesian else 1
    )
    
    # Compute metrics
    val_accuracy = (jnp.argmax(val_probs, axis=-1) == jnp.argmax(y_val_onehot, axis=-1)).mean()
    val_ece = calibration.expected_calibration_error(val_probs, y_val_onehot)
    val_mce = calibration.maximum_calibration_error(val_probs, y_val_onehot)
    val_brier = calibration.brier_score(val_probs, y_val_onehot)
    
    return {
        'accuracy': float(val_accuracy),
        'ece': float(val_ece),
        'mce': float(val_mce),
        'brier': float(val_brier),
        'loss': best_val_loss,
    }


def objective_fnn(trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=200):
    """Objective function for FNN hyperparameter optimization."""
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    
    # Create model
    model = FNN(hidden_dims=(64, 32, 32, 32, 32), num_classes=2, dropout_rate=0.0)
    
    # Train and evaluate
    metrics = train_and_evaluate_model(
        model, X_train, y_train_onehot, X_val, y_val_onehot,
        epochs=epochs, lr=lr
    )
    
    # Minimize: MCE + ECE - 0.1*accuracy
    # Lower is better (better calibration and accuracy)
    score = metrics['mce'] + metrics['ece'] - 0.1 * metrics['accuracy']
    
    trial.set_user_attr('accuracy', metrics['accuracy'])
    trial.set_user_attr('ece', metrics['ece'])
    trial.set_user_attr('mce', metrics['mce'])
    trial.set_user_attr('brier', metrics['brier'])
    
    return score


def objective_dropout_fnn(trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=200):
    """Objective function for DropoutFNN hyperparameter optimization."""
    # Hyperparameters
    lr = trial.suggest_float('lr', 1e-4, 1e-1, log=True)
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    
    # Create model
    model = DropoutFNN(hidden_dims=(64, 32, 32, 32, 32), num_classes=2, dropout_rate=dropout_rate)
    
    # Train and evaluate
    metrics = train_and_evaluate_model(
        model, X_train, y_train_onehot, X_val, y_val_onehot,
        epochs=epochs, lr=lr
    )
    
    # Optimize for low ECE while maintaining good accuracy
    score = metrics['accuracy'] - metrics['ece']
    
    trial.set_user_attr('accuracy', metrics['accuracy'])
    trial.set_user_attr('ece', metrics['ece'])
    trial.set_user_attr('brier', metrics['brier'])
    trial.set_user_attr('dropout_rate', dropout_rate)
    
    return score


def objective_bayesian_fnn(trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=200):
    """Objective function for BayesianFNN hyperparameter optimization."""
    # Hyperparameters - all three must be suggested for Optuna to track them
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    beta = trial.suggest_float('beta', 1e-6, 1e-3, log=True)
    posterior_std_init = trial.suggest_float('posterior_std_init', 1.0e-2, 2.0e-2, log=True)
    
    # Debug: Verify all parameters are suggested
    assert 'lr' in trial.params, "lr not in trial.params"
    assert 'beta' in trial.params, "beta not in trial.params"
    assert 'posterior_std_init' in trial.params, "posterior_std_init not in trial.params"
    
    # Create model
    model = BayesianFNN(
        hidden_dims=(64, 32, 32, 32, 32),
        num_classes=2,
        beta=beta,
        posterior_std_init=posterior_std_init,
        prior_std=1.0
    )
    
    # Train and evaluate
    metrics = train_and_evaluate_model(
        model, X_train, y_train_onehot, X_val, y_val_onehot,
        epochs=epochs, lr=lr
    )
    
    # Minimize: MCE + ECE - 0.1*accuracy
    # Lower is better (better calibration and accuracy)
    score = metrics['mce'] + metrics['ece'] - 0.1 * metrics['accuracy']
    
    trial.set_user_attr('accuracy', metrics['accuracy'])
    trial.set_user_attr('ece', metrics['ece'])
    trial.set_user_attr('mce', metrics['mce'])
    trial.set_user_attr('brier', metrics['brier'])
    trial.set_user_attr('beta', beta)
    trial.set_user_attr('posterior_std_init', posterior_std_init)
    
    return score


def main():
    parser = argparse.ArgumentParser(description="Hyperparameter optimization for toy dataset")
    parser.add_argument(
        '--model-type',
        type=str,
        choices=['fnn', 'dropout_fnn', 'bayesian_fnn', 'all'],
        default='all',
        help='Model type to optimize'
    )
    parser.add_argument(
        '--n-trials',
        type=int,
        default=50,
        help='Number of optimization trials'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=200,
        help='Number of training epochs per trial'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path('experiments/hyperopt'),
        help='Output directory for results'
    )
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    
    # Generate dataset
    print("Generating toy dataset...")
    X, y, y_onehot = generate_toy_dataset(n_samples=2000, overlap=0.8, seed=args.seed)
    
    # Split into train/val/test
    X_train, X_temp, y_train, y_temp, y_train_onehot, y_temp_onehot = train_test_split(
        X, y, y_onehot, test_size=0.4, random_state=args.seed, stratify=y
    )
    X_val, X_test, y_val, y_test, y_val_onehot, y_test_onehot = train_test_split(
        X_temp, y_temp, y_temp_onehot, test_size=0.5, random_state=args.seed, stratify=y_temp
    )
    
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    model_types = []
    if args.model_type == 'all':
        model_types = ['fnn', 'dropout_fnn', 'bayesian_fnn']
    else:
        model_types = [args.model_type]
    
    results = {}
    
    for model_type in model_types:
        print(f"\n{'='*60}")
        print(f"Optimizing {model_type.upper()}")
        print(f"{'='*60}")
        
        # Create study with TPE sampler
        study = optuna.create_study(
            direction='minimize',  # We want to minimize (MCE + ECE - 0.1*accuracy)
            sampler=optuna.samplers.TPESampler(seed=args.seed),
            study_name=f'{model_type}_toy_hyperopt'
        )
        
        # Select objective function
        if model_type == 'fnn':
            objective = lambda trial: objective_fnn(
                trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=args.epochs
            )
        elif model_type == 'dropout_fnn':
            objective = lambda trial: objective_dropout_fnn(
                trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=args.epochs
            )
        elif model_type == 'bayesian_fnn':
            objective = lambda trial: objective_bayesian_fnn(
                trial, X_train, y_train_onehot, X_val, y_val_onehot, epochs=args.epochs
            )
        
        # Optimize
        study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
        
        # Debug: Print all trials to verify parameters (for BayesianFNN)
        if model_type == 'bayesian_fnn' and len(study.trials) > 0:
            print(f"\nDebug: Checking first 3 trials for BayesianFNN:")
            for i, trial in enumerate(study.trials[:min(3, len(study.trials))]):
                print(f"  Trial {i} params: {trial.params}")
        
        # Get best trial
        best_trial = study.best_trial
        print(f"\nBest trial for {model_type}:")
        print(f"  Score (MCE + ECE - 0.1*Acc): {best_trial.value:.4f}")
        print(f"  Params:")
        for param_name, param_value in best_trial.params.items():
            if isinstance(param_value, float):
                print(f"    {param_name}: {param_value:.6f}")
            else:
                print(f"    {param_name}: {param_value}")
        print(f"  Accuracy: {best_trial.user_attrs.get('accuracy', 'N/A'):.4f}")
        print(f"  ECE: {best_trial.user_attrs.get('ece', 'N/A'):.4f}")
        print(f"  MCE: {best_trial.user_attrs.get('mce', 'N/A'):.4f}")
        print(f"  Brier: {best_trial.user_attrs.get('brier', 'N/A'):.4f}")
        
        # Save results
        results[model_type] = {
            'best_params': best_trial.params,
            'best_score': best_trial.value,
            'metrics': {
                'accuracy': best_trial.user_attrs.get('accuracy'),
                'ece': best_trial.user_attrs.get('ece'),
                'mce': best_trial.user_attrs.get('mce'),
                'brier': best_trial.user_attrs.get('brier'),
            },
            'user_attrs': dict(best_trial.user_attrs),
        }
        
        # Save study
        study_path = args.output_dir / f'{model_type}_study.json'
        with open(study_path, 'w') as f:
            json.dump({
                'best_params': best_trial.params,
                'best_value': best_trial.value,
                'user_attrs': dict(best_trial.user_attrs),
            }, f, indent=2)
    
    # Save summary
    summary_path = args.output_dir / 'hyperopt_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Hyperparameter optimization complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"{'='*60}")
    
    # Print summary
    print("\nSummary of best hyperparameters:")
    for model_type, result in results.items():
        print(f"\n{model_type.upper()}:")
        print(f"  Params: {result['best_params']}")
        print(f"  Score (MCE + ECE - 0.1*Acc): {result['best_score']:.4f}")
        print(f"  Accuracy: {result['metrics']['accuracy']:.4f}")
        print(f"  ECE: {result['metrics']['ece']:.4f}")
        print(f"  MCE: {result['metrics']['mce']:.4f}")
        print(f"  Brier: {result['metrics']['brier']:.4f}")


if __name__ == '__main__':
    main()

