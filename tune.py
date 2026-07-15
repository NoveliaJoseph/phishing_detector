"""
tune.py
-------
Hyperparameter tuning for the phishing detector using Optuna.

Key optimization: dataset is loaded ONCE before the study begins,
then reused across all trials — avoids expensive re-downloading
and re-tokenizing on every Optuna trial.

Usage:
    python tune.py                        # 3 trials, full dataset
    python tune.py --max_samples 500      # quick dev run
    python tune.py --n_trials 10          # more thorough search
"""

import argparse
import optuna
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import BertTokenizer
import main
import data_loader as dl


def make_objective(train_loader, val_loader):
    """
    Returns an Optuna objective function with the data loaders baked in.
    This pattern avoids reloading data on every trial.
    """
    def objective(trial):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Hyperparameters to tune
        lr         = trial.suggest_float("lr", 1e-5, 5e-3, log=True)
        dropout    = trial.suggest_float("dropout", 0.1, 0.6)
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])

        # Initialize Model with trial hyperparams
        model     = main.Hybrid_BERT_BiLSTM(hidden_dim=hidden_dim, dropout=dropout).to(device)
        optimizer = AdamW(model.parameters(), lr=lr)
        loss_fn   = CrossEntropyLoss().to(device)

        # Train and evaluate (short epochs for tuning speed)
        best_f1 = 0.0
        for epoch in range(2):
            main.train_epoch(model, train_loader, loss_fn, optimizer, device)
            val_loss, val_acc, val_p, val_r, val_f1 = main.evaluate(
                model, val_loader, loss_fn, device
            )

            # Report intermediate value for pruning
            trial.report(val_f1, epoch)

            # Prune unpromising trials early
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            best_f1 = max(best_f1, val_f1)

        return best_f1

    return objective


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Phishing Detector")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split for fast iteration (default: use all data).")
    parser.add_argument("--n_trials", type=int, default=3,
                        help="Number of Optuna trials (default: 3).")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size for data loading (default: 8).")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load dataset ONCE — reused across all trials
    # ------------------------------------------------------------------
    print("Loading tokenizer and dataset (shared across all trials)...")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    train_loader, val_loader = dl.get_dataloaders(
        tokenizer,
        batch_size=args.batch,
        max_samples=args.max_samples,
    )
    print("Dataset loaded. Starting Optuna study...\n")

    # ------------------------------------------------------------------
    # Run Optuna study
    # ------------------------------------------------------------------
    study = optuna.create_study(
        direction="maximize",
        pruner=optuna.pruners.MedianPruner(),
    )
    study.optimize(make_objective(train_loader, val_loader), n_trials=args.n_trials)

    # ------------------------------------------------------------------
    # Report results
    # ------------------------------------------------------------------
    print(f"\nFinished {len(study.trials)} trial(s).")
    print("Best trial:")
    best = study.best_trial
    print(f"  Validation F1 : {best.value:.4f}")
    print("  Hyperparameters:")
    for key, value in best.params.items():
        print(f"    {key}: {value}")
