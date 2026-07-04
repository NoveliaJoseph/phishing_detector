import optuna
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import main

def objective(trial):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-5, 5e-3, log=True)
    dropout = trial.suggest_float('dropout', 0.1, 0.6)
    hidden_dim = trial.suggest_categorical('hidden_dim', [64, 128, 256])
    
    # Prepare Data
    df = main.load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = main.EmailDataset(df_train['email_text'].to_numpy(), df_train['label'].to_numpy(), tokenizer)
    val_dataset = main.EmailDataset(df_val['email_text'].to_numpy(), df_val['label'].to_numpy(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # Initialize Model
    model = main.Hybrid_BERT_BiLSTM(hidden_dim=hidden_dim, dropout=dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    loss_fn = CrossEntropyLoss().to(device)
    
    # Train and Evaluate
    epochs = 2 # Keep short for tuning demo
    best_f1 = 0
    for epoch in range(epochs):
        main.train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1 = main.evaluate(model, val_loader, loss_fn, device)
        
        # Report intermediate objective value
        trial.report(val_f1, epoch)
        
        # Handle pruning
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_f1 = max(best_f1, val_f1)
        
    return best_f1

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    print("Starting hyperparameter tuning...")
    study.optimize(objective, n_trials=3) # 3 trials for quick demo
    
    print("\nNumber of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    
    print("  Value (Validation F1): ", trial.value)
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
