import argparse
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
import data_loader as dl

# ---------------------------------------------------------
# 1. Data Loading — delegated to data_loader.py
# ---------------------------------------------------------
# Use data_loader.get_dataloaders(tokenizer, batch_size, max_samples)
# See data_loader.py for full documentation.

# ---------------------------------------------------------
# 2. Hybrid BERT-BiLSTM Model Architecture
# ---------------------------------------------------------
class Hybrid_BERT_BiLSTM(nn.Module):
    def __init__(self, bert_model_name='bert-base-uncased', hidden_dim=128, num_classes=2, dropout=0.3):
        super(Hybrid_BERT_BiLSTM, self).__init__()
        
        print("Loading BERT model...")
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # Freeze BERT to speed up training for this demo
        for param in self.bert.parameters():
            param.requires_grad = False
            
        bert_output_dim = self.bert.config.hidden_size # 768
        
        # BiLSTM Layer
        self.bilstm = nn.LSTM(input_size=bert_output_dim, 
                              hidden_size=hidden_dim, 
                              num_layers=1, 
                              bidirectional=True, 
                              batch_first=True)
        
        self.dropout = nn.Dropout(dropout)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim * 2, num_classes)
        
    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state 
        
        lstm_out, (hidden, cell) = self.bilstm(sequence_output)
        hidden_cat = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        
        out = self.dropout(hidden_cat)
        logits = self.fc(out)
        
        return logits

# ---------------------------------------------------------
# 4. Training and Evaluation Loop
# ---------------------------------------------------------
def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch in data_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        targets = batch['label'].to(device)
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        _, preds = torch.max(outputs, dim=1)
        
        loss = loss_fn(outputs, targets)
        
        total_loss += loss.item()
        correct_predictions += torch.sum(preds == targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    return total_loss / len(data_loader), correct_predictions.double() / len(data_loader.dataset)

def evaluate(model, data_loader, loss_fn, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            targets = batch['label'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)
            
            total_loss += loss.item()
            correct_predictions += torch.sum(preds == targets)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
    avg_loss = total_loss / len(data_loader)
    acc = correct_predictions.double() / len(data_loader.dataset)
    precision = precision_score(all_targets, all_preds, zero_division=0)
    recall = recall_score(all_targets, all_preds, zero_division=0)
    f1 = f1_score(all_targets, all_preds, zero_division=0)
    
    return avg_loss, acc, precision, recall, f1

def main():
    # ------------------------------------------------------------------
    # CLI arguments for quick iteration without editing source code
    # Examples:
    #   python main.py                        (full dataset, 3 epochs)
    #   python main.py --max_samples 500      (quick dev run)
    #   python main.py --epochs 1 --batch 16
    # ------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Train Phishing Detector")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit samples per split (train/test). Useful for quick tests.")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs (default: 3).")
    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8).")
    parser.add_argument("--lr", type=float, default=2e-3,
                        help="Learning rate (default: 2e-3).")
    parser.add_argument("--output", type=str, default="model.pth",
                        help="Path to save trained model weights (default: model.pth).")
    args = parser.parse_args()

    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data (via data_loader.py)
    print("\nPreparing dataset...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = dl.get_dataloaders(
        tokenizer,
        batch_size=args.batch,
        max_samples=args.max_samples,
    )

    # 2. Initialize Model
    print("\nInitializing model...")
    model = Hybrid_BERT_BiLSTM().to(device)

    # 3. Setup Optimizer and Loss Function
    # Higher LR is appropriate because BERT layers are frozen
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = CrossEntropyLoss().to(device)

    # 4. Train Model
    print(f"\nStarting training loop ({args.epochs} epoch(s))...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, loss_fn, device)

        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} "
              f"| Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f}")

    # 5. Save Model
    print(f"\nSaving model to {args.output}...")
    torch.save(model.state_dict(), args.output)
    print("Training complete!")


if __name__ == '__main__':
    main()
