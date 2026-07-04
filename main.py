import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
import os

# ---------------------------------------------------------
# 1. Data Loading and Processing
# ---------------------------------------------------------
def load_data(csv_path='dataset.csv'):
    if os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        return pd.read_csv(csv_path)
    
    print(f"{csv_path} not found. Generating dummy dataset...")
    # 0 = Legitimate (Ham), 1 = Phishing
    data = {
        'email_text': [
            "Hey John, are we still on for the meeting tomorrow at 10 AM?",
            "URGENT: Your bank account will be suspended. Click here to verify your identity.",
            "Please find attached the Q3 financial report for your review.",
            "Congratulations! You have won a $1000 Walmart gift card. Claim it now before it expires!",
            "Can you pick up some milk on your way home?",
            "Action Required: Update your PayPal billing information immediately to avoid service interruption."
        ] * 20,
        'label': [0, 1, 0, 1, 0, 1] * 20
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_path, index=False)
    return df

# ---------------------------------------------------------
# 2. Dataset and Tokenizer Setup
# ---------------------------------------------------------
class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long)
        }

# ---------------------------------------------------------
# 3. Hybrid BERT-BiLSTM Model Architecture
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
        targets = batch['targets'].to(device)
        
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
            targets = batch['targets'].to(device)
            
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
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("\nPreparing dataset...")
    df = load_data()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Train/Val Split
    df_train, df_val = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    
    train_dataset = EmailDataset(df_train['email_text'].to_numpy(), df_train['label'].to_numpy(), tokenizer)
    val_dataset = EmailDataset(df_val['email_text'].to_numpy(), df_val['label'].to_numpy(), tokenizer)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    # 2. Initialize Model
    print("\nInitializing model...")
    model = Hybrid_BERT_BiLSTM().to(device)
    
    # 3. Setup Optimizer and Loss Function
    optimizer = AdamW(model.parameters(), lr=2e-3) # Higher LR because BERT is frozen
    loss_fn = CrossEntropyLoss().to(device)
    
    # 4. Train Model
    epochs = 3
    print("\nStarting training loop...")
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, device)
        val_loss, val_acc, val_p, val_r, val_f1 = evaluate(model, val_loader, loss_fn, device)
        
        print(f"Train - Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"Val   - Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Precision: {val_p:.4f} | Recall: {val_r:.4f} | F1: {val_f1:.4f}")
        
    # 5. Save Model
    print("\nSaving model to model.pth...")
    torch.save(model.state_dict(), 'model.pth')
    print("Training complete!")

if __name__ == '__main__':
    main()
