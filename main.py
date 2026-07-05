import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.metrics import precision_score, recall_score, f1_score
from datasets import load_dataset

# ---------------------------------------------------------
# 1. Data Loading and Processing (Hugging Face)
# ---------------------------------------------------------
def prepare_huggingface_data(tokenizer, batch_size=8):
    print("Downloading dataset from Hugging Face (SetFit/enron_spam)...")
    # This dataset contains Enron emails labeled as spam/ham (phishing/legitimate)
    dataset = load_dataset("SetFit/enron_spam")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=128
        )
        
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # We only need the tokenized tensors and the label
    tokenized_datasets.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
    
    # Using 'train' and 'test' splits provided by the dataset
    train_loader = DataLoader(tokenized_datasets["train"], shuffle=True, batch_size=batch_size)
    val_loader = DataLoader(tokenized_datasets["test"], batch_size=batch_size)
    
    return train_loader, val_loader

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
    print("Setting up device...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Prepare Data
    print("\nPreparing dataset...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_loader, val_loader = prepare_huggingface_data(tokenizer, batch_size=8)
    
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
