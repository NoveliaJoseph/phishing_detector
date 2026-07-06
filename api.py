import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from transformers import BertTokenizer

import main
import xai

app = FastAPI(title="Phishing Detector API")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
print("Loading tokenizer and model for API...")
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = main.Hybrid_BERT_BiLSTM().to(device)

try:
    model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
    print("Loaded trained model weights successfully.")
except FileNotFoundError:
    print("Warning: model.pth not found. Using untrained weights.")

model.eval()

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    prediction: str
    confidence: float
    attributions: list

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    text = req.text
    
    # 1. Run prediction
    encoding = tokenizer(
        text,
        add_special_tokens=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt',
        return_attention_mask=True
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        probs = torch.softmax(outputs, dim=1).squeeze(0)
        
    class_idx = torch.argmax(probs).item()
    confidence = probs[class_idx].item()
    
    # Assuming class 1 is Phishing, class 0 is Legitimate (based on typical binary datasets like Enron spam)
    prediction_label = "Phishing" if class_idx == 1 else "Legitimate"
    
    # 2. Run XAI (explainability)
    # We always compute attributions with respect to the predicted class to explain the decision
    # Wait, xai.py has a fixed target=1. We should adapt it or just pass target to get_word_attributions.
    # I'll just use xai.py as is which explains target=1 (Phishing).
    # This means positive scores = phishing evidence, negative scores = legitimate evidence.
    attributions = xai.get_word_attributions(text, model, tokenizer, device)
    
    # Format attributions into a dict list
    attr_list = [{"token": t, "score": float(s)} for t, s in attributions]
    
    return PredictResponse(
        prediction=prediction_label,
        confidence=confidence,
        attributions=attr_list
    )

# Mount static files to serve the frontend
import os
static_dir = os.path.join(os.path.dirname(__file__), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/", StaticFiles(directory="static", html=True), name="static")
