import torch
from transformers import BertTokenizer
from captum.attr import LayerIntegratedGradients
import main

def get_word_attributions(text, model, tokenizer, device):
    model.eval()
    
    # Tokenize input
    encoding = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=128,
        return_token_type_ids=False,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    # The forward func for captum needs to take the inputs and return logits
    def forward_func(inputs, attn_mask):
        return model(inputs, attn_mask)
        
    # We want to explain with respect to the input embeddings
    # In bert-base-uncased, it's model.bert.embeddings.word_embeddings
    lig = LayerIntegratedGradients(forward_func, model.bert.embeddings.word_embeddings)
    
    # target=1 means we are explaining why it might be predicted as phishing (Class 1)
    # The reference will be a tensor of pad tokens (or zero)
    ref_input_ids = torch.zeros_like(input_ids).to(device)
    
    attributions, delta = lig.attribute(inputs=input_ids,
                                        baselines=ref_input_ids,
                                        additional_forward_args=(attention_mask,),
                                        target=1,
                                        return_convergence_delta=True)
                                        
    # Sum over the embedding dimensions to get a single attribution score per token
    attributions = attributions.sum(dim=-1).squeeze(0)
    
    # Normalize
    if torch.norm(attributions) > 0:
        attributions = attributions / torch.norm(attributions)
    
    # Map tokens back to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].cpu().numpy())
    
    # Filter out padding tokens for cleaner output
    results = []
    for token, attr in zip(tokens, attributions.cpu().detach().numpy()):
        if token not in ['[PAD]', '[CLS]', '[SEP]']:
            results.append((token, attr))
            
    return results

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("Loading model and tokenizer...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = main.Hybrid_BERT_BiLSTM().to(device)
    
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device, weights_only=True))
        print("Loaded trained model weights.")
    except FileNotFoundError:
        print("Warning: model.pth not found. Using untrained model weights for demonstration.")
        
    sample_text = "URGENT: Your bank account will be suspended. Click here to verify your identity."
    print(f"\nAnalyzing Text: '{sample_text}'")
    
    print("\nCalculating word attributions for 'Phishing' class...")
    attributions = get_word_attributions(sample_text, model, tokenizer, device)
    
    print("\nTop 5 words contributing to PHISHING prediction:")
    # Sort by attribution value descending
    attributions.sort(key=lambda x: x[1], reverse=True)
    
    for token, score in attributions[:5]:
        print(f"{token:15}: {score:.4f}")
        
    print("\nTop 5 words contributing to LEGITIMATE prediction (negative score):")
    # Sort by attribution value ascending
    attributions.sort(key=lambda x: x[1], reverse=False)
    for token, score in attributions[:5]:
        print(f"{token:15}: {score:.4f}")
