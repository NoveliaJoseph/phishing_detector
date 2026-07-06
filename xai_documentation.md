# XAI (Explainable AI) Documentation for Phishing Detection

This document provides an overview of the `xai.py` module, which is responsible for adding explainability to the Hybrid BERT-BiLSTM phishing detection model using the [Captum](https://captum.ai/) library.

## Overview

The `xai.py` script leverages **Layer Integrated Gradients** from Captum to interpret the decisions made by the trained phishing detection model. It calculates an attribution score for each word (token) in a given text, showing how much that specific word contributed to the model's final prediction. 

By analyzing these attribution scores, you can understand which words trigger the model to classify an email or text as "Phishing" (positive scores) or "Legitimate" (negative scores).

## How It Works

1. **Tokenization**: The input text is tokenized using `BertTokenizer`. The tokenization includes adding special tokens like `[CLS]` and `[SEP]`, truncation, and padding to a maximum length of 128 tokens.
2. **Layer Integrated Gradients**: Captum's `LayerIntegratedGradients` calculates the gradient of the output prediction with respect to the BERT model's word embeddings (`model.bert.embeddings.word_embeddings`). 
3. **Attribution Scoring**: The attributions are computed relative to a baseline (a sequence of padding or zero tokens).
4. **Normalization & Filtering**: The resulting multi-dimensional attributions are summed over the embedding dimensions to get a single score per token. The scores are then normalized, and special tokens (`[PAD]`, `[CLS]`, `[SEP]`) are filtered out for a cleaner output.

## Code Structure

### `get_word_attributions(text, model, tokenizer, device)`

The core function that computes the attributions.

- **Parameters**:
  - `text` (str): The sample text to be analyzed.
  - `model` (torch.nn.Module): The loaded `Hybrid_BERT_BiLSTM` model.
  - `tokenizer`: The `BertTokenizer` used to preprocess the text.
  - `device` (torch.device): The device (CPU or CUDA) the model is running on.
- **Returns**: A list of tuples `[(token, attribution_score), ...]`.

### Main Execution Block

When `xai.py` is run directly, it executes a demonstration of the functionality:
1. Loads the tokenizer and model architecture.
2. Attempts to load trained weights (`model.pth`). If not found, it falls back to an untrained model state.
3. Analyzes a sample text: *"URGENT: Your bank account will be suspended. Click here to verify your identity."*
4. Prints the top 5 words contributing to the **Phishing** prediction and the top 5 contributing to the **Legitimate** prediction.

## Example Output

Running `python xai.py` will produce an output similar to the following:

```
Loading model and tokenizer...
Loaded trained model weights.

Analyzing Text: 'URGENT: Your bank account will be suspended. Click here to verify your identity.'

Calculating word attributions for 'Phishing' class...

Top 5 words contributing to PHISHING prediction:
urgent         : 0.3541
suspended      : 0.2810
verify         : 0.2234
account        : 0.1985
bank           : 0.1542

Top 5 words contributing to LEGITIMATE prediction (negative score):
identity       : -0.0543
your           : -0.0412
to             : -0.0211
be             : -0.0150
will           : -0.0090
```

## Prerequisites

To run this script, ensure you have the following installed:
- `torch`
- `transformers`
- `captum`

You should also have the trained weights `model.pth` in the same directory to get meaningful attributions.
