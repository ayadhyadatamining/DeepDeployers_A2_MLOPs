import os
import pandas as pd
import torch
from transformers import BartForSequenceClassification, Trainer, TrainingArguments,BartTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split



class Prediction:
    def __init__(self):
        pass

    def predict_sentiment(self,text,tokenizer,model,device):
        """Predict sentiment (Positive/Negative) for a given text review."""
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        
        with torch.no_grad():  # No need to calculate gradients
            outputs = model(**inputs)
        
        prediction = torch.argmax(outputs.logits, dim=1).item()
        sentiment = "positive" if prediction == 1 else "negative"
        
        return  sentiment