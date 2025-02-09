import pandas as pd

import os
import pandas as pd
import torch
from transformers import BartTokenizer
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split





class Preprocessing:
    def __init__(self):
        pass

    def create_sentiment_dataframe(self,positive_reviews, negative_reviews):

        data = {
            "review": positive_reviews + negative_reviews,
            "sentiment": ["positive"] * len(positive_reviews) + ["negative"] * len(negative_reviews)
        }
        return pd.DataFrame(data)




    def preprocess_data(self,df):

        df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
        
        # Ensure equal distribution of positive and negative samples in train and test sets
        df_positive = df[df["label"] == 1]
        df_negative = df[df["label"] == 0]
        
        train_pos, val_pos = train_test_split(df_positive, test_size=0.2, random_state=42)
        train_neg, val_neg = train_test_split(df_negative, test_size=0.2, random_state=42)
        
        train_df = pd.concat([train_pos, train_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        val_df = pd.concat([val_pos, val_neg]).sample(frac=1, random_state=42).reset_index(drop=True)
        
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        
        train_encodings = tokenizer(train_df["review"].tolist(), truncation=True, padding=True, max_length=512)
        val_encodings = tokenizer(val_df["review"].tolist(), truncation=True, padding=True, max_length=512)
        
        train_dataset = Dataset.from_dict({"input_ids": train_encodings["input_ids"], "attention_mask": train_encodings["attention_mask"], "labels": train_df["label"].tolist()})
        val_dataset = Dataset.from_dict({"input_ids": val_encodings["input_ids"], "attention_mask": val_encodings["attention_mask"], "labels": val_df["label"].tolist()})
        
        return DatasetDict({"train": train_dataset, "test": val_dataset})