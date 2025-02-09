import mlflow
import mlflow.pytorch

import os
import pandas as pd
import torch
from transformers import BartForSequenceClassification, Trainer, TrainingArguments,BartTokenizer


class Model_train:
    def __init__(self):
        pass
    def train_model(self,dataset):
        """
        Fine-tunes the BART model for sentiment analysis with MLflow tracking.
        """
        mlflow.set_experiment("BART Sentiment Analysis")
        with mlflow.start_run():
            model_path = "model/pretrained_bart"
            model = BartForSequenceClassification.from_pretrained(model_path, num_labels=2)
            
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            
            training_args = TrainingArguments(
                output_dir="./results",
                evaluation_strategy="epoch",
                save_strategy="no",  # Disabling frequent saving to reduce GPU load
                per_device_train_batch_size=2,  # Reduced from 8 to 4
                per_device_eval_batch_size=2,   # Reduced batch size
                num_train_epochs=3,
                weight_decay=0.01,
                logging_dir="./logs",
                logging_steps=50,  # Reduce logging frequency
                fp16=True,  # Enable mixed precision training
                gradient_accumulation_steps=2,  # Helps simulate larger batches
            )
            
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=dataset["train"],
                eval_dataset=dataset["test"]
            )
            
            mlflow.log_params({
                "batch_size": training_args.per_device_train_batch_size,
                "epochs": training_args.num_train_epochs,
                "learning_rate": training_args.learning_rate,
                "weight_decay": training_args.weight_decay
            })
            
            trainer.train()
            
            model.save_pretrained("./fine_tuned_bart")
            tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
            tokenizer.save_pretrained("./fine_tuned_bart")
            
            mlflow.pytorch.log_model(model, "bart_model")
            print("Fine-tuning completed and model saved with MLflow tracking.")