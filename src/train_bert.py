import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from transformers import get_scheduler
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os

class TweetDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTModel:
    def __init__(self, num_labels=3):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=num_labels
        ).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        self.reverse_label_map = {v: k for k, v in self.label_map.items()}
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df (pandas.DataFrame): Processed DataFrame
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        # Convert labels to numeric values
        df['label'] = df['sentiment'].map(self.label_map)
        
        # Split data
        train_df, temp_df = train_test_split(df, test_size=0.3, random_state=42)
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
        
        # Create datasets
        train_dataset = TweetDataset(
            train_df['cleaned_text'].values,
            train_df['label'].values,
            self.tokenizer
        )
        
        val_dataset = TweetDataset(
            val_df['cleaned_text'].values,
            val_df['label'].values,
            self.tokenizer
        )
        
        test_dataset = TweetDataset(
            test_df['cleaned_text'].values,
            test_df['label'].values,
            self.tokenizer
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def train(self, train_dataset, val_dataset, batch_size=16, epochs=3):
        """
        Train the model
        
        Args:
            train_dataset (TweetDataset): Training dataset
            val_dataset (TweetDataset): Validation dataset
            batch_size (int): Batch size
            epochs (int): Number of epochs
        """
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size)
        
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-5)
        
        num_training_steps = len(train_loader) * epochs
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=num_training_steps
        )
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch + 1}/{epochs}")
            
            # Training
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item()
                
                loss.backward()
                optimizer.step()
                lr_scheduler.step()
            
            avg_train_loss = train_loss / len(train_loader)
            print(f"Training loss: {avg_train_loss:.4f}")
            
            # Validation
            self.model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in tqdm(val_loader, desc="Validation"):
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    val_loss += outputs.loss.item()
                    
                    _, predicted = torch.max(outputs.logits, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            avg_val_loss = val_loss / len(val_loader)
            val_accuracy = correct / total
            print(f"Validation loss: {avg_val_loss:.4f}")
            print(f"Validation accuracy: {val_accuracy:.4f}")
    
    def save_model(self, model_dir='../models'):
        """
        Save the trained model and tokenizer
        
        Args:
            model_dir (str): Directory to save the models
        """
        os.makedirs(model_dir, exist_ok=True)
        
        self.model.save_pretrained(os.path.join(model_dir, 'bert_model'))
        self.tokenizer.save_pretrained(os.path.join(model_dir, 'bert_tokenizer'))
        
        print(f"Model and tokenizer saved to {model_dir}")

def main():
    # Load processed data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(base_dir, 'data', 'processed_tweets.csv')
    df = pd.read_csv(processed_file)
    
    # Initialize model
    bert_model = BERTModel()
    
    # Prepare data
    train_dataset, val_dataset, test_dataset = bert_model.prepare_data(df)
    
    # Train model
    print("Training BERT model...")
    bert_model.train(train_dataset, val_dataset)
    
    # Save model
    bert_model.save_model()

if __name__ == "__main__":
    main() 