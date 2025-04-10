import time
import pandas as pd
from datetime import datetime
import os
import sys
import joblib
import torch
from transformers import BertTokenizer, BertForSequenceClassification

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.twitter_api import TwitterAPI
from src.preprocess import TweetPreprocessor
from src.train_baseline import BaselineModel
from src.train_bert import BERTModel

class SentimentMonitor:
    def __init__(self, query, interval_minutes=5):
        self.query = query
        self.interval = interval_minutes * 60  # Convert to seconds
        self.twitter_api = TwitterAPI()
        self.preprocessor = TweetPreprocessor()
        
        # Load models
        self.baseline_model = self._load_baseline_model()
        self.bert_model = self._load_bert_model()
        
        # Initialize results storage
        self.results = []
    
    def _load_baseline_model(self):
        """Load the baseline model and vectorizer"""
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_dir = os.path.join(base_dir, 'models')
        
        vectorizer = joblib.load(os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        model = joblib.load(os.path.join(model_dir, 'baseline_model.joblib'))
        
        baseline_model = BaselineModel()
        baseline_model.vectorizer = vectorizer
        baseline_model.model = model
        
        return baseline_model
    
    def _load_bert_model(self):
        """Load the BERT model and tokenizer"""
        bert_model = BERTModel()
        bert_model.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased',
            num_labels=3
        )
        bert_model.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        return bert_model
    
    def process_tweets(self, tweets):
        """Process and analyze tweets"""
        if not tweets:
            return []
        
        # Convert to DataFrame
        df = pd.DataFrame(tweets)
        
        # Preprocess tweets
        df = self.preprocessor.preprocess_tweets(df)
        
        # Get baseline predictions
        X = self.baseline_model.vectorizer.transform(df['cleaned_text'])
        baseline_preds = self.baseline_model.model.predict(X)
        
        # Get BERT predictions
        bert_preds = []
        for text in df['cleaned_text']:
            encoding = self.bert_model.tokenizer(
                text,
                add_special_tokens=True,
                max_length=128,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            with torch.no_grad():
                outputs = self.bert_model.model(
                    input_ids=encoding['input_ids'],
                    attention_mask=encoding['attention_mask']
                )
                pred = torch.argmax(outputs.logits, dim=1).item()
                bert_preds.append(self.bert_model.reverse_label_map[pred])
        
        # Combine results
        results = []
        for i, tweet in enumerate(tweets):
            results.append({
                'id': tweet['id'],
                'text': tweet['text'],
                'created_at': tweet['created_at'],
                'baseline_sentiment': baseline_preds[i],
                'bert_sentiment': bert_preds[i]
            })
        
        return results
    
    def monitor(self):
        """Monitor tweets in real-time"""
        print(f"Starting sentiment monitoring for query: {self.query}")
        print("Press Ctrl+C to stop")
        
        try:
            while True:
                # Fetch new tweets
                tweets = self.twitter_api.fetch_tweets(self.query, max_results=100)
                
                if tweets:
                    # Process and analyze tweets
                    results = self.process_tweets(tweets)
                    self.results.extend(results)
                    
                    # Print results
                    print(f"\n{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    print(f"Processed {len(results)} new tweets")
                    
                    # Save results
                    self.save_results()
                
                # Wait for the next interval
                time.sleep(self.interval)
        
        except KeyboardInterrupt:
            print("\nStopping sentiment monitoring...")
            self.save_results()
    
    def save_results(self):
        """Save monitoring results to CSV"""
        if not self.results:
            return
        
        df = pd.DataFrame(self.results)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = os.path.join(base_dir, 'data', f'monitoring_results_{timestamp}.csv')
        
        df.to_csv(filename, index=False)
        print(f"Results saved to {filename}")

def main():
    # Example usage
    query = "Tesla"  # Replace with your target brand/company
    monitor = SentimentMonitor(query, interval_minutes=5)
    monitor.monitor()

if __name__ == "__main__":
    main() 