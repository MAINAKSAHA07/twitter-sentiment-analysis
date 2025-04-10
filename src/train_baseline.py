import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

class BaselineModel:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            stop_words='english'
        )
        self.model = LogisticRegression(
            max_iter=1000,
            multi_class='multinomial',
            solver='lbfgs'
        )
    
    def prepare_data(self, df):
        """
        Prepare data for training
        
        Args:
            df (pandas.DataFrame): Processed DataFrame
            
        Returns:
            tuple: (X, y) where X is the feature matrix and y is the target vector
        """
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        y = df['sentiment']
        return X, y
    
    def train(self, X, y):
        """
        Train the model
        
        Args:
            X (scipy.sparse.csr.csr_matrix): Feature matrix
            y (pandas.Series): Target vector
        """
        self.model.fit(X, y)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Args:
            X_test (scipy.sparse.csr.csr_matrix): Test feature matrix
            y_test (pandas.Series): Test target vector
            
        Returns:
            dict: Evaluation metrics
        """
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        
        return {
            'accuracy': accuracy,
            'report': report
        }
    
    def save_model(self, model_dir=None):
        """
        Save the trained model and vectorizer
        
        Args:
            model_dir (str): Directory to save the models
        """
        if model_dir is None:
            model_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
        
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))
        joblib.dump(self.model, os.path.join(model_dir, 'baseline_model.joblib'))
        
        print(f"Models saved to {model_dir}")

def main():
    # Load processed data
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    processed_file = os.path.join(base_dir, 'data', 'processed_tweets.csv')
    df = pd.read_csv(processed_file)
    
    # Initialize model
    baseline_model = BaselineModel()
    
    # Prepare data
    X, y = baseline_model.prepare_data(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    print("Training baseline model...")
    baseline_model.train(X_train, y_train)
    
    # Evaluate model
    print("\nEvaluating model...")
    metrics = baseline_model.evaluate(X_test, y_test)
    
    print(f"\nAccuracy: {metrics['accuracy']:.4f}")
    print("\nClassification Report:")
    print(pd.DataFrame(metrics['report']).transpose())
    
    # Save model
    baseline_model.save_model()

if __name__ == "__main__":
    main() 