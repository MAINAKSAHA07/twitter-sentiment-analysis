import re
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')

class TweetPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
    def clean_text(self, text):
        """
        Clean tweet text by removing URLs, mentions, hashtags, and special characters
        
        Args:
            text (str): Raw tweet text
            
        Returns:
            str: Cleaned text
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove special characters and numbers
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\d+', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text.lower()
    
    def tokenize(self, text):
        """
        Tokenize text and remove stopwords
        
        Args:
            text (str): Cleaned text
            
        Returns:
            list: List of tokens
        """
        tokens = word_tokenize(text)
        return [token for token in tokens if token not in self.stop_words]
    
    def get_sentiment_label(self, text):
        """
        Get sentiment label using VADER sentiment analyzer
        
        Args:
            text (str): Cleaned text
            
        Returns:
            str: Sentiment label ('positive', 'neutral', or 'negative')
        """
        scores = self.sentiment_analyzer.polarity_scores(text)
        
        if scores['compound'] >= 0.05:
            return 'positive'
        elif scores['compound'] <= -0.05:
            return 'negative'
        else:
            return 'neutral'
    
    def preprocess_tweets(self, df):
        """
        Preprocess all tweets in the dataframe
        
        Args:
            df (pandas.DataFrame): DataFrame containing tweets
            
        Returns:
            pandas.DataFrame: Preprocessed DataFrame
        """
        # Clean text
        df['cleaned_text'] = df['text'].apply(self.clean_text)
        
        # Tokenize
        df['tokens'] = df['cleaned_text'].apply(self.tokenize)
        
        # Get sentiment labels
        df['sentiment'] = df['cleaned_text'].apply(self.get_sentiment_label)
        
        return df

def main():
    # Example usage
    preprocessor = TweetPreprocessor()
    
    # Load tweets
    tweets_file = '../data/tweets_*.csv'  # Replace with your actual file
    df = pd.read_csv(tweets_file)
    
    # Preprocess tweets
    processed_df = preprocessor.preprocess_tweets(df)
    
    # Save processed data
    output_file = '../data/processed_tweets.csv'
    processed_df.to_csv(output_file, index=False)
    print(f"Saved processed tweets to {output_file}")

if __name__ == "__main__":
    main() 