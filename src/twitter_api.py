import os
import tweepy
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import time
import random

# Load environment variables
load_dotenv()

class TwitterAPI:
    def __init__(self):
        self.bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
        if not self.bearer_token:
            raise ValueError("Twitter Bearer Token not found in environment variables")
        
        self.client = tweepy.Client(bearer_token=self.bearer_token)
        
    def fetch_tweets(self, query, max_results=5000):
        """
        Fetch tweets based on a search query
        
        Args:
            query (str): Search query
            max_results (int): Maximum number of tweets to fetch
            
        Returns:
            list: List of tweet dictionaries
        """
        tweets = []
        
        # Add language filter and exclude retweets
        query = f"{query} lang:en -is:retweet"
        
        try:
            # For small number of tweets, we can fetch them in a single request
            response = self.client.search_recent_tweets(
                query=query,
                tweet_fields=['created_at', 'public_metrics', 'text'],
                max_results=min(100, max_results)  # Twitter API max is 100 per request
            )
            
            if response.data:
                for t in response.data:
                    tweets.append({
                        'id': t.id,
                        'text': t.text,
                        'created_at': t.created_at,
                        'retweet_count': t.public_metrics['retweet_count'],
                        'reply_count': t.public_metrics['reply_count'],
                        'like_count': t.public_metrics['like_count'],
                        'quote_count': t.public_metrics['quote_count']
                    })
                    
                    # Stop if we have enough tweets
                    if len(tweets) >= max_results:
                        break
                
                print(f"Fetched {len(tweets)} tweets")
                
                # Save tweets
                if tweets:
                    self.save_tweets(tweets)
            
        except tweepy.errors.TooManyRequests:
            print("Rate limit reached. Saving fetched tweets...")
            if tweets:
                self.save_tweets(tweets)
            return tweets
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            if tweets:
                self.save_tweets(tweets)
            return tweets
        
        return tweets
    
    def save_tweets(self, tweets, filename=None):
        """
        Save tweets to a CSV file
        
        Args:
            tweets (list): List of tweet dictionaries
            filename (str, optional): Output filename
        """
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
            os.makedirs(data_dir, exist_ok=True)
            filename = os.path.join(data_dir, f'tweets_{timestamp}.csv')
        
        df = pd.DataFrame(tweets)
        df.to_csv(filename, index=False)
        print(f"Saved {len(tweets)} tweets to {filename}")

def main():
    # Example usage
    api = TwitterAPI()
    
    # Example query - replace with your target brand/company
    query = "Tesla"
    
    print(f"Fetching tweets for query: {query}")
    tweets = api.fetch_tweets(query, max_results=25)  # Fetch only 25 tweets
    
    if tweets:
        api.save_tweets(tweets)
    else:
        print("No tweets found for the given query")

if __name__ == "__main__":
    main() 