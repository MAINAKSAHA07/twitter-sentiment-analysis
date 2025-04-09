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
            for tweet in tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'public_metrics', 'text'],
                max_results=100,
                limit=max_results//100
            ):
                if tweet.data:
                    for t in tweet.data:
                        tweets.append({
                            'id': t.id,
                            'text': t.text,
                            'created_at': t.created_at,
                            'retweet_count': t.public_metrics['retweet_count'],
                            'reply_count': t.public_metrics['reply_count'],
                            'like_count': t.public_metrics['like_count'],
                            'quote_count': t.public_metrics['quote_count']
                        })
                
                # Add a random delay between requests to avoid rate limits
                time.sleep(random.uniform(1, 3))
                
                # Print progress
                print(f"Fetched {len(tweets)} tweets so far...")
                
        except tweepy.errors.TooManyRequests:
            print("Rate limit reached. Waiting for 15 minutes before retrying...")
            time.sleep(900)  # Wait for 15 minutes
            return self.fetch_tweets(query, max_results)  # Retry the request
            
        except Exception as e:
            print(f"An error occurred: {str(e)}")
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
            filename = f'../data/tweets_{timestamp}.csv'
        
        df = pd.DataFrame(tweets)
        df.to_csv(filename, index=False)
        print(f"Saved {len(tweets)} tweets to {filename}")

def main():
    # Example usage
    api = TwitterAPI()
    
    # Example query - replace with your target brand/company
    query = "Tesla"
    
    print(f"Fetching tweets for query: {query}")
    tweets = api.fetch_tweets(query, max_results=1000)  # Reduced from 5000 to 1000
    
    if tweets:
        api.save_tweets(tweets)
    else:
        print("No tweets found for the given query")

if __name__ == "__main__":
    main() 