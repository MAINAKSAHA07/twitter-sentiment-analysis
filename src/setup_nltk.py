import nltk

def download_nltk_resources():
    """Download all required NLTK resources"""
    resources = [
        'punkt',
        'stopwords',
        'vader_lexicon',
        'averaged_perceptron_tagger',
        'wordnet',
        'omw-1.4'
    ]
    
    for resource in resources:
        print(f"Downloading {resource}...")
        nltk.download(resource)
        print(f"Downloaded {resource} successfully!")

if __name__ == "__main__":
    print("Downloading NLTK resources...")
    download_nltk_resources()
    print("All NLTK resources downloaded successfully!") 