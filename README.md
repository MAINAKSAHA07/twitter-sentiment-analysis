# Twitter Sentiment Analysis

A real-time sentiment analysis system for Twitter data using machine learning and natural language processing techniques.

## Important Note
The virtual environment (`venv/`) is not included in this repository. Each user should create their own virtual environment following the setup instructions below.

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/MAINAKSAHA07/twitter-sentiment_analysis.git
cd twitter-sentiment_analysis
```

2. Create and activate a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root with your Twitter API credentials:
```
TWITTER_BEARER_TOKEN=your_bearer_token
```

5. Download required NLTK resources:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('vader_lexicon')
```

## Usage

1. Collect tweets:
```bash
python src/twitter_api.py
```

2. Preprocess data:
```bash
python src/preprocess.py
```

3. Train models:
```bash
python src/train_baseline.py
python src/train_bert.py
```

4. Run real-time monitoring:
```bash
python src/real_time_monitor.py
```

5. Launch dashboard:
```bash
streamlit run dashboard/app.py
```

## Project Structure
```
twitter-sentiment/
├── data/              # Store raw and processed data
├── models/            # Save trained models
├── src/              # Source code
│   ├── twitter_api.py
│   ├── preprocess.py
│   ├── train_baseline.py
│   ├── train_bert.py
│   ├── real_time_monitor.py
├── dashboard/        # Streamlit dashboard
│   └── app.py
├── requirements.txt
└── README.md
```

## Features

- Real-time tweet collection
- Text preprocessing and cleaning
- Sentiment analysis using both traditional ML and BERT
- Real-time sentiment monitoring
- Interactive dashboard for visualization

## License

MIT License 