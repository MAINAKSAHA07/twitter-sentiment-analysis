import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import glob

# Page config
st.set_page_config(
    page_title="Twitter Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("Twitter Sentiment Analysis Dashboard")

# Sidebar
st.sidebar.header("Settings")
query = st.sidebar.text_input("Search Query", "Tesla")
time_range = st.sidebar.selectbox(
    "Time Range",
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"]
)

# Load data
@st.cache_data
def load_data():
    data_files = glob.glob("../data/monitoring_results_*.csv")
    if not data_files:
        return pd.DataFrame()
    
    # Load and combine all data files
    dfs = []
    for file in data_files:
        df = pd.read_csv(file)
        df['created_at'] = pd.to_datetime(df['created_at'])
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

# Main content
data = load_data()

if data.empty:
    st.warning("No data available. Please run the monitoring script first.")
else:
    # Filter data based on time range
    now = datetime.now()
    if time_range == "Last 24 hours":
        filtered_data = data[data['created_at'] >= now - timedelta(days=1)]
    elif time_range == "Last 7 days":
        filtered_data = data[data['created_at'] >= now - timedelta(days=7)]
    elif time_range == "Last 30 days":
        filtered_data = data[data['created_at'] >= now - timedelta(days=30)]
    else:
        filtered_data = data
    
    # Sentiment distribution
    st.header("Sentiment Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Baseline Model")
        baseline_counts = filtered_data['baseline_sentiment'].value_counts()
        fig1 = px.pie(
            values=baseline_counts.values,
            names=baseline_counts.index,
            title="Sentiment Distribution (Baseline)"
        )
        st.plotly_chart(fig1, use_container_width=True)
    
    with col2:
        st.subheader("BERT Model")
        bert_counts = filtered_data['bert_sentiment'].value_counts()
        fig2 = px.pie(
            values=bert_counts.values,
            names=bert_counts.index,
            title="Sentiment Distribution (BERT)"
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Sentiment over time
    st.header("Sentiment Over Time")
    
    # Resample data by hour
    hourly_data = filtered_data.set_index('created_at').resample('H').agg({
        'baseline_sentiment': lambda x: x.value_counts().to_dict(),
        'bert_sentiment': lambda x: x.value_counts().to_dict()
    })
    
    # Create time series plot
    fig3 = go.Figure()
    
    # Add traces for each sentiment
    sentiments = ['positive', 'neutral', 'negative']
    colors = {'positive': 'green', 'neutral': 'blue', 'negative': 'red'}
    
    for sentiment in sentiments:
        baseline_counts = [d.get(sentiment, 0) if isinstance(d, dict) else 0 for d in hourly_data['baseline_sentiment']]
        bert_counts = [d.get(sentiment, 0) if isinstance(d, dict) else 0 for d in hourly_data['bert_sentiment']]
        
        fig3.add_trace(go.Scatter(
            x=hourly_data.index,
            y=baseline_counts,
            name=f"Baseline - {sentiment}",
            line=dict(color=colors[sentiment], dash='dash')
        ))
        
        fig3.add_trace(go.Scatter(
            x=hourly_data.index,
            y=bert_counts,
            name=f"BERT - {sentiment}",
            line=dict(color=colors[sentiment])
        ))
    
    fig3.update_layout(
        title="Sentiment Trends Over Time",
        xaxis_title="Time",
        yaxis_title="Number of Tweets",
        hovermode="x unified"
    )
    
    st.plotly_chart(fig3, use_container_width=True)
    
    # Recent tweets
    st.header("Recent Tweets")
    
    # Display recent tweets with sentiment
    recent_tweets = filtered_data.sort_values('created_at', ascending=False).head(10)
    
    for _, tweet in recent_tweets.iterrows():
        with st.expander(f"{tweet['created_at']} - {tweet['text'][:100]}..."):
            st.write("Full Text:", tweet['text'])
            st.write("Baseline Sentiment:", tweet['baseline_sentiment'])
            st.write("BERT Sentiment:", tweet['bert_sentiment'])

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
### About
This dashboard displays real-time sentiment analysis results from Twitter data.
The analysis is performed using both a baseline model (TF-IDF + Logistic Regression)
and a fine-tuned BERT model.
""") 