# Import necessary libraries
#! C:\Users\amanu\OneDrive\Desktop\Stock-Market-Analysis\.venv\Scripts\python.exe
import os
import sys
import talib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from textblob import TextBlob

nltk.download('vader_lexicon')

# Descriptive Statistics Functions
def descriptive_statistics(df):
    """Generates and displays descriptive statistics and visualizations."""
    print_headline_statistics(df)
    plot_publisher_article_counts(df)
    analyze_publication_dates(df)
    analyze_day_of_week(df)
    analyze_stock_mentions(df)

def print_headline_statistics(df):
    """Prints basic statistics for textual lengths (like headline length)."""
    headline_length = df['headline'].apply(len)
    print("Headline Length Statistics:")
    print(headline_length.describe())

def plot_publisher_article_counts(df):
    """Counts and plots the number of articles per publisher."""
    publisher_counts = df['publisher'].value_counts()
    print("\nPublisher Article Counts:")
    print(publisher_counts.head(10))
    
    plt.figure(figsize=(10, 6))
    publisher_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Most Active Publishers')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def analyze_publication_dates(df):
    """Analyzes and plots publication dates to see trends over time."""
    df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=False)
    publication_counts = df['date'].dt.date.value_counts().sort_index()
    print("\nPublication Date Counts:")
    print(publication_counts)

    plt.figure(figsize=(12, 6))
    publication_counts.plot(kind='line')
    plt.title('Publication Date Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()

def analyze_day_of_week(df):
    """Analyzes and plots article counts by day of the week."""
    df['day_of_week'] = df['date'].dt.day_name()
    day_of_week_counts = df['day_of_week'].value_counts()
    print("\nDay of the Week Counts:")
    print(day_of_week_counts)

    plt.figure(figsize=(8, 6))
    day_of_week_counts.plot(kind='bar')
    plt.title('Day of the Week Counts')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def analyze_stock_mentions(df):
    """Counts and plots the number of articles mentioning each stock."""
    stock_counts = df['stock'].value_counts()
    print("\nStock Article Counts:")
    print(stock_counts.head(10))
    
    plt.figure(figsize=(10, 6))
    stock_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Most Frequently Mentioned Stocks')
    plt.xlabel('Stock')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

# Sentiment Analysis Functions
def perform_vader_sentiment_analysis(df, text_column):
    """Performs VADER sentiment analysis on the specified text column."""
    sid = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df[text_column].apply(lambda x: sid.polarity_scores(x)['compound'])
    return df

def perform_textblob_sentiment_analysis(df, text_column):
    """Performs TextBlob sentiment analysis on the specified text column."""
    df['textblob_sentiment'] = df[text_column].apply(lambda x: TextBlob(x).sentiment.polarity)
    return df

def visualize_sentiment_distribution(df, sentiment_column, title):
    """Visualizes the distribution of sentiment scores."""
    plt.figure(figsize=(10, 6))
    sns.histplot(df[sentiment_column], bins=50, kde=True)
    plt.title(title)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()

# Topic Modeling Functions
def perform_lda_topic_modeling(df, text_column, n_topics=5, max_features=5000):
    """Performs LDA topic modeling on the specified text column."""
    vectorizer = CountVectorizer(stop_words='english', max_features=max_features)
    X = vectorizer.fit_transform(df[text_column])
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42)
    lda.fit(X)

    words = vectorizer.get_feature_names_out()
    for i, topic in enumerate(lda.components_):
        print(f"Top words in topic {i+1}:")
        print([words[j] for j in topic.argsort()[-10:]])
        print("\n")

    topic_results = lda.transform(X)
    df['lda_topic'] = topic_results.argmax(axis=1)
    return df

def perform_nmf_topic_modeling(df, text_column, n_topics=5, max_features=5000):
    """Performs NMF topic modeling on the specified text column."""
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
    tfidf = tfidf_vectorizer.fit_transform(df[text_column])

    nmf = NMF(n_components=n_topics, random_state=42)
    nmf.fit(tfidf)

    words = tfidf_vectorizer.get_feature_names_out()
    for i, topic in enumerate(nmf.components_):
        print(f"Top words in topic {i+1}:")
        print([words[j] for j in topic.argsort()[-10:]])
        print("\n")

    topic_results = nmf.transform(tfidf)
    df['nmf_topic'] = topic_results.argmax(axis=1)
    return df

def visualize_top_words_in_topics(nmf, tfidf_vectorizer, n_topics=5):
    """Visualizes the top words in each topic."""
    words = tfidf_vectorizer.get_feature_names_out()
    for i, topic in enumerate(nmf.components_):
        top_words_idx = topic.argsort()[-10:]
        top_words = [words[j] for j in top_words_idx]
        top_scores = topic[top_words_idx]

        plt.figure(figsize=(8, 6))
        plt.barh(top_words, top_scores, color='skyblue')
        plt.xlabel('Score')
        plt.title(f'Top words in topic {i+1}')
        plt.show()

def visualize_topic_distribution(df, topic_column, title):
    """Visualizes the distribution of topics across the dataset."""
    plt.figure(figsize=(8, 6))
    sns.countplot(x=topic_column, data=df, palette='viridis')
    plt.title(title)
    plt.xlabel('Topic')
    plt.ylabel('Number of Headlines')
    plt.show()

# Main Analysis Function
def analyze_data(df, text_column):
    """Main function to perform sentiment analysis and topic modeling."""
    # Perform sentiment analysis
    df = perform_vader_sentiment_analysis(df, text_column)
    df = perform_textblob_sentiment_analysis(df, text_column)
    
    # Visualize sentiment distribution
    visualize_sentiment_distribution(df, 'vader_sentiment', 'Distribution of VADER Sentiment Scores')
    visualize_sentiment_distribution(df, 'textblob_sentiment', 'Distribution of TextBlob Sentiment Scores')

    # Perform and visualize LDA topic modeling
    df = perform_lda_topic_modeling(df, text_column)
    visualize_topic_distribution(df, 'lda_topic', 'Distribution of LDA Topics Across Headlines')

    # Perform and visualize NMF topic modeling
    df = perform_nmf_topic_modeling(df, text_column)
    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_vectorizer.fit_transform(df[text_column])
    visualize_top_words_in_topics(NMF(n_components=5, random_state=42).fit(tfidf_vectorizer.fit_transform(df[text_column])), tfidf_vectorizer)
    visualize_topic_distribution(df, 'nmf_topic', 'Distribution of NMF Topics Across Headlines')

    # Display sample data
    print(df.head(10))

# Entry point for standalone execution
if __name__ == "__main__":
    raw_analyst_ratings_df = pd.read_csv('C:/Users/amanu/OneDrive/Desktop/Stock-Market-Analysis/data/raw_analyst_ratings.csv')
    descriptive_statistics(raw_analyst_ratings_df)
    analyze_data(raw_analyst_ratings_df, 'headline')


def analyze_stock_data(file_path):
    # Load the CSV file into a DataFrame
    data = pd.read_csv(file_path)

    # Display the first few rows of the DataFrame to check the data
    print("First few rows of the data:")
    print(data.head())

    # Check for missing values
    print("\nMissing values:")
    print(data.isnull().sum())

    # Convert the 'Date' column to datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Set the 'Date' column as the index
    data.set_index('Date', inplace=True)

    # Display the data types of the columns
    print("\nData types of columns:")
    print(data.dtypes)

    # Ensure the relevant columns are in the correct format
    data['Open'] = data['Open'].astype(float)
    data['High'] = data['High'].astype(float)
    data['Low'] = data['Low'].astype(float)
    data['Close'] = data['Close'].astype(float)
    data['Volume'] = data['Volume'].astype(int)
    data['Adj Close'] = data['Adj Close'].astype(float)
    data['Dividends'] = data['Dividends'].astype(float)
    data['Stock Splits'] = data['Stock Splits'].astype(float)

    # Display the cleaned DataFrame
    print("\nCleaned DataFrame info:")
    print(data.info())

    # Calculate Simple Moving Averages (SMA)
    data['SMA_20'] = talib.SMA(data['Close'], timeperiod=20)
    data['SMA_50'] = talib.SMA(data['Close'], timeperiod=50)

    # Calculate Exponential Moving Averages (EMA)
    data['EMA_20'] = talib.EMA(data['Close'], timeperiod=20)
    data['EMA_50'] = talib.EMA(data['Close'], timeperiod=50)

    # Calculate Relative Strength Index (RSI)
    data['RSI_14'] = talib.RSI(data['Close'], timeperiod=14)

    # Calculate MACD (Moving Average Convergence Divergence)
    data['MACD'], data['MACD_signal'], data['MACD_hist'] = talib.MACD(data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)

    # Display the first few rows with the new indicators
    print("\nFirst few rows with technical indicators:")
    print(data[['Close', 'SMA_20', 'SMA_50', 'EMA_20', 'EMA_50', 'RSI_14', 'MACD', 'MACD_signal', 'MACD_hist']].head())

    # Plotting the Close price along with SMAs
    plt.figure(figsize=(14, 7))
    plt.plot(data['Close'], label='Close Price', color='black')
    plt.plot(data['SMA_20'], label='SMA 20', color='blue')
    plt.plot(data['SMA_50'], label='SMA 50', color='red')
    plt.title('Close Price with SMAs')
    plt.legend()
    plt.show()

    # Plotting RSI
    plt.figure(figsize=(14, 7))
    plt.plot(data['RSI_14'], label='RSI 14', color='purple')
    plt.title('RSI 14')
    plt.axhline(70, color='red', linestyle='--')
    plt.axhline(30, color='green', linestyle='--')
    plt.legend()
    plt.show()

    return data
