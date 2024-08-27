import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, 'C:/Users/amanu/OneDrive/Desktop/Stock-Market-Analysis/')

def descriptive_statistics(df):
    # Obtain basic statistics for textual lengths (like headline length)
    headline_length = df['headline'].apply(len)
    print("Headline Length Statistics:")
    print(headline_length.describe())

    # Count the number of articles per publisher to identify which publishers are most active
    publisher_counts = df['publisher'].value_counts()
    print("\nPublisher Article Counts:")
    print(publisher_counts.head(10))  # Print the top 10 most active publishers

    # Plot a bar chart to visualize the publisher article counts
    plt.figure(figsize=(10, 6))
    publisher_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Most Active Publishers')
    plt.xlabel('Publisher')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    # Analyze the publication dates to see trends over time
    df['date'] = pd.to_datetime(df['date'], format='ISO8601', utc=False)
    publication_counts = df['date'].dt.date.value_counts().sort_index()
    print("\nPublication Date Counts:")
    print(publication_counts)

    # Plot a line chart to visualize the publication date counts
    plt.figure(figsize=(12, 6))
    publication_counts.plot(kind='line')
    plt.title('Publication Date Counts Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.grid(True)
    plt.show()

    # Analyze the publication dates to see trends over time (by day of the week)
    df['day_of_week'] = df['date'].dt.day_name()
    day_of_week_counts = df['day_of_week'].value_counts()
    print("\nDay of the Week Counts:")
    print(day_of_week_counts)

    # Plot a bar chart to visualize the day of the week counts
    plt.figure(figsize=(8, 6))
    day_of_week_counts.plot(kind='bar')
    plt.title('Day of the Week Counts')
    plt.xlabel('Day of the Week')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # Analyze the publication dates to see trends over time (by stock)
    stock_counts = df['stock'].value_counts()
    print("\nStock Article Counts:")
    print(stock_counts.head(10))  # Print the top 10 most frequently mentioned stocks

    # Plot a bar chart to visualize the stock article counts
    plt.figure(figsize=(10, 6))
    stock_counts.head(10).plot(kind='bar')
    plt.title('Top 10 Most Frequently Mentioned Stocks')
    plt.xlabel('Stock')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def main():
    raw_analyst_ratings_df = pd.read_csv('../data/raw_analyst_ratings.csv')
    descriptive_statistics(raw_analyst_ratings_df)

if __name__ == "__main__":
    main()