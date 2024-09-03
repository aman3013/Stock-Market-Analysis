# Stock-Market-Analysis
===========================================================

**Overview**
------------

This challenge focuses on analyzing a large corpus of financial news data to discover correlations between news sentiment and stock market movements. The goal is to refine skills in Data Engineering, Financial Analytics, and Machine Learning Engineering.

**Dataset**
------------

The Financial News and Stock Price Integration Dataset (FNSPID) is used for this challenge. The dataset contains the following columns:

* `headline`: Article release headline
* `url`: Direct link to the full news article
* `publisher`: Author/creator of the article
* `date`: Publication date and time (UTC-4 timezone)
* `stock`: Stock ticker symbol (unique series of letters assigned to a publicly traded company)

**Task 1: Exploratory Data Analysis (EDA)**
------------------------------------------

### Step 1: Set up Python environment and Git version control

* Create a new GitHub repository for this challenge
* Create a new branch called "task-1" for the EDA analysis
* Commit work at least three times a day with descriptive commit messages

### Step 2: Perform EDA analysis

* **Descriptive Statistics**: Calculate basic statistics for textual lengths (like headline length)
* **Count the number of articles per publisher**: Identify which publishers are most active
* **Analyze publication dates**: See trends over time, such as increased news frequency on particular days or during specific events
* **Text Analysis (Sentiment analysis & Topic Modeling)**: Perform sentiment analysis on headlines to gauge the sentiment (positive, negative, neutral) associated with the news
* **Time Series Analysis**: Analyze the publication frequency over time and identify spikes in article publications related to specific market events
* **Publisher Analysis**: Identify which publishers contribute most to the news feed and analyze the type of news they report



**Task 2: Quantitative Analysis using PyNance and TA-Lib**
---------------------------------------------------------

### Step 1: Load and prepare the data

* Load stock price data into a pandas DataFrame
* Ensure the data includes columns like Open, High, Low, Close, and Volume

### Step 2: Apply analysis indicators with TA-Lib

* Calculate various technical indicators such as moving averages, RSI (Relative Strength Index), and MACD (Moving Average Convergence Divergence)

### Step 3: Visualize the data

* Create visualizations to better understand the data and the impact of different indicators on the stock price



**Task 3: Correlation between News and Stock Movement**
------------------------------------------------------

### Step 1: Date alignment

* Ensure that both datasets (news and stock prices) are aligned by dates

### Step 2: Sentiment analysis

* Conduct sentiment analysis on news headlines to quantify the tone of each article (positive, negative, neutral)

### Step 3: Calculate daily stock returns

* Compute the percentage change in daily closing prices to represent stock movements

### Step 4: Correlation analysis

* Use statistical methods to test the correlation between daily news sentiment scores and stock returns



This repository contains the code and findings for the Financial News Sentiment Analysis and Stock Movement Correlation Challenge. The challenge is divided into three tasks:

* Task 1: Exploratory Data Analysis (EDA)
* Task 2: Quantitative Analysis using PyNance and TA-Lib
* Task 3: Correlation between News and Stock Movement



**Folder Structure**
--------------------

```
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows
│       ├── unittests.yml
├── .gitignore
├── requirements.txt
├── src/
│   ├── __init__.py
├── notebooks/
│   ├── __init__.py
│   
├── tests/
│   ├── __init__.py
└── scripts/
    ├── __init__.py
    
```


**License**
------------

This repository is licensed under the MIT License.