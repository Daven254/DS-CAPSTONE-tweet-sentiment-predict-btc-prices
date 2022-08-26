# tweet-sentiment-predict-btc-prices

This is a capstone project that is still in progress for my ongoing DS bootcamp.

## Extraction 

- Extracting Tweets from Twitter for NLP using Twint. This process is rather manual, but can be eventually automated using a scheduler (to be added later down the road)
  - Important data from each tweet that I decided to use includes: 
    - time 
    - text 
    - like count 
    - retweet count 
    - reply count
    - conversation_id
    - username
  - Sentiment Analysis on tweets regarding Bitcoin (or other cryptocurrencies) produces a more robust sampling of from Tweet data. Applying **VADER** for sentiment analysis provides: 
    - positive sentiment
    - negative sentiment
    - neutral sentiment
    - overall sentiment scoring. **Only the overall sentinment was used for this project**
    
    
- Extracting Binance KLINE data (1 Minute Candles) to gauge Price changes over time
  - Important data includes Open, Close, High, Low, Number of Trades, and Volume of BTC on Binance's Exchange
  
 ## Transformation
- The two datasets were cleaned and merged used built-in python functionality, and required the usage of **Pandas** for manipulation of raw data (.csv)
  - They were joined along the key value of date. Although a tweet or many tweets were found per second, each candle was only based on the minute; therefore, multiple tweets were associated with a single candle.
- Additional columns were also added to measure percent change/price change of BTC


## Data Visualization

EDA was the first usage of this data. Data visualization required the use of **Seaborn**, **Plotly Express**, **Matplotlib**.

## Regression Models

The first ML techniques used were regression models, including:
  
  - Linear Regression
  - Multiple Variable Linear Regression
  - Log-Lin Regression
  - Log-Log Regression
  - Interaction Regression
  
## Clustering

  - Unsupervised ML was also utilized via: 
    - K-Means Clustering 
    - Hierarchial Clustering (Agglomerative)
    - DBScan was ran as well

## Simple Classification

- All Classification was ran to find accuracy score and ROC and AUC
- Confuscion Matrices were generated
- All results were saved into a dataframe to determine the best simple classification technique
- 
  - kNN
  - Logistic Regression
  - Decision Trees


## Data Sources
1. Twitter: Tweets regarding bitcoin/BTC
  - Extracted using Twint
  - Sentiment produced from VADER 
2. Binance: 1 Minute KLINE data

### Barebones Libraries Used

### Manipulations and Calculations
- pandas as pd
- numpy as np

### Statistics
- statsmodels.api as sm
- statsmodels.stats.outliers_influence 
- statsmodels.graphics.gofplots
- sklearn

### Visualizations
- matplotlib.pyplot as plt
- seaborn as sns
- plotly.express as px

### Regression Modeling
- sklearn.feature_selection
- sklearn.model_selection 


### Clustering 
- sklearn 
- sklearn.cluster (KMeans, MiniBatchKMeans, AgglomerativeClustering)
- sklearn.preprocessing
- scipy.cluster (kmean, dendrogram)
- kneed

### Classification
- sklearn.metrics
- sklearn.pipeline
- sklearn.model_selection (train_test_split, cross_val_score, GridSearchCV)
- sklearn.preprocessing  (StandardScaler, MinMaxScaler, scale)
- sklearn.metrics (accuracy_score, classification_report, confusion_matrix, roc_curve, auc)
- sklearn.neighbors.KNeighborsClassifier
- sklearn.linear_model.LogisticRegression
- sklearn.tree.DecisionTreeClassifier

**Data was collected between 01 Aug 2022 up to and including 09 Aug 2022**

**ETH data was also collected and transformed into the overall dataframe, but was removed for the purpose of testing the program for functionality**
