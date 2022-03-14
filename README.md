### Financial Sentiment Analysis 📈: Project Overview

Created a model that can classify a Financial sentence as a Positive, Negative, or Neutral sentiment with **(66% Accuracy)** to detect polarity within the text.

Pulled over **5842 examples** from Kaggle using pandas and opendatasets libraries in python.

Applied **Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Naive Bayes**, and **KNeighborsClassifier** and optimized using **GridSearchCV** to find the best model.

### Code Used

Python version: *Python 3.7.11* 

Packages: *pandas, opendatasets, seaborn, matplotlib, numpy, nltk, wordcloud, collections, imblearn.over_sampling, re, string* and *textblob*

For Web Framework Requirements: *pip install -r requirements.txt*


### Resources Used

[The dataset from Kaggle](https://www.kaggle.com/sbhatti/financial-sentiment-analysis)

[How to download Kaggle datasets to Jupyter notebook guide](https://www.analyticsvidhya.com/blog/2021/04/how-to-download-kaggle-datasets-using-jupyter-notebook/)

[Cheatsheet for Markdown](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet)


## Data Collection
Used Kaggle to pull the datasets 5842 books with 2 columns:
* Sentence              
* Sentiment             


## Data Cleaning

After pulling the data, I cleaned up the dataset to reduce noises in the dataset. The changes were made follows:

* Made lowercase the sentences, cleaned punctuations in the sentences, deleted the newlines, removed numbers and possible links from the sentences.
* Removed stop words from the sentences and lemmatized them.


## Exploratory Data Analysis

Visualized the cleaned data to see the trends.

* Created *WordCloud* for **Sentence** variables.
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/wordcloud.png "Word Cloud")

* Created *Donut chart* for **Sentiment** data.
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/donut_chart.png "% of sentiments")
It looks like our data contains **negative** sentiments more than half of the whole dataset.

* Created *2-Gram Analysis Bar Graphs* for **Sentence** variables.
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/p_2gram.png "2-gram of Reviews with Positive Sentiments")
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/n_2gram.png "W2-gram of Reviews with Negative Sentiments")
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/nl_2gram.png "2-gram of Reviews with Neutral Sentiments")

* Created a histogram for **Polarity Score** in Sentences
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/polarity_score.png "Polarity Score in Sentences")
Sentences with *negative* polarity will be in range of [-1, 0), *neutral* ones will be 0.0, and *positive* reviews will have the range of (0, 1).

* Created a histogram for **Length of Sentences** 
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/length_of_reviews.png "Length of Reviews")
Based on this histogram, we know that our review has text length between approximately 50-100 characters.

* Created a histogram for **Word Counts** in Sentences
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/word_counts.png "Word Counts in Reviews")
From the figure above, we infer that most of the reviews consist of 5-15 words. 

## Model Building

Encoded the target variable:
* **Sentiment** variables were encoded.

Gave importance of each words in the Sentence column with Term Frequency - Inverse Document Frequency (TF-IDF) Vectorizer.

Resampled the dataset with Synthetic Minority Oversampling Technique (SMOTE) to balance the dataset.
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/donut_chart_balanced_data.png "R% of sentimets after resampling")

Data were split into **train (80%)** and **test (20%)** sets.

I used six models *(Decision Tree Classifier, Logistic Regression, Support Vector Classifier, Random Forest Classifier, Bernoulli Bayes, and KNeighborsClassifier)* to predict the sentiment and evaluated them by using *Accuracy*.


## Model Performance Evalution
Logistic Regression model performed better than any other models in this project.

|Model                      |Test Accuracy Score|                      
| -------------             |:-----------------:|                       
|Decision Tree              |0.5323477929984781|
|Logistic Regression        |0.6259826132771339|
|Support Vector Classifier  |0.5890109471958787|
|Random Forest Classifier   |0.5638441049057488|
|Naive Bayes                |0.5997986184287554|
|K-Neighbots                |0.5013675213675214|


## Hyperparameter Tuning

We got the best accuracy 65.08 % with GridSearchCV and find the optimal hyperparameters.

## Best Model

Applied Logistic Regression model with the optimal hyperparameters and got 66% Accuracy score.

## Confusion Matrix
![alt text](https://github.com/cerenkasap/financial_sentiment_analysis/blob/master/images/confusion_matrix.png "Confusion Matrix of Financial Sentiment Analysis")

The Confusion Matrix above shows that our model needs to be improved to categorize sentiments better.

Thanks for reading :) 
