#!/usr/bin/env python
# coding: utf-8

# # Stock Price Prediction
# Let's say we want to make money by buying stocks. Since we want to make money, we only want to buy stock on days when the price will go up (we're against shorting the stock). We'll create a machine learning algorithm to predict if the stock price will increase tomorrow. If the algorithm says that the price will increase, we'll buy stock. If the algorithm says that the price will go down, we won't do anything.
# 
# We want to maximize our true positives - days when the algorithm predicts that the price will go up, and it actually goes go up. Therefore, we'll be using precision as our error metric for our algorithm, which is true positives / (false positives + true positives). This will ensure that we minimize how much money we lose with false positives (days when we buy the stock, but the price actually goes down).
# 
# This means that we will have to accept a lot of false negatives - days when we predict that the price will go down, but it actually goes up. This is okay, since we'd rather minimize our potential losses than maximize our potential gains.

# # Method
# Before we get to the machine learning, we need to do a lot of work to acquire and clean up the data. Here are the steps we'll follow:
# 
# Download historical stock prices from Yahoo finance..
# Explore the data...
# Setup the dataset to predict future prices using historical prices...
# Test a machine learning model...
# Setup a backtesting engine....
# Improve the accuracy of the model...
# At the end, we'll document some potential future directions we can go in to improve the technique.

# # Downloading the data
# First, we'll download the data from Yahoo Finance. We'll save the data after we download it, so we don't have to re-download it every time (this could cause our IP to get blocked).
# 
# We'll use data for a single stock (Microsoft) from when it started trading to the present.

# In[2]:


conda install -c conda-forge yfinance


# In[1]:


# Import finance API and get historical stock data
import yfinance as yf
import os
import json
import pandas as pd

DATA_PATH = "msft_data.json"

if os.path.exists(DATA_PATH):
    # Read from file if we've already downloaded the data.
    with open(DATA_PATH) as f:
        msft_hist = pd.read_json(DATA_PATH)
else:
    msft = yf.Ticker("MSFT")
    msft_hist = msft.history(period="max")

    # Save file to json in case we need it later.  This prevents us from having to re-download it every time.
    msft_hist.to_json(DATA_PATH)


# As we can see, we have one row of data for each day that Microsoft stock was traded. Here are the columns:
# 
# Open - the price the stock opened at.
# High - the highest price during the day
# Low - the lowest price during the day
# Close - the closing price on the trading day
# Volume - how many shares were traded
# Stock doesn't trade every day (there is no trading on weekends and holidays), so some dates are missing.

# In[2]:


# Display microsoft stock price history so we can look at the structure of the data
msft_hist.head(5)


# Next, we'll plot the data so we can see how the stock price has changed over time. This gives us another overview of the structure of the data.

# In[3]:


# Visualize microsoft stock prices
msft_hist.plot.line(y="Close", use_index=True)


# Preparing the data
# Ok, hopefully you've stopped kicking yourself for not buying Microsoft stock at any point in the past 30 years now.
# 
# Now, let's prepare the data so we can make predictions. We'll be predicting if the price will go up or down tomorrow based on data from today.
# 
# First, we'll identify a target that we're trying to predict. Our target will be if the price will go up or down tomorrow. If the price went up, the target will be 1 and if it went down, the target will be 0.
# 
# Next, we'll shift the data from previous days "forward" one day, so we can use it to predict the target price. This ensures that we don't accidentally use data from the same day to make predictions! (a very common mistake)
# 
# Then, we'll combine both so we have our training data.

# In[4]:


# Ensure we know the actual closing price
data = msft_hist[["Close"]]
data = data.rename(columns = {'Close':'Actual_Close'})

# Setup our target.  This identifies if the price went up or down
data["Target"] = msft_hist.rolling(2).apply(lambda x: x.iloc[1] > x.iloc[0])["Close"]


# In[5]:


# Shift stock prices forward one day, so we're predicting tomorrow's stock prices from today's prices.
msft_prev = msft_hist.copy()
msft_prev = msft_prev.shift(1)


# In[6]:


# Create our training data
predictors = ["Close", "Volume", "Open", "High", "Low"]
data = data.join(msft_prev[predictors]).iloc[1:]


# In[8]:


data.head(5)


# # Creating a machine learning model
# 
# 
# Next, we'll create a machine learning model to see how accurately we can predict the stock price.
# 
# Because we're dealing with time series data, we can't just use cross-validation to create predictions for the whole dataset. This will cause leakage where data from the future will be used to predict past prices. This doesn't match with the real world, and will make us think that our algorithm is much better than it actually is.
# 
# Instead, we'll split the data sequentially. We'll start off by predicting just the last 100 rows using the other rows.
# 
# We'll use a random forest classifier to generate our predictions. This is a good "default" model for a lot of applications, because it can pick up nonlinear relationships in the data, and is somewhat robust to overfitting with the right parameters.

# In[9]:


from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Create a random forest classification model.  Set min_samples_split high to ensure we don't overfit.
model = RandomForestClassifier(n_estimators=100, min_samples_split=200, random_state=1)

# Create a train and test set
train = data.iloc[:-100]
test = data.iloc[-100:]

model.fit(train[predictors], train["Target"])


# Next, we'll need to check how accurate the model was. Earlier, we mentioned using precision to measure error. We can do this by using the precision_score function from scikit-learn.

# In[10]:


from sklearn.metrics import precision_score

# Evaluate error of predictions
preds = model.predict(test[predictors])
preds = pd.Series(preds, index=test.index)
precision_score(test["Target"], preds)


# So our model is directionally accurate 51% of the time. This is only a little bit better than a coin flip! We can take a deeper look at the individual predictions and the actuals, and see where we're off.

# In[11]:


combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
combined.plot()


# # Backtesting
# 
# 
# Our model isn't great, but luckily we can still improve it. Before we do that, let's figure out how to make predictions across the entire dataset, not just the last 100 rows. This will give us a more robust error estimate. The last 100 days may have has atypical market conditions or other issues that make error metrics on those days unrealistic for future predictions (which are what we really care about).
# 
# To do this, we'll need to backtest. Backtesting ensures that we only use data from before the day that we're predicting. If we use data from after the day we're predicting, the algorithm is unrealistic (in the real world, you won't be able to use future data to predict that past!).
# 
# Our backtesting method will loop over the dataset, and train a model every 750 rows. We'll make it a function so we can avoid rewriting the code if we want to backtest again.
# 
# In the backtesting function, we will:
# 
# Split the training and test data
# Train a model
# Make predictions on the test data using predict_proba - this is because we want to really optimize for true positives. By default, the threshold for splitting 0/1 is .5, but we can set it to different values to tweak the precision. If we set it too high, we'll make fewer trades, but will have a lower potential for losses.

# In[12]:


def backtest(data, model, predictors, start=1000, step=750):
    predictions = []
    # Loop over the dataset in increments
    for i in range(start, data.shape[0], step):
        # Split into train and test sets
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        
        # Fit the random forest model
        model.fit(train[predictors], train["Target"])
        
        # Make predictions
        preds = model.predict_proba(test[predictors])[:,1]
        preds = pd.Series(preds, index=test.index)
        preds[preds > .6] = 1
        preds[preds<=.6] = 0
        
        # Combine predictions and test values
        combined = pd.concat({"Target": test["Target"],"Predictions": preds}, axis=1)
        
        predictions.append(combined)
    
    return pd.concat(predictions)


# In[13]:


predictions = backtest(data, model, predictors)


# In[14]:


predictions["Predictions"].value_counts()


# In[15]:


precision_score(predictions["Target"], predictions["Predictions"])


# # Improving accuracy
# 
# The model isn't very accurate, but at least now we can make predictions across the entire history of the stock. For this model to be useful, we have to get it to predict more accurately.
# 
# Let's add some more predictors to see if we can improve accuracy.
# 
# We'll add in some rolling means, so the model can evaluate the current price against recent prices. We'll also look at the ratios between different indicators.

# In[16]:


weekly_mean = data.rolling(7).mean()
quarterly_mean = data.rolling(90).mean()
annual_mean = data.rolling(365).mean()
weekly_trend = data.shift(1).rolling(7).mean()["Target"]


# In[17]:


data["weekly_mean"] = weekly_mean["Close"] / data["Close"]
data["quarterly_mean"] = quarterly_mean["Close"] / data["Close"]
data["annual_mean"] = annual_mean["Close"] / data["Close"]

data["annual_weekly_mean"] = data["annual_mean"] / data["weekly_mean"]
data["annual_quarterly_mean"] = data["annual_mean"] / data["quarterly_mean"]
data["weekly_trend"] = weekly_trend

data["open_close_ratio"] = data["Open"] / data["Close"]
data["high_close_ratio"] = data["High"] / data["Close"]
data["low_close_ratio"] = data["Low"] / data["Close"]


# In[18]:


full_predictors = predictors + ["weekly_mean", "quarterly_mean", "annual_mean", "annual_weekly_mean", "annual_quarterly_mean", "open_close_ratio", "high_close_ratio", "low_close_ratio", "weekly_trend"]
predictions = backtest(data.iloc[365:], model, full_predictors)


# In[19]:


precision_score(predictions["Target"], predictions["Predictions"])


# In[20]:


# Show how many trades we would make

predictions["Predictions"].value_counts()


# In[21]:


predictions.iloc[-100:].plot()


# Next steps
# We've come far in this project! We've downloaded and cleaned data, and setup a backtesting engine. We now have an algorithm that we can add more predictors to and continue to improve the accuracy of.
# 
# There are a lot of next steps we could take to improve our predictions:
# 
# Improve the technique
# Calculate how much money you'd make if you traded with this algorithm
# Improve the algorithm
# Run with a reduced step size! This will take longer, but increase accuracy
# Try discarding older data (only keeping data in a certain window)
# Try a different machine learning algorithm
# Tweak random forest parameters, or the prediction threshold
# Add in more predictors
# Account for activity post-close and pre-open
# Early trading
# Trading on other exchanges that open before the NYSE (to see what the global sentiment is)
# Economic indicators
# Interest rates
# Other important economic news
# Key dates
# Dividends
# External factors like elections
# Company milestones
# Earnings calls
# Analyst ratings
# Major announcements
# Prices of related stocks
# Other companies in the same sector
# Key partners, customers, etc.
# 

# In[ ]:




