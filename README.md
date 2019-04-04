# DeepDayTrade

In this implementation, we are using a Multilayer Perceptron (MLP) for predicting the stock price of a company based on the historical prices available. Here we are using day-wise closing price of two different stock markets, Standard & Poor's 500
(S&P) and the New York Stock Exchange (NYSE) as well as several companies: Apple (AAPL), IBM, General Mills, Nike, Goldman. The network was trained with the historical stock prices of the indexes and companies.

The results obtained were compared with an ARIMA model and it has been observed that the neural networks are outperforming the existing linear model (ARIMA).


## Data Set and Training & Parameters

Historical stock price information is collected over a 2-year time period. Every Fifth week is used as validation data to test the prediction of our neural network. The validation set is a segmented portion of the dataset that is not used for training the neural network and has effectively never been seen by the model. This allows us to see how well our machine learning method generalizes it’s learning. Generalization refers to your model's ability to adapt properly to new, previously unseen data, drawn from the same distribution as the one used to create the model.


## Performance Assessment

To assess performance, we use several measures to indicate the significance of our findings. A straight forward method of prediction is to look at the absolute difference our prediction is from the actual value and calculate the percentage of chance. While this is useful for noting generalize how close our predictions, it doesn’t describe all the important elements of the problem.

 Capital gain is an underlying incentive in stock price prediction, so it is not only necessary to predict the price accurately in relative amount, but directional correctness is desirable as well. This is because trades are profitable when bought low and sold high. To be able to profit from stock price changes, it is necessary to know which direction the stock will move in the future.


## References

Navon, et al. “Financial Time Series Prediction Using Deep Learning.” [Astro-Ph/0005112] A Determination of the Hubble Constant from Cepheid Distances and a Model of the Local Peculiar Velocity Field, American Physical Society, 11 Nov. 2017, arxiv.org/abs/1711.04174.

Yue-Gang, et al. “Neural Networks for Stock Price Prediction.” [Astro-Ph/0005112] A Determination of the Hubble Constant from Cepheid Distances and a Model of the Local Peculiar Velocity Field, American Physical Society, 29 May 2018, arxiv.org/abs/1805.11317.
