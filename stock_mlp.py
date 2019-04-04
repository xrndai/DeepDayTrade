#https://stackoverflow.com/questions/42112260/how-do-i-use-the-tensorboard-callback-of-keras
import math
import numpy as np

import keras
from keras.layers import Dense, Dropout
from keras.models import Sequential

from sklearn.preprocessing import MinMaxScaler

import pylab as pl

from pandas_datareader import DataReader
# from datetime import datetime

import datetime, quandl

from keras.utils.vis_utils import plot_model

 companies = ["gis", "nke", "gs", "ibm","aapl"]
# companies = ["gm"]

#companies = ["NASDAQOMX/COMP-NASDAQ"]



for company in companies:

     train = DataReader(company,  'yahoo', datetime(2016, 6, 13), datetime(2018, 6, 8))
     test = DataReader(company,  'yahoo', datetime(2018, 6, 11), datetime(2018, 11, 30))

#    train = quandl.get(company,
#              trim_start='2016-06-13',
#              trim_end='2018-06-08')
#
#    test = quandl.get(company,
#              trim_start='2016-06-13',
#              trim_end='2018-06-08')

	index = []
	for i in range(len(train)):
		index.append(i)

	train["index"] = index

	index = []
	for i in range(len(test)):
		index.append(i)

	test["index"] = index


	# print(train)
	# print(test.to_string())

	# print (train)
	# print (test)

	# print(str(train.index[0]).split()[0].split("-")[2])

	X = []
	Y = []

	X_TEST = []
	Y_TEST = []

	# print(len(train))
	# print(len(test))

	close_list = []
	close_list.append(train["Close"][0])

	test_close = []
	test_close.append(test["Close"][0])

	
	for i in range(1,len(train)):
		date = int(str(train.index[i]).split()[0].split("-")[2])
		prev_date = int(str(train.index[i-1]).split()[0].split("-")[2])



		if date > (prev_date+1) and date < (prev_date+3) and date != 1:
			print("date: {}, prev: {}".format(date,prev_date))
			close_list.append(train["Close"][i-1])

		close_list.append(train["Close"][i])

	# print(len(close_list))

	for i in range(1,len(test)):
		date = int(str(test.index[i]).split()[0].split("-")[2])
		prev_date = int(str(test.index[i-1]).split()[0].split("-")[2])

		if date > (prev_date+1) and date < (prev_date+3) and date != 1:
			print("date: {}, prev: {}".format(date,prev_date))
			test_close.append(test["Close"][i-1])

		test_close.append(test["Close"][i])

	test_close.insert(60, test_close[59])

	# count = 0
	# for data in test_close:
	# 	print(count, end=" ")
	# 	print(data)
	# 	count +=1

	# print(len(test_close))

	# exit()


	for i in range(len(close_list)):
		x = i+1

		if (x%5) != 0:
			X.append(close_list[i])
		else:
			Y.append(close_list[i])
			count =0

	# print(len(X))
	# print(len(Y))
	
	for i in range(len(test_close)):
		x= i+1

		if (x%5) != 0:
			X_TEST.append(test_close[i])
		else:
			Y_TEST.append(test_close[i])
			count = 0

	# print(len(X_TEST))
	# print(len(Y_TEST))


	print(len(X))
	print(len(Y))

	# print(len(X))
	# print(len(Y))

	X = np.array(X).reshape(-1,4)
	Y = np.array(Y).reshape(-1,1)

	# print(test)
	# print(len(X_TEST)+len(Y_TEST))
	# print((len(X_TEST)+len(Y_TEST))/5)

	X_TEST = np.array(X_TEST).reshape(-1,4)
	Y_TEST = np.array(Y_TEST).reshape(-1,1)

	# print(X.shape)
	# print(Y.shape)

	# print(X_TEST.shape)
	# print(Y_TEST.shape)

	original = Y[:]


	# print(X)
	# print(Y)
		

	# # X =  1st - 4th day array
	# # #[[1]	[2]]
	# # Y = 5th day array


	x_scaler = MinMaxScaler()
	y_scaler = MinMaxScaler()
	Y = y_scaler.fit_transform(Y)
	X = x_scaler.fit_transform(X)

	X_TEST = x_scaler.fit_transform(X_TEST)
	Y_TEST = y_scaler.fit_transform(Y_TEST)

	# print(X.shape[0])

	# X_TEST = X[int(X.shape[0] - (int(X.shape[0])/4)):]
	# Y_TEST = Y[int(Y.shape[0] - (int(Y.shape[0])/4)):]

	# index = int(-(int(X.shape[0])/4))
	# index2 = int(-(int(Y.shape[0]) /4))

	

	# print(X)

	# for i in range(len(X)):

	# 	#every fourth week put into validation input

	# 	#every fourth friday put into validation output

	# X_TEST = X[int(X.shape[0] - (int(X.shape[0])/4)):]
	# Y_TEST = Y[int(Y.shape[0] - (int(Y.shape[0])/4)):]

	# index = int(-(int(X.shape[0])/4))
	# index2 = int(-(int(Y.shape[0]) /4))

	# X = X[:index]
	# Y = Y[:index2-1]

	# X = X[:index]
	# Y = Y[:index2]

	# print(len(X))
	# print(len(Y))

	print(len(X_TEST))
	# print(len(Y_TEST))

	model = Sequential()
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(10, activation='relu'))
	model.add(Dense(1, activation='linear'))

	model.compile(loss='mse', optimizer=keras.optimizers.Adam(lr=0.01))


	plot_model(model, to_file='model.png')

	model.fit(	X,
			 	Y, 
			 	epochs=1000, 
			 	batch_size=32, 
			 	verbose=0,
			 	validation_data=(X_TEST,Y_TEST))

	# score = model.evaluate(X, Y, verbose=1)
	score = model.evaluate(X_TEST, Y_TEST, verbose=0)


	print('Test loss:', score)
	# print('Test accuracy:', score)

	Y_PREDICT = model.predict(X_TEST, batch_size=32)

	Y_FIT = model.predict(X, batch_size=32)

	# latest_result = model.predict(x_scaler.transform(X_LATEST), batch_size=32)

	# print("previous prices: ", X_LATEST)
	# print("latest pred: {} | actual {}".format(y_scaler.inverse_transform(latest_result), Y_LATEST))


	pred = y_scaler.inverse_transform(Y_PREDICT)
	actual = y_scaler.inverse_transform(Y_TEST)

	fit = y_scaler.inverse_transform(Y_FIT)
	trainY = y_scaler.inverse_transform(Y)

	diff = pred - actual

	print("predicted | original | difference | accuracy")
	for i in range(len(Y_TEST)):
		print("{} | {} | {} | {}".format(pred[i], actual[i], diff[i] , (1-(abs(diff[i]/actual[i])))))

	accuracy = 0
	for i in range(len(Y_TEST)):
		accuracy += (1-(abs(diff[i]/actual[i])))

	accuracy /= len(Y_TEST)

	print("Overall accuracy: " ,accuracy)

	correct_direction = 0
	for i in range(len(Y_TEST)):
		if Y_PREDICT[i] > X_TEST[i][3] and Y_TEST[i] >  X_TEST[i][3]:
			correct_direction += 1
		elif Y_PREDICT[i] < X_TEST[i][3] and Y_TEST[i] <  X_TEST[i][3]:
			correct_direction += 1


	correct_direction_accuracy = (correct_direction / len(Y_TEST))

	print("Directional accuracy: " , correct_direction_accuracy)

	print(company)

	pl.clf()

	pl.plot(fit, label='fit')
	pl.plot(trainY, label='real')
	pl.savefig(company+"_train_"+".png")

	

	pl.clf()

	pl.plot(pred, label='prediction')
	pl.plot(actual, label='real')

	pl.xlabel('index')
	pl.ylabel('values')
	pl.legend()
	pl.savefig(company+"_test_"+".png")
	model.save(company+"_model_.h5")
	model = None
