
# 01  regression, used to predict. source 01-08: @sentdex
# this code forecasts cost


import pandas as pd
import quandl
import math, datetime
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')
quandl.ApiConfig.api_key = '3yygLjxqGXzV1QmonzyW'
df = quandl.get('WIKI/GOOGL')
'''the next line prints the data in the line above'''
print(df.head())

'''we need these five rows in the df to make the calculations below'''
df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume']]
df['HL_PCT'] = (df['Adj. High']-df['Adj. Low'])/ df ['Adj. Low']* 100.0
df['PCT_CHANGE'] = (df['Adj. Close']-df['Adj. Open'])/ df ['Adj. Open']* 100.0

'''These are the columns in the df we actually need'''
df = df[['Adj. Close','HL_PCT','PCT_CHANGE','Adj. Volume']]
#print(df.head())

#Regression features and label
'''we will start the forecast here with forecast_col as input or feature'''
forecast_col = 'Adj. Close'
df.fillna(-99999,inplace = True)

forecast_out = int(math.ceil(0.01*len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)#- shifts the column up by 0.01 of df

#print(df.tail())# shows the last 5 rows of table
#print(df.head())

#Regression Training and Testing
'''this returns a new df feature dropping the label'''
X = np.array(df.drop(['label'], 1))
X = preprocessing.scale(X)
X_lately= X[-forecast_out:]# 1% of data
X = X[:-forecast_out]# 1st 99% of data


df.dropna(inplace = True)

y = np.array(df['label'])
'''lenght X must be = lenght y'''
print(len(X),len(y))

'''this is 20% of the data for testing'''
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2)
'''classifier is used for predictions'''
clf = LinearRegression(n_jobs=-1)#with n_jobs it runs in parallel, here -1 means it's running as many jobs as possible
'''we can switch the algorithm using either clf abv or below'''
#clf = svm.SVR()
'''we can add a parameter, say'''
#clf = svm.SVR(kernel='poly')
clf.fit(X_train, y_train)

'''for pickle'''
with open('linearregression.pickle','wb') as f:
    pickle.dump(clf, f)

'''if we comment everything above from clf = ... to pickle.dump...,the code still runs'''
'''to read the pickle'''
pickle_in = open('linearregression.pickle','rb')
'''we renamed classifier here'''
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
'''accuracy is squared error'''
print(accuracy)

#Regression forecasting and predicting
forecast_set=clf.predict(X_lately)
print(forecast_set, accuracy,forecast_out)
# to plot the values for these 35 days on a graph
df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day

'''we will populate our dataframe with the new date and forecast values'''
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]  # df.loc is the index of the dataframe
'''this sets the first column as nan and the final column is i(forecast). df.loc[next_date] makes the date index. 
The _ refers to the values in each column that are not numbers, so we have np.nan for them. then we add forecast(i) as
    our last column'''


print(df.head())
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
