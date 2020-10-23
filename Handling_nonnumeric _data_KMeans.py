#14 Handling non numeric data. We want to see how to treat non numeric data since our machine learning
# does not understand non numeric data. find better methods to do this, source @sentdex
#https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd

''' Pclass Passenger Class (1= 1st; 2= 2nd; 3= 3rd)
survival  Survival(0=No; 1=Yes)
name Name
sex Sex
age Age
sibsp Number of siblings/spouses aboard
parch Number of parents/children aboard
ticket Ticket Number
fare Passenger Fare (British pound)
cabin Cabin
embarked Port of embarkation (C= Cherbourg, Q=Queenstown, S=Southampton)
boat Lifeboat
body Body Identification Number
home.dest Home/Destination
'''

df = pd.read_excel('titanic.xls')
#print(df.head())
df.drop(['body', 'name'], 1, inplace = True)
df.apply(pd.to_numeric, errors = 'ignore')
df.fillna(0, inplace = True)
#print(df.head())

def handle_non_numerical_data(df):
    columns = df.columns.values

    for column in columns:
        text_digit_vals = {}#Eg here we could have {'Female' : 0, etc}, it converts it thus
        def convert_to_int(val):
            return text_digit_vals[val]

        if df[column].dtype != np.int64 and df[column].dtype != np.float64:#means if the value is not a number
            column_contents = df[column].values.tolist()#we convert to list, get the set of the list
            unique_elements = set(column_contents)#then take the unique elements and populate the dict
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique] = x
                    x+=1

            df[column] = list(map(convert_to_int, df[column]))#we reset the values in df[column] by mapping the fns of
            #values in first parameter to the 2nd parameter
    return df
df = handle_non_numerical_data(df)
print(df.head())


# KMeans with titanic dataset
# we want to know the likelihood of survival based on the information in the data. after we have trained with a clf,
# then we can predict using .predict  or svm or NN
# we wont use model selection because this is unsupervised ML and it wont let us test against all the data.
# from #13 Clustering Intro.py


X = np.array(df.drop(['survived'], 1).astype(float))# this is dropped because this is what we are predicting.
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = KMeans(n_clusters=2)
clf.fit(X)

correct = 0
for i in range(len(X)):
    #this is only used with unsupervised learning
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction[0] == y[i]:
        correct += 1

print(correct/len(X))#if we get from 10-30%, its the reciprocal, as long as its consistent
# we can drop columns to see how each affects the accuracy.

