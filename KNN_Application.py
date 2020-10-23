
# 06 KNN is used to classify data by determining the distance between the the unknown point to any of the other k
# points ,to determine its class. Here, we are classifying the cancer of patients as either benign or malignant
# from data collected from them. This code is written using scikitlearn,source @sentdex

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd

accuracies = []
for i in range(5):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)

    X = np.array(df.drop(['class'], 1))
    y = np.array(df['class'])

    X_train, X_test, y_train, y_test = model_selection.train_test_split(X,y,test_size=0.2)

    clf=neighbors.KNeighborsClassifier(n_jobs=1)#this runs parallel jobs,speeding up performance
    clf.fit(X_train,y_train)

    accuracy=clf.score(X_test,y_test)
    print(accuracy)

    '''to predict'''
    example_measures=np.array([4,2,1,1,1,2,3,2,1])# this dataset is unique and isnt in our data .txt file
    example_measures=example_measures.reshape(1, -1)#the first value in the () reps dimensions.if we have 2 sample
    # datasets, the 1 in the reshape becomes 2. For unknown number of datasets,the 1 will be len(example_measures)
    prediction= clf.predict(example_measures)
    print(prediction)#with the answer as 2, it means this sample is from a patient whose cancer is benign

    accuracies.append(accuracy)
print(sum(accuracies)/len(accuracies))
