
# 09 Applying and testing the code in 08 on the breast cancer dataset. compare this accuracy with 06.source @sentdex

import numpy as np
from math import sqrt
from collections import Counter
import warnings
import pandas as pd
import random

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than the total voting groups!')
    #we start comparing the new point with other points to find it's class
    distances = []
    for group in data:
        for features in data[group]:
            #euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2) or for speed
            euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
            distances.append([euclidean_distance, group])
            #print(distances)
    votes = [i[1] for i in sorted(distances) [:k]]
    print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1]/k
    #print(vote_result, confidence)
    return vote_result, confidence

accuracies = []
for i in range(5):
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    #print(df.head())
    full_data = df.astype(float).values.tolist()
    #print(full_data[ :10])
    random.shuffle(full_data)
    #print(20*'#') is used for demarcation
    #print(full_data[ :10])

    test_size = 0.2
    train_set = {2:[], 4:[]}#2 benign, 4 malignant
    test_set =  {2:[], 4:[]}
    train_data = full_data[:-int(test_size*len(full_data))]#everything up to the last 20% of data
    test_data = full_data[-int(test_size*len(full_data)):]# the last 20% of data
    # to populate the dictionary at line 39,40
    for i in train_data:
        train_set[i[-1]].append(i[:-1])#-1 here refers to the last column ie the class column ie 2 or 4
    for i in test_data:
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total =0
    for group in test_set:
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct +=1
            else:
                print(confidence)#gives the confidence scores of the votes we had incorrect eg 1.0 means 100% were
                # incorrect. comment line 27 to see just this
            total +=1
    print('Accuracy:', correct/total)
    accuracies.append(correct/total)

print(sum(accuracies)/len(accuracies))

# Observation :
#Increasing the value of k here doesnt increase accuracy.
# Accuracy is how right we got the classification. confidence here is the number of votes
# To check the average of accuracies for a number of tests,say 5, add lines for accuracies. we add this to sklearn KNN
# 06, we observe the higher the number of tests,say 25,the slower this code performance. sklearn KNN is faster.
# KNN works on linear and non linear data. Regression can only classify linear data