# 17  Mean Shift with titanic dataset..from Custom means,source @sentdex

import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')
import numpy as np
from sklearn.cluster import MeanShift
from sklearn import preprocessing, model_selection
import pandas as pd

#this fixes the print headache pycharm has been giving me
desired_width = 320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_columns', 15)
np.set_printoptions(linewidth=desired_width)

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
original_df = pd.DataFrame.copy(df)
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

#add or remove features just to see the effects they have
#df.drop(['ticket','home.dest'],1, inplace = True)

X = np.array(df.drop(['survived'], 1).astype(float))# this is dropped because this is what we are predicting.
X = preprocessing.scale(X)
y = np.array(df['survived'])

clf = MeanShift()
clf.fit(X)

labels = clf.labels_
cluster_centers = clf.cluster_centers_

original_df['cluster_group'] = np.nan #this adds a new column to the original_df
#we populate the values of the new column above
for i in range(len(X)):
    original_df['cluster_group'].iloc[i] = labels[i] #with iloc we reference the index of the row of the dataset. so we say
    # the ith row of the original_df under the cluster_group column, we set it to the value of the label of that row
n_clusters_ = len(np.unique(labels))

survival_rates = {} # the key is the cluster group and value is the number
for i in range(n_clusters_):
    temp_df = original_df[(original_df['cluster_group'] ==float(i))] #
    survival_cluster = temp_df[ (temp_df['survived']==1)]
    survival_rate = len(survival_cluster)/len(temp_df)
    survival_rates[i] = survival_rate

print(survival_rates)
print(10*'########## ')
print(original_df[ (original_df['cluster_group']==1)])
print(10*'########## ')
print(original_df[ (original_df['cluster_group']==2)])
print(10*'########## ')
print(original_df[ (original_df['cluster_group']==3)])
print(10*'########## ')
print(original_df[ (original_df['cluster_group']==0)])
print(original_df[ (original_df['cluster_group']==0)].describe())
 #to find the survival rate of first class passengers in cluster 0, we say
cluster_0 = original_df[ (original_df['cluster_group']==0)]
cluster_0_fc = cluster_0[(cluster_0['pclass']==1)]
print(cluster_0_fc.describe())