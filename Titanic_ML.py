# -*- coding: utf-8 -*-
"""
Titanic: Machine Learning from Disaster

This is a first stab at setting up a machine learning model.
"""

import pandas as pd
from pandas.plotting import scatter_matrix #used for?
import numpy as np
from matplotlib import pyplot as plt
from sklearn import model_selection #used for?
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Step 1: Import data from csv files using pandas
file_location = '/Users/changjeremy/documents/python/titanic/train.csv'
data = pd.read_csv(file_location)
print(data) # use print as a first pass visualizatio of data


# Step 2: "Get a feel" for the dataset

print(data.shape) #shape function of dataframes gets the dimensions of the data

print(data.head(10)) #view the datal; head looks at the first __ rows

print(data.describe()) #summary function gives an overhead statistical summary

# Step 3: Evaluate distribution 
print(data.groupby('Sex').size())

#What to do when too many columns/rows? 
# Is American Ninja Warrior cultural appropriation?

# Step 4: Data Visualization
# Univariate Plots
data.plot(kind = 'box', subplots = True, layout = (4,3), sharex=False, sharey=False)

# Histogram
data.hist()

#scatter plots
scatter_matrix(data)


# plt.plot(data.Survived, data.Fare)
males = data[data.Sex == 'male']
females = data[data.Sex == 'female']
m_survivors = sum(males.Survived == 1) #109 males survived
f_survivors = sum(females.Survived == 1) #233 females survived
# conclusion: more females survived than males
plt.plot(data.Age, data.Fare)