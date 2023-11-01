# -*- coding: utf-8 -*-
"""
Supervised Machine Learning Exercise using Scikit Learn

"""
import pandas as pd
from  matplotlib import pyplot as plt

from copy import deepcopy

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

import pickle

#Input variables
n_params = 10
n_objectives = 1

#Set the file path
filePath = "C:\\Users\\harri\\OneDrive\\Documents\\ITECH2020\\Academic Materials\\21_Som Sem\\3064044_Computing in Architecture\\2_Assignments\\_Final\\_Final Logs\\Supervised ML\\SOO\\"
fileName = "dataset.csv"

#Read h=the file path
dataset = pd.read_csv(filePath + fileName)
print(dataset.head)

#Analyze the features using histogram
#dataset.hist()
#plt.show()

#Create a separate dataset for parameters
parameters_dataset = deepcopy(dataset)
for i in range(n_objectives):
    parameters_dataset.drop(parameters_dataset.columns[-1], axis=1, inplace=True)

print(parameters_dataset.head)

#Create a separate dataset for objectives
objectives_dataset = deepcopy(dataset)
for i in range(n_params):
    objectives_dataset.drop(objectives_dataset.columns[0], axis=1, inplace=True)

print(objectives_dataset.head)

# perform a robust scaler transform of the paramatersdataset
sc = StandardScaler()
std_params_data = sc.fit_transform(parameters_dataset)
std_parameters_dataset = pd.DataFrame(std_params_data)
std_parameters_dataset.columns = parameters_dataset.columns

#Save the standard scaler model
pickle.dump(sc, open(filePath + 'scaler_model.pkl','wb'))


#std_parameters_dataset.hist()
#plt.show()

print(std_parameters_dataset.head)

# Split data for training and testing
X_train, X_test, Y_train, Y_test = train_test_split(std_parameters_dataset, objectives_dataset, test_size=0.33, random_state=5)

######### Polynomial Linear Regression #########
#Transform the parameters by adding polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X_train)
poly.fit(X_poly, Y_train)

#Do the regression
reg_model = LinearRegression()
reg_model.fit(X_poly, Y_train)

#Evaluate the polynomial linear regression model with a cross validation
cv_score_1 = cross_val_score(reg_model, X_poly, Y_train, cv=10).mean()

#Test the model on the optimized model
Y_predict_poly = reg_model.predict(poly.fit_transform(X_test))


######### Support Vector Regression #########
svr_model = SVR(kernel='poly', degree=2)
multi_svr_model = MultiOutputRegressor(svr_model)
multi_svr_model.fit(X_train, Y_train)

#Evaluate the multi-output support vector regression model with a cross validation
cv_score_2 = cross_val_score(multi_svr_model, X_train, Y_train, cv=10).mean()

#Test the model on the optimized model
Y_predict_svr = multi_svr_model.predict(X_test)

#Save the pretrained model with the best accuracy
model_fileName = 'prediction_model.pkl'
pickle.dump(reg_model, open(filePath + model_fileName, 'wb'))