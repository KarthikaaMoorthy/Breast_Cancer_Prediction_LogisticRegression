#import libraries
import numpy as np
import sklearn.datasets

#Datasets
breast_cancer = sklearn.datasets.load_breast_cancer()
print(breast_cancer)
X = breast_cancer.data
Y = breast_cancer.target
print(X)
print(Y)
print(X.shape, Y.shape)
#import data to the pandas data frame
import pandas as pd
data = pd.DataFrame(breast_cancer.data, columns = breast_cancer.feature_names)
data['class'] = breast_cancer.target
data.head()
data.describe()
print(data['class'].value_counts())
print(breast_cancer.target_names)
#split the dataset into train and test data set
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test =  train_test_split(X, Y)
print(Y.shape, Y_train.shape, Y_test.shape)
#test size == to specify the percentage of test data model
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
print(Y.shape, Y_train.shape, Y_test.shape)
print(Y.mean(), Y_train.mean(), Y_test.mean())
# stratify -- correct distribution of data as of the original data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y)
print(Y.mean(), Y_train.mean(), Y_test.mean())
#random_state -- specific split of data each value of random_state splits the data differently
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, stratify=Y, random_state=1)
print(X_train.mean(), X_test.mean(), X.mean())
print(X_train)
#Logistic Regression\n",
from sklearn.linear_model import LogisticRegression
#Loading the logistic regression model to the variable 'classifier'
classifier = LogisticRegression()
#training the model on training data
classifier.fit(X_train, Y_train)
#Evaluation of the model
#import accuracy_score
from sklearn.metrics import accuracy_score
#prediction on train data\n",
prediction_on_training_data = classifier.predict(X_train)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)
print("Accuracy on training data:", accuracy_on_training_data)
#prediction on test data\n",
prediction_on_test_data = classifier.predict(X_test)
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)
print("Accuracy on test data: ", accuracy_on_test_data)
#detecting whether the patient has breast cancer in benign or malignant stage
input_data = (13.03,18.42,82.61,523.8,0.08983,0.03766,0.02562,0.02923,0.1467,0.05863,0.1839,2.342,1.17,14.16,0.004352,0.004899,0.01343,0.01164,0.02671,0.001777,13.3,22.81,84.46,545.9,0.09701,0.04619,0.04833,0.05013,0.1987,0.06169)
#change the input data to numpy_array to make prediction
input_data_as_numpy_array = np.asarray(input_data)
print(input_data)
#reshape the array as we are predicting the output for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
#prediction
prediction = classifier.predict(input_data_reshaped)
#returns a list with element [0] if Malignant; returns a list with element [1], if benign.
print(prediction)
if(prediction[0] == 0):
    print("The breast Cancer is Malignant")
else:
    print("The breast Cancer is Benign")
