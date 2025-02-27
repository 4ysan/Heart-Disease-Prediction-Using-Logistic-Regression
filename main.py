import pandas as pd
import numpy as np
import pylab as pl
import scipy.optimize as opt
import statsmodels.api as sm
from sklearn import preprocessing
'exec(% matplotlib inline)'
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn as sns

#loading the dataset
disease_df = pd.read_csv("framingham.csv")
disease_df.drop(['education'], axis=1, inplace=True)
disease_df.rename(columns = {'male':'Sex_male'},inplace=True)

#cleaning the dataset
disease_df.dropna(axis = 0, inplace = True)
print(disease_df.head(), disease_df.shape)
print(disease_df.TenYearCHD.value_counts())

#splitting the dataset into features and targeted values
x  = np.asarray(disease_df[['age', 'Sex_male', 'cigsPerDay',
                           'totChol', 'sysBP', 'glucose']])
y = np.asarray(disease_df['TenYearCHD'])

#normalization of the datset
x = preprocessing.StandardScaler().fit(x).transform(x)

#splitting the dataset into test data and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.3,random_state=4)

print('train test:',x_train.shape,y_train.shape)
print('test train:',x_test.shape,y_test.shape)

# counting no. of patients affected with CHD
import matplotlib
matplotlib.use('TkAgg')
plt.figure(figsize=(7, 5))
sns.countplot(x='TenYearCHD', data=disease_df,
             palette="pastel",hue='TenYearCHD')
plt.show()

#creating a line plot
laste = disease_df['TenYearCHD'].plot()
plt.show()

#applying logistic regression model on the dataset
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)

#getting the accuracy of the model
from sklearn.metrics import accuracy_score
print('Accuracy of the model is =',
      accuracy_score(y_test, y_pred))

#getting the sensetivity of the model
from sklearn.metrics import recall_score
sensitivity = recall_score(y_test, y_pred)
print('Sensitivity (Recall) of the model is =', sensitivity)

#confusion matrix
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_test, y_pred)
conf_matrix = pd.DataFrame(data = cm,
                           columns = ['Predicted:0', 'Predicted:1'],
                           index =['Actual:0', 'Actual:1'])

plt.figure(figsize = (8, 5))
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = "pink")

plt.show()
print('The details for confusion matrix is =')
print (classification_report(y_test, y_pred))