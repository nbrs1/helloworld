import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import sklearn.metrics as metrics
import matplotlib.pyplot as plt
import random as rd 
from math import sqrt
from sklearn.datasets import make_classification
from matplotlib import pyplot as plt
from sklearn.linear_model import LogisticRegression
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.special import expit

sns.set()
#premiere etape: creer le dictionnaire a partir du xlsx
storage_data=pd.read_excel("storage_datarealone.xlsx",sheet_name=None)

print(storage_data["SF - UGS Peckensen"].columns)

#2e étape : Calculate a net withdrawal column
for key in storage_data:
	inj=storage_data[key]["injection"].values
	wit=storage_data[key]["withdrawal"].values
	l=[]
	for i in range(len(inj)):
		l.append(wit[i]-inj[i])
	storage_data[key]["NW"]=pd.DataFrame(l)
#3e étape : Calculate a lagged net withdrawal column of day prior
	l1=storage_data[key]["NW"].values
	l2=[0]
	for i in range(len(l1)-1):
		l2.append(l1[i])
	storage_data[key]["lagged_NW"]=pd.DataFrame(l2)
#Create a new Net Withdrawal column named Net Withdrawal_binary
	l1=storage_data[key]["NW"].values
	l2=[]
	for i in range(len(l1)):
		if l1[i]>0:
			l2.append(1)
		else :
			l2.append(0)
	storage_data[key]["Net Withdrawal_binary"]=pd.DataFrame(l2)
#FSW1 = max(Full Stock - 45, 0)
#FSW2 = max(45 - Full Stock, 0)
	l=storage_data[key]["full"].values
	l1=[]
	l2=[]
	for i in range(len(l)):
		l1.append(max(l[i]-45,0))
		l2.append(max(45-l[i],0))
	storage_data[key]["FSW1"]=pd.DataFrame(l1)
	storage_data[key]["FSW2"]=pd.DataFrame(l2)

price_data = pd.read_csv("price_data.csv", sep=";")
price_data.rename(columns={"Date": "gasDayStartedOn"}, inplace=True)	
#Part 1: Classification
#dictionnaries to put the results of the 2 models
Logistic_Regression={}
random_forest={}
#inner join
#print(storage_data["SF - UGS Peckensen"]["gasDayStartedOn"])
print(price_data["gasDayStartedOn"])
data={}
for key in storage_data:
	data[key]=pd.merge(storage_data[key],price_data, on="gasDayStartedOn")



# logistic regression
#Your y array is the Net Withdrawal_binary column
#Your X matrix is composed of the Lagged_NW, FSW1, FSW2 and all the time spreads price columns

for key in storage_data:
	y=np.array(storage_data[key]["Net Withdrawal_binary"].values)
	#X a définir
	x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=1)
	lr = LogisticRegression()
	Logi=lr.fit(x_train, y_train)
	y_pred = lr.predict(x_test)
	cm=confusion_matrix(y_test, y_pred)
	proba=lr.predict_proba(x_test)
	Logistic_Regression[key]={"recall": metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]), "confusion": cm,"precision": metrics.precision_score(y_test, y_pred),"neg_precision":cm[1,1]/cm.sum(axis=1)[1],"roc": metrics.roc_auc_score(y_test, probs)}
print(Logistic_Regression)													

#storage_data=pd.read_excel("storage_data(1).xlsx",sheet_name=None)

#print(storage_data['SF -UGS Rehden'])


