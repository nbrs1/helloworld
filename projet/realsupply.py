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
import datetime

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.special import expit

sns.set()
#stock données
storage_data=pd.read_excel("storage_datarealone.xlsx",sheet_name=None)
#time spread données
price_data = pd.read_csv("price_data.csv", sep=";")
price_data.rename(columns={"Date": "gasDayStartedOn"}, inplace=True)	
price_data["gasDayStartedOn"]=pd.to_datetime(price_data["gasDayStartedOn"])

#dictionnaire ou l'on met nos résultats
Logistic_Regression={}
random_forest={}

#boucle ou l'on travail sur les données
for key in storage_data:
#calcul NW
	inj=storage_data[key]["injection"].values
	wit=storage_data[key]["withdrawal"].values
	l=[]
	for i in range(len(inj)):
		l.append(wit[i]-inj[i])
	storage_data[key]["NW"]=pd.DataFrame(l)
#calcul lagged_NW
	l1=storage_data[key]["NW"].values
	l2=[0]
	for i in range(len(l1)-1):
		l2.append(l1[i])
	storage_data[key]["lagged_NW"]=pd.DataFrame(l2)
#Colonne binaire
	l1=storage_data[key]["NW"].values
	l2=[]
	for i in range(len(l1)):
		if l1[i]>0:
			l2.append(1)
		else :
			l2.append(0)
	storage_data[key]["Net Withdrawal_binary"]=pd.DataFrame(l2)
#calcul FSW1 = max(Full Stock - 45, 0) et FSW2 = max(45 - Full Stock, 0)
	l=storage_data[key]["full"].values
	l1=[]
	l2=[]
	for i in range(len(l)):
		l1.append(max(l[i]-45,0))
		l2.append(max(45-l[i],0))
	storage_data[key]["FSW1"]=pd.DataFrame(l1)
	storage_data[key]["FSW2"]=pd.DataFrame(l2)
#jointure avec le time spread
	storage_data[key]=storage_data[key].merge(price_data, left_on="gasDayStartedOn", right_on="gasDayStartedOn")
# logistic regression
#X matrix is composed of the Lagged_NW, FSW1, FSW2 and all the time spreads price columns 
	y=np.array(storage_data[key]["Net Withdrawal_binary"].values)
	x=np.array([storage_data[key]["gasDayStartedOn"].values,storage_data[key]["lagged_NW"].values,storage_data[key]["FSW1"].values,storage_data[key]["FSW2"].values,storage_data[key]["SAS_GPL"].values,storage_data[key]["SAS_TTF"].values,storage_data[key]["SAS_NCG"].values,storage_data[key]["SAS_NBP"].values])
	x=x.transpose()
	x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
	lr = LogisticRegression()
	Logi=lr.fit(x_train, y_train)
	y_pred = lr.predict(x_test)
	cm=confusion_matrix(y_test, y_pred)
	proba=lr.predict_proba(x_test)
	Logistic_Regression[key]={"recall": metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]), "confusion": cm,"precision": metrics.precision_score(y_test, y_pred),"neg_precision":cm[1,1]/cm.sum(axis=1)[1],"roc": metrics.roc_auc_score(y_test, proba),"class_mod": Logi}
	print(Logistic_Regression[key])													
