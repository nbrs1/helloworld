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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from scipy.special import expit
from sklearn.linear_model import LinearRegression

def set_wd():
    sns.set()
#stock données
def import_storage_data():
	storage_data=pd.read_excel("storage_datarealone.xlsx",sheet_name=None)
	return storage_data
#time spread données
def import_price():
	price_data = pd.read_csv("price_data.csv", sep=";")
	price_data.rename(columns={"Date": "gasDayStartedOn"}, inplace=True)	
	price_data["gasDayStartedOn"]=pd.to_datetime(price_data["gasDayStartedOn"])
	return price_data
	
class classification:
	def __init__(self,storage_data,price_data):
		self.storage_data=storage_data
		self.price_data=price_data
#les 5 premieres fonctions ajoutent des colonnes au tableau, et inner join avec time spread
	def calcul_NW(self):
		for key in self.storage_data:
			inj=self.storage_data[key]["injection"].values
			wit=self.storage_data[key]["withdrawal"].values
			l=[]
			for i in range(len(inj)):
				l.append(wit[i]-inj[i])
			self.storage_data[key]["NW"]=pd.DataFrame(l)
		return self.storage_data

	def calcul_lagged_NW(self):
		self.storage_data=self.calcul_NW()
		for key in self.storage_data:
			l1=self.storage_data[key]["NW"].values
			l2=[0]
			for i in range(len(l1)-1):
				l2.append(l1[i])
			self.storage_data[key]["lagged_NW"]=pd.DataFrame(l2)
		return self.storage_data

	def binaryNW(self):
		self.storage_data=self.calcul_lagged_NW()
		for key in self.storage_data:
			l1=self.storage_data[key]["NW"].values
			l2=[]
			for i in range(len(l1)):
				if l1[i]>0:
					l2.append(1)
				else :
					l2.append(0)
			self.storage_data[key]["Net_Withdrawal_binary"]=pd.DataFrame(l2)
		return self.storage_data

	def fsw1fsw2(self):
		self.storage_data=self.binaryNW()
		for key in self.storage_data:
			l=self.storage_data[key]["full"].values
			l1=[]
			l2=[]
			for i in range(len(l)):
				l1.append(max(l[i]-45,0))
				l2.append(max(45-l[i],0))
			self.storage_data[key]["FSW1"]=pd.DataFrame(l1)
			self.storage_data[key]["FSW2"]=pd.DataFrame(l2)
		return self.storage_data

	def innerjoin(self):
		self.storage_data=self.fsw1fsw2()
		for key in self.storage_data:
			self.storage_data[key]=self.storage_data[key].merge(self.price_data, \
				left_on="gasDayStartedOn", right_on="gasDayStartedOn")
		return self.storage_data

	def predictionlogreg(self,key):
		self.storage_data[key] = self.storage_data[key].dropna()
		y=self.storage_data[key]["Net_Withdrawal_binary"].to_numpy()
		x  = self.storage_data[key].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
		lr=LogisticRegression().fit(x_train, y_train)
		y_pred = lr.predict(x_test)
		return y_pred

	def predictionrandforest(self,key):
		self.storage_data[key] = self.storage_data[key].dropna()
		y=self.storage_data[key]["Net_Withdrawal_binary"].to_numpy()
		x  = self.storage_data[key].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
		rf = RandomForestClassifier().fit(x_train, y_train)
		y_pred = rf.predict(x_test)
		return y_pred
#retourne dictionnaire de dictionnaire de metrics pour logistic regression
	def metricslogreg(self):		
		Logistic_Regression={}
		for key in self.storage_data:
			self.storage_data[key] = self.storage_data[key].dropna()
			y=self.storage_data[key]["Net_Withdrawal_binary"].to_numpy()
			x  = self.storage_data[key].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
			x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
			lr=LogisticRegression().fit(x_train, y_train)
			y_pred = lr.predict(x_test)
			cm=confusion_matrix(y_test, y_pred)
			proba=lr.predict_proba(x_test)[:,1]
			Logistic_Regression[key]={"recall": metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]),\
			 "confusion": cm,"precision": metrics.precision_score(y_test, y_pred),\
			 "neg_precision":cm[1,1]/cm.sum(axis=1)[1],"roc": metrics.roc_auc_score(y_test, proba),"class_mod": lr}
		return Logistic_Regression
#retourne dictionnaire de dictionnaire de metrics pour random forest
	def metricsrandforest(self):		
		random_forest={}
		for key in self.storage_data:
			self.storage_data[key] = self.storage_data[key].dropna()
			y=self.storage_data[key]["Net_Withdrawal_binary"].to_numpy()
			x  = self.storage_data[key].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
			x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
			rf = RandomForestClassifier().fit(x_train, y_train)
			y_pred = rf.predict(x_test)
			cm=confusion_matrix(y_test, y_pred)
			proba=rf.predict_proba(x_test)[:,1]
			random_forest[key]={"recall": metrics.recall_score(y_test, y_pred), "neg_recall": cm[1,1]/(cm[0,1] + cm[1,1]),\
			 "confusion": cm,"precision": metrics.precision_score(y_test, y_pred),\
			 "neg_precision":cm[1,1]/cm.sum(axis=1)[1],"roc": metrics.roc_auc_score(y_test, proba),"class_mod": rf}
		return random_forest
#compare les 2 modèles pour un storage en particulier, 1=victoire log reg, 0=égalité, -1=victoire rand forest
	def compare_lr_rf_OnOneStorage(self, key):
		lr=self.metricslogreg()		
		rf=self.metricsrandforest()		
		lr=lr[key]		
		rf=rf[key]
		del lr["confusion"]
		del lr["class_mod"]
		del rf["confusion"]
		del rf["class_mod"]
		successlr=0
		successrf=0

		for cle in lr:
			
			if lr[cle]>rf[cle]:
				successlr+=1
			if lr[cle]<rf[cle]:
				successrf+=1
		if successrf<successlr:			
			return 1
		if successlr==successrf:			
			return 0
		else:
			return -1

	def compare_lr_rf_globally(self):
		victory=0
		for key in self.storage_data:
			victory=victory+self.compare_lr_rf_OnOneStorage(key)
		if victory>0:
			print("victoire log reg")
			return victory
		if victory==0:
			print("égalité")
			return victory
		if victory<0:
			print("victoire rand forest")
			return victory

class regression:
	def __init__(self,storage_data,price_data):
		self.storage_data=storage_data
		self.price_data=price_data
#les 5 premieres fonctions ajoutent des colonnes au tableau, et inner join avec time spread
	def calcul_NW(self):
		for key in self.storage_data:
			inj=self.storage_data[key]["injection"].values
			wit=self.storage_data[key]["withdrawal"].values
			l=[]
			for i in range(len(inj)):
				l.append(wit[i]-inj[i])
			self.storage_data[key]["NW"]=pd.DataFrame(l)
		return self.storage_data

	def calcul_lagged_NW(self):
		self.storage_data=self.calcul_NW()
		for key in self.storage_data:
			l1=self.storage_data[key]["NW"].values
			l2=[0]
			for i in range(len(l1)-1):
				l2.append(l1[i])
			self.storage_data[key]["lagged_NW"]=pd.DataFrame(l2)
		return self.storage_data

	def binaryNW(self):
		self.storage_data=self.calcul_lagged_NW()
		for key in self.storage_data:
			l1=self.storage_data[key]["NW"].values
			l2=[]
			for i in range(len(l1)):
				if l1[i]>0:
					l2.append(1)
				else :
					l2.append(0)
			self.storage_data[key]["Net_Withdrawal_binary"]=pd.DataFrame(l2)
		return self.storage_data

	def fsw1fsw2(self):
		self.storage_data=self.binaryNW()
		for key in self.storage_data:
			l=self.storage_data[key]["full"].values
			l1=[]
			l2=[]
			for i in range(len(l)):
				l1.append(max(l[i]-45,0))
				l2.append(max(45-l[i],0))
			self.storage_data[key]["FSW1"]=pd.DataFrame(l1)
			self.storage_data[key]["FSW2"]=pd.DataFrame(l2)
		return self.storage_data

	def innerjoin(self):
		self.storage_data=self.fsw1fsw2()
		for key in self.storage_data:
			self.storage_data[key]=self.storage_data[key].merge(self.price_data, \
				left_on="gasDayStartedOn", right_on="gasDayStartedOn")
		return self.storage_data

	def rawsWithWithdrawal(self):
		self.storage_data=self.innerjoin()
		for key in self.storage_data:
			self.storage_data[key] = self.storage_data[key][self.storage_data[key].Net_Withdrawal_binary != 0]
		return self.storage_data
			
	def metricsregressionmodel(self):
		self.storage_data=self.rawsWithWithdrawal()
		regression_model={}
		for key in self.storage_data:
			self.storage_data[key] = self.storage_data[key].dropna()
			y=self.storage_data[key]["NW"].to_numpy()
			x  = self.storage_data[key].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
			x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
			regressor = LinearRegression().fit(x_train, y_train)
			y_pred = regressor.predict(x_test)
			if(len(y_pred) != len(y_test)):
				print("Difference in length between Fit and Real Consumption vectors")
			else:
      			 #somme des carrés des écarts sur longueur de la liste
				s=0
				for i in range(len(y_pred)):
					s+=((y_pred[i]-y_test[i])**2)/len(y_pred)
        #moyenne exp
				avr=0
				for i in range(len(y_pred)):
					avr+=y_test[i]/len(y_pred)
        #moyenne sigmoid
				avh=0
				for i in range(len(y_pred)):
					avh+=y_pred[i]/len(y_pred)
				shr=0
				for i in range(len(y_pred)):
					shr+=(y_pred[i]-avh)*(y_test[i]-avr)
				s2=0
				for i in range(len(y_pred)):
					s2+=(y_pred[i]-avh)**2
				sh=sqrt(s2)
				s3=0
				for i in range(len(y_pred)):
					s3+=(y_test[i]-avr)**2
				sr=sqrt(s3)

         #self.__corr, self.__rmse, self.__nrmse, self.__anrmse 
				corr=shr/(sh*sr)
				rmse=sqrt(s)
				nrmse=rmse/avr
				anrmse=abs(nrmse)
				regression_model[key]={'r2': metrics.r2_score(y_test, y_pred), 'rmse': rmse, 'nrmse': nrmse, \
			'anrmse': anrmse, 'cor': corr, 'l_reg': regression}
		return regression_model

	def prediction(self,storage):
		self.storage_data=self.rawsWithWithdrawal()
		self.storage_data[storage] = self.storage_data[storage].dropna()
		y=self.storage_data[storage]["NW"].to_numpy()
		x  = self.storage_data[storage].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
		regressor = LinearRegression().fit(x_train, y_train)
		y_pred = regressor.predict(x_test)
		return y_pred

	def droitepredite(self, storage):
		self.storage_data=self.rawsWithWithdrawal()
		self.storage_data[storage] = self.storage_data[storage].dropna()
		y=self.storage_data[storage]["NW"].to_numpy()
		x  = self.storage_data[storage].loc[:,["lagged_NW","FSW1","FSW2","SAS_GPL","SAS_TTF","SAS_NCG","SAS_NBP"]].to_numpy()
		x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)
		regressor = LinearRegression().fit(x_train, y_train)
		y_pred = regressor.predict(x_test)
		b=regressor.intercept_
		a=regressor.coef_
		return [a,b] 

			
if __name__ == '__main__':
	set_wd()
	storage_data=import_storage_data()
	price_data=import_price()
	#classification=classification(storage_data,price_data)
	#classification.innerjoin()
	#print(classification.predictionrandforest("SF - UGS Peckensen"))
	reg=regression(storage_data,price_data)
	print(reg.droitepredite("SF - UGS Peckensen")[1])

