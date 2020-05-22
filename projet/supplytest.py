import unittest
import supply as sp
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

class valueTest(unittest.TestCase):
	knownvalues=[[2,0,pd.to_datetime(13/10/2013)],[3,0,pd.to_datetime(8/10/2013)]]
	storage_data=pd.read_excel("storage_datarealone.xlsx",sheet_name=None)
	price_data = pd.read_csv("price_data.csv", sep=";")
	price_data.rename(columns={"Date": "gasDayStartedOn"}, inplace=True)	
	price_data["gasDayStartedOn"]=pd.to_datetime(price_data["gasDayStartedOn"])
	def test_importation_data(self):
		for i,j,real_value in self.knownvalues:
			result=sp.import_price()
			res=result.values
			self.assertEqual(res[i][j],real_value)
	def test_NWandLaggedNW(self):
		classif=sp.classification(self.storage_data,self.price_data)
		NW=classif.calcul_NW().values
		lagged=classif.calcul_lagged_NW().values
		for i in range(50):
			self.assertEqual(lagged[i+1],NW[i])

	def test_binaryNW(self):
		classif=sp.classification(self.storage_data,self.price_data)
		BNW=classif.binaryNW().values
		for i in range(len(BNW)):
			self.assertEqual(BNW[i],0) or self.assertEqual(BNW[i],1)

class metricsTest(unittest.TestCase):
	storage_data=pd.read_excel("storage_datarealone.xlsx",sheet_name=None)
	price_data = pd.read_csv("price_data.csv", sep=";")
	price_data.rename(columns={"Date": "gasDayStartedOn"}, inplace=True)	
	price_data["gasDayStartedOn"]=pd.to_datetime(price_data["gasDayStartedOn"])
	def test_valeursEntre0et1(self):
		classif=
		
if __name__ == '__main__':
    unittest.main()