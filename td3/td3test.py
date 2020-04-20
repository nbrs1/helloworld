import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random as rd 
import td3
import unittest


class valuetest(unittest.TestCase):
	knownvalues=[[2,1,653.5600356],[9,2,7.132067932]]
	DE = pd.read_csv("DE.csv", sep=";")
	DE.columns=['Date','LDZ','actual','normal']
#fig=plt.figure()
	y1=DE['LDZ']
	y2=DE['actual']
	y3=DE['normal']
	x=DE['Date']
	def test_importation_data(self):
		for i,j,real_value in self.knownvalues:
			result=td3.import_csv("DE.csv",";",False)
			res=result.values
			self.assertAlmostEqual(res[i][j],real_value)
	"""def test_metrics(self):
		 créé deux listes au hasard on teste si elles ne sont pas corrélé, 
		vis versa on créé deux liste identiques et on test si elles sont bien correlées
		l1=[]
		l2=[]
		for i in range(100):
			x=rd.random()
			y=rd.random()
			l1.append(x)
			l2.append(y)
		metrics1=td3.get_fit_metrics(l1,l2)
		metrics2=td3.get_fit_metrics(l1,l1)
		self.assertAlmostEqual(metrics1[0]**2,0)
		self.assertAlmostEqual(metrics2[0]**2,1)"""
	def test_consumption_sigmoid(self):
		t=np.linspace(0,100,1000)
		h=td3.consumption_sigmoid(t,self.y1,900,-35,12,400,False)
		for i in range(len(t)):
			self.assertEqual(h[i],td3.h(t[i],900,-35,12,400))


if __name__ == '__main__':
    unittest.main()

