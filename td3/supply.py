import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random as rd 
from math import sqrt



#premiere etape: creer le dictionnaire a partir du xlsx

dictionnaire={}
dictionnaire["SF -UGS Rehden"]=pd.read_csv("storage_data.csv", sep=",")
dictionnaire["SF -UGS Kraak"]=pd.read_csv("storage_data1.csv", sep=",")
dictionnaire["SF -UGS Stassfurt"]=pd.read_csv("storage_data2.csv", sep=",")
dictionnaire["SF -UGS Harsefeld"]=pd.read_csv("storage_data3.csv", sep=",")
dictionnaire["SF -UGS Breitbrunn"]=pd.read_csv("storage_data4.csv", sep=",")
dictionnaire["SF -UGS Epe Uniper H-Gas "]=pd.read_csv("storage_data5.csv", sep=",")
dictionnaire["SF -UGS Eschenfelden"]=pd.read_csv("storage_data6.csv", sep=",")
dictionnaire["SF -UGS Inzenham-West"]=pd.read_csv("storage_data7.csv", sep=",")
dictionnaire["SF -UGS Bierwang"]=pd.read_csv("storage_data8.csv", sep=",")
dictionnaire["SF -UGS Jemgum H (EWE)"]=pd.read_csv("storage_data9.csv", sep=",")
dictionnaire["SF -UGS Peckensen"]=pd.read_csv("storage_data10.csv", sep=",")
dictionnaire["SF -UGS Etzel ESE (Uniper energy) "]=pd.read_csv("storage_data11.csv", sep=",")


dictionnaire["SF -UGS Rehden"]['Lagged-NW']=[0 for i in range(2323)]

for cle in dictionnaire() :		#parcours du dictionnaire
	dictionnaire[cle]["Lagged-NW"]=dictionnaire[cle]['withdrawal']-dictionnaire[cle]['injection']
	

print(dictionnaire["SF -UGS Rehden"]['withdrawal'])