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

storage_data=pd.read_excel("storage_data(1).xlsx",sheet_name=None)

print(storage_data['SF -UGS Rehden'])

#for cle in storage_data() :		#parcours du dictionnaire
	#storage_data[cle]["Lagged-NW"]=storage_data[cle]['withdrawal']-storage_data[cle]['injection']
	
#print(storage_data)