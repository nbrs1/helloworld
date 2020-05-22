import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

#set working directory

#1) import consumption data from DE.csv into a pandas DataFrame and rename Date (CET) column to Date
    # The LDZ represents gas consumption in GWh, Actual is the Actual temperature and Normal is the normal temperature
    #This function imports a csv file and has the option to plot its value columns as a function of the first column
DE = pd.read_csv("DE.csv", sep=";")
DE.columns=['Date','LDZ','actual','normal']
#fig=plt.figure()
y1=DE['LDZ']
y2=DE['actual']
y3=DE['normal']
x=DE['Date']
# Plot using Matplotlib all three series on 3 sub plots to see them varying together
# Chanegr la taille de police par défaut
plt.rcParams.update({'font.size': 15})

fig1 = plt.figure()
ax = plt.axes()
# Couleur spécifiée par son nom, ligne solide
plt.plot(x, y1, color='blue', linestyle='solid', label='LDZ')
# Nom court pour la couleur, ligne avec des traits
plt.plot(x, y2, color='g', linestyle='dashed', label='actual')
# Valeur de gris entre 0 et 1, des traits et des points
plt.plot(x, y3, color='0.75', linestyle='dashdot', label='normal')

# Les labels
plt.title("DATA sous forme de graphe")

# La légende est générée à partir de l'argument label de la fonctio
# plot. L'argument loc spécifie le placement de la légende
plt.legend(loc='lower left');

# Titres des axes
ax = ax.set(xlabel='Date')
#plt.show(fig1)
    


    # Do not forget to add a legend and a title to the plot


    
    # Comment on their variation and their relationships
"""on remarque que la courbes de consommation augmente quand la température baisse et vis versa. C'est le résultat attendu."""

    
    # use dfply to transform Date column to DateTime type

    

#2) work on consumption data (non-linear regression)
#2)1. Plot with a scatter plot the consumption as a function of temperature
fig2=plt.figure()
ax=plt.axes()
plt.plot(y2,y1,c='blue',linestyle='dashed',label='reality')

#2)2. define the consumption function (I give it to you since it is hard to know it without experience)

#This function takes temperature and 4 other curve shape parameters and returns a value of consumption
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#These are random initial values of a, b, c, d
guess_values= [500, -25, 2, 100]



#2)3. Fill out this h_hat array with values from the function h

# You will take the 'Actual' column from the DE.csv file as being the input temperature so its length should be the number of rows in the DataFrame imported
h_hat = np.empty(len(y2))


for i in range(len(y2)):
    h_hat[i]=h(y2[i],guess_values[0],guess_values[1],guess_values[2],guess_values[3])

plt.plot(y2,h_hat,c='red',linestyle='solid',label='theory')   
#plt.show(fig2) 
t=y2.values.tolist()
cons=y1.values.tolist()

j=0
while j<len(t):
    if np.isfinite(t[j])==False or np.isfinite(cons[j])==False:
        del t[j]
        del cons[j]
        j-=1
    j+=1

c,cov=curve_fit(h,t,cons,guess_values)
"""On test désormais l'affichage avec les valeurs optimales trouvées"""
fig3=plt.figure()
ax=plt.axes()
plt.plot(y2,y1,c='blue',linestyle='dashed',label='reality')

#2)2. define the consumption function (I give it to you since it is hard to know it without experience)



#These are random initial values of a, b, c, d




#2)3. Fill out this h_hat array with values from the function h

# You will take the 'Actual' column from the DE.csv file as being the input temperature so its length should be the number of rows in the DataFrame imported
h_hat = np.empty(len(y2))


for i in range(len(y2)):
    h_hat[i]=h(y2[i],c[0],c[1],c[2],c[3])

plt.plot(y2,h_hat,c='red',label='theory')   
plt.show() 
    # For each value of temmperature of this column you will calculate the consumption using the h function above
    # Use the array guess_values for the curve parameters a, b, c, d that is to say a = guess_values[0], b = guess_values[1], c = guess_values[2], d = guess_values[3]

    # Plot on a graph the real consumption (LDZ column) as a function of Actual temperature use blue dots
    # On this same graph add the h_hat values as a function of Actual temperature use a red line for this
    # Do not forget to add a legend and a title to the plot
    # Play around with the parameters in guess_values until you feel like your curve is more or less correct


#2)4. optimize the parameters

    # Your goal right now is to find the optimal values of a, b, c, d using SciPy
    # Inspire yourselves from the following video
    # https://www.youtube.com/watch?v=4vryPwLtjIY

#2)5. check the new fit

#Repeat what we did in 2)3. but with the new optimized coefficients a, b, c, d


#calculate goodness of fit parameters: correlation, root mean square error (RMSE), Average normalised RMSE, normalized RMSE
#averaged normalized RMSE is RMSE/(average value of real consumption)
#normalized RMSE is RMSE/(max value of real consumption - min value of real consumption)
#Any other metric we could use ?"""