import pandas as pd
import os
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random as rd 
from math import sqrt

DE = pd.read_csv("DE.csv", sep=";")
DE.columns=['Date','LDZ','actual','normal']
#fig=plt.figure()
y1=DE['LDZ']
y2=DE['actual']
y3=DE['normal']
x=DE['Date']
#This function sets the working directory
def set_wd(wd):
    os.chdir(wd)

#This function imports a csv file and has the option to plot its value columns as a function of the first column
def import_csv(f_name = "DE.csv", delimeter = ";", plot = True):

   # return f >> mutate(Date = pd.to_datetime(conso['Date']))
    DE = pd.read_csv(f_name, delimeter)
    DE.columns=['Date','LDZ','actual','normal']
    if plot:
        fig=plt.figure()
        y1=DE['LDZ']
        y2=DE['actual']
        x=DE['Date']
# Plot using Matplotlib all three series on 3 sub plots to see them varying together
# Chanegr la taille de police par défaut
        plt.rcParams.update({'font.size': 15})

        
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
    return DE
#This function creates a scatter plot given a DataFrame and an x and y column
def scatter_plot(dataframe = "conso", col = "red"):
    x='LDZ'
    y='actual'
    y1=conso[x]
    y2=conso[y]
    fig2=plt.figure()
    ax=plt.axes()
    plt.plot(y2,y1,col,linestyle='dashed',label='reality')
#This function is the sigmoid function for gas consumption as a function of temperature
def h(t, a, b, c, d):
    return(d+a/(1+(b/(t-40))**c))

#The following function takes sigmoid parameters values and a temperature vector as input and plots the sigmoid, can compare it with the actual values
def consumption_sigmoid(t, real_conso, a = 900, b = -35, c = 6, d = 300, plot = True):
    h_hat = np.empty(len(t))
    for i in range(len(t)):
       h_hat[i] = h(t[i], a, b, c, d)

    if plot:
        plt.plot(t,h_hat)
        #if real_conso is not None you plot it as well
        if not isinstance(real_conso, type(None)):
            plt.plot(t,real_conso)
            if(len(t) != len(real_conso)):
                print("Difference in length between Temperature and Real Consumption vectors")
            # add title and legend and show plot
    return h_hat

#The following function gets the fit metrics list between 2 sigmoids
def get_fit_metrics(h_hat, real_conso):
    if(len(h_hat) != len(real_conso)):
        print("Difference in length between Fit and Real Consumption vectors")
    else:
       #somme des carrés des écarts sur longueur de la liste
        s=0
        for i in range(len(h_hat)):
            s+=((h_hat[i]-real_conso[i])**2)/len(h_hat)
        #moyenne exp
        avr=0
        for i in range(len(h_hat)):
            avr+=real_conso[i]/len(h_hat)
        #moyenne sigmoid
        avh=0
        for i in range(len(h_hat)):
            avh+=h_hat[i]/len(h_hat)
        shr=0
        for i in range(len(h_hat)):
            shr+=(h_hat[i]-avh)*(real_conso[i]-avr)
        
        s2=0
        for i in range(len(h_hat)):
            s2+=(h_hat[i]-avh)**2
        sh=sqrt(s2)
        s3=0
        for i in range(len(h_hat)):
            s3+=(real_conso[i]-avr)**2
        sr=sqrt(s3)

         #self.__corr, self.__rmse, self.__nrmse, self.__anrmse 
        corr=shr/(sh*sr)
        rmse=sqrt(s)
        nrmse=rmse/avr
        anrmse=abs(nrmse)
        return [corr,rmse,nrmse,anrmse]

#The following class is the cosumption class it takes sigmoid parameters as well as a temperature as input
class consumption:
    #Initialize class
    def __init__(self, a, b, c, d):
        self.a=a
        self.b=b
        self.c=c
        self.d=d


    #calculate the consumption given a temperature
    def get_consumption(self, temperature):
        return h(temperature,self.a, self.b, self.c, self.d)

        

    #get the sigmoid considering a temperature between -40 and 39, use the function consumption_sigmoid above
    def sigmoid(self, p):
        t=np.linspace(-40,39,1000)
        h_hat = np.empty(len(t))
        for i in range(len(t)):
            h_hat[i] = self.get_consumption(t[i])
        if p:
            plt.plot(t,h_hat)
        return h_hat
        
    #This is what the class print if you use the print function on it
    def __str__(self):
        
        return "sigmoid de parametre {} {} {} {}".format(self.a,self.b,self.c,self.d)

#The following class optimizes the parameters of the sigmoid and returns an object of class consumption
class optimize_sigmoid:
    #Initialize guess values that are common to all instances of the clasee
    g=[500,-25,2,100]

    def __init__(self, f):
        if isinstance(f, pd.DataFrame):
            if 'Actual' and 'LDZ' in f.columns:
                self.f=f
            else:
                print("Class not initialized since f does not contain Actual and LDZ column names")
        else:
            print("Class not initialized since f is not a DataFrame")

    #optimize and return metrics use functions h, consumption_sigmoid defined above as well as get_fit_metrics
    def optimize(self):
        if self.f is not None:
            y1=self.f['LDZ']
            y2=self.f['actual']
            t=y2.values.tolist()
            cons=y1.values.tolist()
            j=0
            while j<len(t):
                if np.isfinite(t[j])==False or np.isfinite(cons[j])==False:
                    del t[j]
                    del cons[j]
                    j-=1
                j+=1
            self.coef, self.cov = curve_fit(h,t,cons,self.g)           
            s = consumption_sigmoid(t,cons,self.coef[0],self.coef[1],self.coef[2],self.coef[3],True)

            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, cons)
        else:
            print("Class not initialized")
        return self.coef

    #this function returns the fit metrics calculated above
    def fit_metrics(self):
        if self.f is not None:
            y1=self.f['LDZ']
            y2=self.f['actual']
            t=y2.values.tolist()
            cons=y1.values.tolist()
            j=0
            while j<len(t):
                if np.isfinite(t[j])==False or np.isfinite(cons[j])==False:
                    del t[j]
                    del cons[j]
                    j-=1
                j+=1
            self.coef, self.cov = curve_fit(h,t,cons,self.g)           
            s = consumption_sigmoid(t,cons,self.coef[0],self.coef[1],self.coef[2],self.coef[3],True)

            self.__corr, self.__rmse, self.__nrmse, self.__anrmse = get_fit_metrics(s, cons)
        else:
            print("optimize method is not yet run")
        return self.__corr, self.__rmse, self.__nrmse, self.__anrmse 
    #This function creates the class consumption
    def create_consumption(self):
        if self.f is not None:
            coef=self.optimize()
            ex=consumption(coef[0],coef[1],coef[2],coef[3])
            return ex
        else:
            print("optimize method is not yet run")

    #This is what the class print if you use the print function on it
    def __str__(self):
        if self.f is not None:
            coef=self.optimize()
            return "les parametres optimaux sont {}{}{}{}".format(coef[0],coef[1],coef[2],coef[3])
        else:
            t = "optimize method is not yet run"
        return t

#If you have filled correctly the following code will run without an issue        
if __name__ == '__main__':

    #set working directory
   # set_wd(Documents\in104\td3)

    #1) import consumption data and plot it
    conso = import_csv("DE.csv",";",True)

    #2) work on consumption data (non-linear regression)
    #2)1. Plot consumption as a function of temperature    
   
    
    scatter_plot(conso,'red')
     #2)2. optimize the parameters
    sig = optimize_sigmoid(conso)
    print(sig)
    coef = sig.optimize()
    c = sig.create_consumption()
    print(sig)


    #2)3. check the new fit

    # These are the 3 ways to access a protected attribute, it works the same for a protected method
    # An attribute/method is protected when it starts with 2 underscores "__"
    # Protection is good to not falsy create change
    
    print(
            [
            sig.__dict__['_optimize_sigmoid__corr'],
            sig.__dict__['_optimize_sigmoid__rmse'],
            sig.__dict__['_optimize_sigmoid__nrmse'],
            sig.__dict__['_optimize_sigmoid__anrmse']
            ]
        )

    print(
            [
            sig._optimize_sigmoid__corr,
            sig._optimize_sigmoid__rmse,
            sig._optimize_sigmoid__nrmse,
            sig._optimize_sigmoid__anrmse
            ]
        )

    print(
            [
            getattr(sig, "_optimize_sigmoid__corr"),
            getattr(sig, "_optimize_sigmoid__rmse"),
            getattr(sig, "_optimize_sigmoid__nrmse"),
            getattr(sig, "_optimize_sigmoid__anrmse")
            ]
        )
    
  
    print(sig.fit_metrics())
    print(c)
    c.sigmoid(True)
    plt.show()
    print(c)
    
    #3) If time allows do TSA on actual temperature
    #3)1. Check trend (and Remove it)
    #3)2. Check Seasonality (Normal Temperature)
    #3)3. Model stochastic part that is left with ARIMA
    #3)4. Use this to forecast consumption over N days

