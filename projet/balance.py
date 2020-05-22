import pandas as pd
import supply as s
import demand as d
conso = d.import_csv("DE.csv",";",True)      
sig = d.optimize_sigmoid(conso)
coef = sig.optimize()
c = sig.create_consumption()
Demand=[]
temp=conso["actual"].values
for i in range(len(conso["Date"].values)):
	Demand.append(c.get_consumption(temp[i]))

consumption = pd.DataFrame({"Date": conso["Date"].values, "Demand": Demand})

storage_data=s.import_storage_data()
price_data=s.import_price()
reg=s.regression(storage_data,price_data)
classif=s.classification(storage_data,price_data)
classif.innerjoin()

def coef(storage):		
		return reg.droitepredite(storage)
print(coef("SF - UGS Peckensen"))

def classification_choisie(storage):		
		return classif.compare_lr_rf_OnOneStorage(storage)

def forecast_NW(storage, classification):
	if classification==1:#lr
		pred=classif.predictionlogreg(storage)
	else:#rf
		pred=classif.predictionrandforest(storage)
	for i in range(len(pred)):
		if pred[i]==1:
			c=coef(storage)
			nw=c[0][0]*classif["lagged_NW"]+c[1]
			pred[i]=nw
	return pred

def sum_nw():
	l=[]
	for storage in storage_data:
		classification=classification_choisie(storage)
		pred=forecast_NW(storage,classification)
		l.append(pred)
	sum=[]
	for i in range(len(l[0])):
		s=0
		for j in range(len(l)):
			s+=l[j][i]
		sum.append(s)
	return sum
som=sum_nw()
supply=pd.DataFrame({"Date": classif["gasDayStartedOn"].values, "supply":som })
		
jointure=merge(supply,consumption,left_on="Date", right_on="Date")
print(jointure)






