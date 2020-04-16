class car:
    def __init__(self,type):
        self.marque=marque
class pbsigne(ValueError):
	pass
class puissancenulle(ValueError):
	pass
class puissancetropgrande(ValueError):
	pass
  #classe créée par nicolas      
class berlin:
    def __init__(self,maxspeed,places,length,autonomy,automaticité,confort):
        self.maxspeed=maxspeed
        self.places=places
        self.autonomy=autonomy
        self.marque=marque   
        self.automaticité=automaticité
    def trajet_possible(self,ville1,ville2):  #première méthode
        if ((ville1[0]-ville2[0])**2+(ville1[1]-ville2[1])**2)**(1/2)<self.autonomy:
            return True
        else:
            return False
    def dureetrajet(self,v1,v2):    #deuxième méthode
        return ((ville1[0]-ville2[0])**2+(ville[1]-ville[2])**2)**(1/2)/self.maxspeed
 
 
 #classe créée par mélanie       
class quatrequatre:
    def __init__(self,places,autonomy,taillepneu,confort,puissance):
        self.places=places
        self.autonomy=autonomy
        self.taillepneu=taillepneu
        self.confort=confort
        self.puissance=puissance

    def sensationenmontagne(self):
    	if self.puissance<0:
    		raise pbsigne('dure')
    	if self.puissance==0:
    		raise puissancenulle('dure')
    	if self.puissance>100:
    		raise puissancetropgrande('dure')
    	return self.confort*self.puissance
    	
        
    def trajet_possible(self,ville1,ville2):
        if ((ville1[0]-ville2[0])**2+(ville1[1]-ville2[1])**2)**(1/2)<self.autonomy:
            return True
        else:
            return False
       
