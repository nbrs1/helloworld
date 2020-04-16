class car
    def __init__(self,type):
        self.marque=marque
        
class berlin
    def __init__(self,maxspeed,places,length,autonomy,automaticité,confort):
        self.maxspeed=maxspeed
        self.places=places
        self.autonomy=autonomy
        self.marque=marque
        self.automaticité=automaticité
    def trajet_possible(self,ville1,ville2):
        if ((ville1[0]-ville2[0])**2+(ville[1]-ville[2])**2)**(1/2)<self.autonomy:
            return true
        else:
            return false
    def dureetrajet(self,v1,v2):
        return ((ville1[0]-ville2[0])**2+(ville[1]-ville[2])**2)**(1/2)/self.maxspeed
        
class quatrequatre
    def __init__(self,places,autonomy,taillepneu,confort,puissance):
        self.places=places
        self.autonomy=autonomy
        self.taillepneu=taillepneu
        self.confort=confort
        self.puissance=puissance
    def sensationenmontagne(self):
        return self.confort*self.puissance
    def trajet_possible(self,ville1,ville2):
        if ((ville1[0]-ville2[0])**2+(ville[1]-ville[2])**2)**(1/2)<self.autonomy:
            return true
        else:
            return false
       