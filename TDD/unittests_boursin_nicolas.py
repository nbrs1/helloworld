import random
import unittest
import td2real
class distancesVilleConnue(unittest.TestCase):
	villesConnues = [[-150,5],[900,-8],[14,14],[-78,98]]
	def test_trajetPossible(self):
		autonomy=random.randint(0,500)
		voiture=td2real.quatrequatre(0,0,0,0,0)
		"""ces valeurs ne sont pas importantes dans ce test\
		 a part autonomy que l on modifie apres"""
		voiture.autonomy=autonomy
		i=0
		while i<(len(self.villesConnues)-1):
			test=voiture.trajet_possible(self.villesConnues[i],self.villesConnues[i+1])
			ans=False
			if ((self.villesConnues[i][0]-self.villesConnues[i+1][0])**2+(self.villesConnues[i][1]-self.villesConnues[i+1][1])**2)**(1/2)<voiture.autonomy:
				ans=True
			self.assertEqual(test,ans)
			i+=1

class valeursPossible(unittest.TestCase):
	def test_puissanceneg(self):
		voiture=td2real.quatrequatre(0,0,0,0,-1)
		self.assertRaises(td2real.pbsigne,voiture.sensationenmontagne)
	def test_puissancenonnulle(self):
		voiture=td2real.quatrequatre(0,0,0,0,0)
		self.assertRaises(td2real.puissancenulle,voiture.sensationenmontagne)
	def test_puissancepastropgrande(self):
		voiture=td2real.quatrequatre(0,0,0,0,101)
		self.assertRaises(td2real.puissancetropgrande,voiture.sensationenmontagne)
		
if __name__ == '__main__':
    unittest.main()


