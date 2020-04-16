import td2
import unittest



ville=[[10,1],[0,5],[-5,20],[2,25]]

voiture=td2.berlin(100,5,6,5,1,6)



class Test(unittest.TestCase):
	def test_trajet_possible(self):
		for i in range(2):
			result=voiture.trajet_possible(self.ville[i],self.ville[i+1])
			rtest=False
			if ((self.ville[i][0]-self.ville[i+1][0])**2+(self.ville[i][1]-self.ville[i+1][1])**2)**(1/2)<self.autonomy:
				rtest=True
			self.assertEqual(result,rtest)
		
		
	def test_autonomy_sign(self):
		assert(voiture.autonomy>0)
	def test_confort_minimal(self):
		assert(voiture.confort>3)
	def test_reguled_lenght(self):
		assert(2<voiture.length<5)

	




if __name__ == '__main__':
 	unittest.main()
