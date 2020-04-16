from tp2 import berlin
import unittest


ville1=[0,1]
ville2=[0,2]
voiture=berlin(100,5,6,5,1,6)

class Test(unittest.TestCase):
	def test_trajet_possible(self):
		
		assert(trajet_possible(voiture,ville1,ville2))
	def test_autonomy_sign(self):
		assert(voiture.autonomy>0)
	def test_confort_minimal(self):
		assert(voiture.confort>3)
	def test_reguled_lenght(self):
		assert(2<voiture.length<5)

	




if __name__ == '__main__':
 	unittest.main()