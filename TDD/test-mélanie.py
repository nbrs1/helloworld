from tp2 import berlin
import unittest


<<<<<<< HEAD:TDD/test.py
ville1=[0,1]
ville2=[0,2]
voiture=berlin(100,5,6,5,1,6)
=======

>>>>>>> 77b3e67e40a22fa35ab169573efbebeaf0834eca:TDD/test-mélanie.py

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
