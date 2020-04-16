from tp2 import berlin
import unittest


ville1=[0,1]
ville2=[0,2]


class Test(unittest.TestCase):
	def test_trajet_possible(self):
		ville1=[0,1]
		ville2=[0,2]
		voiture=berlin(100,5,6,5,1,2)
		
		assert (trajet_possible(voiture,ville1,ville2))

		


if __name__ == '__main__':
 	unittest.main()