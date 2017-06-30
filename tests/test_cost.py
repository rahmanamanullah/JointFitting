import unittest
import numpy as np
from iminuit import Minuit
from astropy.modeling import models

#from .context import cost

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import jointfitting.cost as cost


def _linear(x,k,m) :
    """
    Linear hypothesis
    """
    return k*x + m

class TestCost(unittest.TestCase):
	def test_validate_shape_of_identical_arrays1D(self):
		nx = 5
		x = np.random.rand(nx)
		y = np.random.rand(nx)
		return self.assertTrue(cost._validate_shape(x,y))


	def test_validate_shape_of_identical_arrays2D(self):
		nx,ny = 3,3
		x = np.random.rand(nx,ny)
		y = np.random.rand(nx,ny)
		z = None
		w = None
		return self.assertTrue(cost._validate_shape(x,y,z,w))


	def test_validate_shape_of_identical_arrays3D(self):
		nx,ny,nz = 5,5,3
		x = np.random.rand(nx,ny,nz)
		y = np.random.rand(nx,ny,nz)
		z = np.random.rand(nx,ny,nz)
		w = np.random.rand(nx,ny,nz)
		return self.assertTrue(cost._validate_shape(x,y,z,w))


	def test_validate_shape_of_non_identical_arrays_1D(self):
		nx,ny = 3,5
		x = np.random.rand(nx)
		y = np.random.rand(ny)
		return self.assertFalse(cost._validate_shape(x,y))


	def test_not_in_list(self):
		a = ['a','b','c','d','e']
		b = ['a','c','e']
		c2 = ['b','d']
		c1 = cost._not_in_list(a,b)
		return self.assertTrue(c1==c2)


	def test_determine_nsets1D_simple(self):
		a = np.array([[2,3],[4,5]])
		na = cost._determine_nsets(a,None)
		return self.assertEqual(na,2)

	def test_determine_nsets1D_with_extra_dimension1(self):
		a = np.array([[2,3,4,5]])
		na = cost._determine_nsets(a,None)
		return self.assertEqual(na,1)

	def test_determine_nsets1D_with_extra_dimension2(self):
		b = np.array([[2],[3],[4],[5]])
		nb = cost._determine_nsets(b,None)
		return self.assertEqual(nb,1)

	def test_determine_nsets2D_simple1(self):
		b = np.array([[2,3],[4,5]])
		nb = cost._determine_nsets(b,b)
		return self.assertEqual(nb,1)

	def test_determine_nsets2D_simple2(self):
		b = np.array([[[2,3],[4,5]],[[6,7],[8,9]]])	
		nb = cost._determine_nsets(b,b)
		return self.assertEqual(nb,2)

	def test_determine_nsets2D_with_extra_dimension1(self):
		b = np.array([[[2,3],[4,5]]])
		nb = cost._determine_nsets(b,b)
		return self.assertEqual(nb,1)

	def test_determine_nsets2D_with_extra_dimension1(self):
		b = np.array([[[2,3],[4,5]],[[6,7],[8,9]]])	
		nb = cost._determine_nsets(b,b)
		return self.assertEqual(nb,2)


	def test_all_param_names_simple(self):
		nsets = 2
		param_names = ['a','b','c']
		common_names = ['c']
		expected_result = ['a_0','b_0','a_1','b_1','c']
		result = cost._all_param_names(nsets,param_names,common_names)
		return self.assertTrue(result==expected_result)

class TestCostWithFitting(unittest.TestCase):

	def test_linefit_with_iminuit_simple(self):
		# generate some fake data
		m,k = 3.,5.         # intercept and slope
		x = np.array([1,2,3,4,5])
		y = m + k*x

		# setup minuit function
		test = cost.JointCostFunctor(_linear,x,y)
		mm = Minuit(test, print_level=0,
					errordef=1.,
					m=0.0,k=0.0,
					error_m=0.1,error_k=0.1)
		mm.migrad()

		self.assertAlmostEqual(mm.values['m'],m)
		self.assertAlmostEqual(mm.values['k'],k)


	def test_linefit_with_iminuit_astropymodel(self):
		# generate some fake data
		m,k = 3.,5.         # intercept and slope
		x = np.array([1,2,3,4,5])
		y = m + k*x

		# setup minuit function
		l = models.Linear1D()
		test = cost.JointCostFunctor(l.evaluate,x,y)
		mm = Minuit(test, print_level=0,
					errordef=1.,
					intercept=0.0,slope=0.0,
					error_intercept=0.1,error_slope=0.1)
		mm.migrad()

		self.assertAlmostEqual(mm.values['intercept'],m)
		self.assertAlmostEqual(mm.values['slope'],k)



if __name__ == '__main__' :
	unittest.main()

