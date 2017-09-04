import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from astropy.modeling import models

from jfit.models import ImageModel, Sersic2D


def xy_data(nx=5,ny=7):
	# generate data
	y, x = np.mgrid[:nx, :ny]
	return x,y


class TestImageModels(unittest.TestCase):
	def test_plain_evaluate(self):
		"""This will only test that an array of zeros is returned with dimensions
		that match the input arrays"""
		x,y = xy_data()
		m = ImageModel()
		z = m._evaluate(x,y)

		s1 = x.shape
		s2 = z.shape
		self.assertTrue(s1 == s2)

		self.assertFalse(np.any(z))


	def test_oversample_dimensions(self):
		"""This will test that the dimensions returned by the input oversampling routine
		are what we want"""
		nx,ny = 5,7
		factor = 5
		x,y = xy_data(nx,ny)
		m = ImageModel()
		xx,yy = m._oversample_input(x,y,factor)

		sx,sy = xx.shape,yy.shape

		self.assertEqual(sx[0],nx*factor)
		self.assertEqual(sy[0],nx*factor)
		self.assertEqual(sx[1],ny*factor)
		self.assertEqual(sy[1],ny*factor)


	def test_oversample_input_range(self):
		"""Test that the range of the oversampled input arrays are the same as the original
		arrays"""
		nx,ny = 5,7
		factor = 5
		x,y = xy_data(nx,ny)
		m = ImageModel()
		xx,yy = m._oversample_input(x,y,factor)

		# this will both test that the boundary values are the same and that all values are
		# the same of the oversampled input boundary arrays
		self.assertTrue(set(xx[:,0])==set(x[:,0]))
		self.assertTrue(set(xx[:,-1])==set(x[:,-1]))
		self.assertTrue(set(yy[0,:])==set(y[0,:]))
		self.assertTrue(set(yy[-1,:])==set(y[-1,:]))


	def test_oversample_model_dimensions(self):
		"""Test that the rebinned model has the right dimensions"""
		s = (5,7)
		factor = 5
		x,y = xy_data(s[0],s[1])
		m = ImageModel()
		z = m.evaluate(x,y)
		return self.assertTrue(z.shape == s)


	def test_sersic2d_instantiate(self):
		"""Test that the Sersic2D instantiates correctly and that the
		profile end up where we expect"""

		# setup grid
		hpw = 10
		s = (2*hpw+1,4*hpw+1)
		y, x = xy_data(s[0],s[1])

		m = Sersic2D(x_0=hpw,y_0=hpw,n=2,r_eff=0.5*hpw)
		f = m(x,y)

		# brightest pixel should be equal to hpw
		mask = f == f.max()

		return self.assertTrue(x[mask]==hpw and y[mask]==hpw)


	def test_check_bounds(self):
		"""Test the functionality of the check_bounds() method."""
		hpw = 10
		m = Sersic2D(x_0=hpw,y_0=hpw,n=2,r_eff=0.5*hpw,theta=0.5,ellip=0.5)
		m.bounds['x_0'] = (0,2*hpw+1) # parameter svalue is within bounds
		m.bounds['n'] = (2,4)         # parameter value is at lower limit
		failed = m.check_bounds()
		return self.assertTrue(len(failed)==1 and 'n' in failed)



if __name__ == '__main__' :
	unittest.main()
