import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from astropy.modeling import models

from jfit.models import ImageModel


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


if __name__ == '__main__' :
	unittest.main()
