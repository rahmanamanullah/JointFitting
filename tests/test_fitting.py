import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from astropy.modeling import models

import jfit.fitting as fitting
import test_imagemodels as testimage

class TestFitting(unittest.TestCase):

	def test_linefit_to_one_dataset(self):
		# some fake data
		m,k = 3.,5.
		x1 = np.arange(10)
		y1 = k*x1 + m

		# linear model from astropy.modeling
		l1 = models.Linear1D()

		# fitter
		jf = fitting.JointMinuitFitter(verbose=False)
		fitl = jf(l1,x1,y1)

		self.assertAlmostEqual(fitl.slope.value,k)
		self.assertAlmostEqual(fitl.intercept.value,m)


	def test_linefit_to_two_datasets(self):
		# some fake data
		m1,k1 = 3.,5.
		x1 = np.arange(10)
		y1 = k1*x1 + m1

		m2,k2 = m1*m1,1.1*k1
		x2 = np.arange(10)
		y2 = k2*x2 + m2

		xx = np.vstack((x1,x1))
		yy = np.vstack((y1,y2))

		# linear model from astropy.modeling
		l1 = models.Linear1D()
		l2 = models.Linear1D()

		# fitter
		jf = fitting.JointMinuitFitter(verbose=False)
		fitl = jf([l1,l2],xx,yy)

		fl1,fl2 = fitl

		self.assertAlmostEqual(fl1.slope.value,k1)
		self.assertAlmostEqual(fl1.intercept.value,m1)
		self.assertAlmostEqual(fl2.slope.value,k2)
		self.assertAlmostEqual(fl2.intercept.value,m2)

	def test_linefit_to_two_datasets_with_common_slope(self):
		# some fake data
		m1,k1 = 3.,5.
		x1 = np.arange(-10,11)  # important in order for the intercept to be the same
		y1 = k1*x1 + m1

		m2,k2 = m1*m1,1.1*k1
		x2 = np.arange(-10,11)
		y2 = k2*x2 + m2

		xx = np.vstack((x1,x1))
		yy = np.vstack((y1,y2))

		# linear model from astropy.modeling
		l1 = models.Linear1D()
		l2 = models.Linear1D()

		# fitter
		jf = fitting.JointMinuitFitter(verbose=False)
		fitl = jf([l1,l2],xx,yy,common_names=['slope'])

		fl1,fl2 = fitl

		self.assertAlmostEqual(fl1.intercept.value,m1)
		self.assertAlmostEqual(fl2.intercept.value,m2)
		self.assertEqual(fl1.slope.value,fl2.slope.value)

	def test_gaussian2Dfit_to_two_datasets_with_common_sigma(self):
		"""Joint 2D fitting with common and individual image dependent
		parameters"""
		nx,ny = 128,128
		common_names = ['x_stddev','y_stddev','theta']		

		# models (sigma are offset with 10% in x-direction)
		x0,y0 = 40,40
		x1,y1 = 100,100
		g1 = models.Gaussian2D(amplitude=5.,x_mean=x0,y_mean=y0,x_stddev=5,y_stddev=5)
		g2 = models.Gaussian2D(amplitude=3.,x_mean=x1,y_mean=y1,x_stddev=1.1*5,y_stddev=5)

		# generate data
		y, x = np.mgrid[:nx, :ny]
		z1,z2 = x*0.0,x*0.0
		z1 += g1(x,y)
		z2 += g2(x,y)

		xx = np.stack((x,x),axis=0)
		yy = np.stack((y,y),axis=0)
		zz = np.stack((z1,z2),axis=0)
	
		# fitter
		jf = fitting.JointMinuitFitter(verbose=False)
		fitl = jf([g1,g2],xx,yy,zz,maxiter=50000,common_names=common_names)

		fl0,fl1 = fitl

		self.assertAlmostEqual(fl0.x_mean.value,x0)
		self.assertAlmostEqual(fl0.y_mean.value,y0)
		self.assertAlmostEqual(fl1.x_mean.value,x1)
		self.assertAlmostEqual(fl1.y_mean.value,y1)
		self.assertEqual(fl0.x_stddev.value,fl1.x_stddev.value)
		self.assertEqual(fl0.y_stddev.value,fl1.y_stddev.value)


	
if __name__ == '__main__' :
	unittest.main()
