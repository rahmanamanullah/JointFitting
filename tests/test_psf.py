import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from astropy.modeling import models

from jfit.psf import SymmetricGaussian2D, SymmetricMoffat2D
import test_imagemodels as testimage

class TestPSF(unittest.TestCase):
	def test_oversampling_gauss_flux_conservation(self):
		"""Make sure that the flux is conserved after the oversampled model has been 
		rebinned to the original size"""
		s = (5,7)
		x0,y0 = 3,4
		amplitude = 100.
		fwhm = 1
		factor = 5
		x,y = testimage.xy_data(s[0],s[1])
		m = SymmetricGaussian2D()
		m.sample_factor = factor
		z1 = m._oversampled_model(x,y,factor,amplitude,x0,y0,fwhm)
		z2 = m.evaluate(x,y,amplitude,x0,y0,fwhm)

		return self.assertAlmostEqual(np.sum(z1),np.sum(z2))


	def test_oversampling_moffat_flux_conservation(self):
		"""Make sure that the flux is conserved after the oversampled model has been 
		rebinned to the original size"""
		s = (5,7)
		x0,y0 = 3,4
		amplitude = 100.
		fwhm = 1
		alpha = 2.
		factor = 5
		x,y = testimage.xy_data(s[0],s[1])
		m = SymmetricMoffat2D()
		m.sample_factor = factor
		z1 = m._oversampled_model(x,y,factor,amplitude,x0,y0,fwhm,alpha)
		z2 = m.evaluate(x,y,amplitude,x0,y0,fwhm,alpha)

		return self.assertAlmostEqual(np.sum(z1),np.sum(z2))


	def test_oversampled_gauss_kernel_dimensions(self):
		"""Make sure that the oversampled kernel has the correct 
		dimensions for the given oversampling factor."""

		sample_factor = 5
		fwhm = 2
		khpw = 2*fwhm
		m = SymmetricGaussian2D(fwhm=fwhm)
		d = m.kernel(khpw,sample_factor)

		kx,ky = d.shape

		return self.assertTrue(kx == sample_factor*(2*khpw+1) and ky == kx)


	def test_oversampled_gauss_kernel_position(self):
		"""Make sure that the center of the Gaussian is in the correct
		position for the given oversampling factor."""

		sample_factor = 5
		fwhm = 2
		khpw = 2*fwhm
		m = SymmetricGaussian2D(fwhm=fwhm)
		d = m.kernel(khpw,sample_factor)

		# setup x and y grids
		kx,ky = d.shape
		y,x = np.mgrid[:kx, :ky]

		# the center position is at the brightest pixel
		mask = d == d.max()
		x1,y1 = x[mask],y[mask]

		return self.assertTrue(x1==y1 and x1==(khpw+0.5)*sample_factor - 0.5)


if __name__ == '__main__' :
	unittest.main()
