import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import unittest
from astropy.modeling import models, fitting

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
		m.oversample_factor(factor)
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
		m.oversample_factor(factor)
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


class TestKernel(unittest.TestCase):
	def test_kernel_assignment(self):
		"""Test that an assigned kernel is automatically generated and
		has the correct dimensions"""

		factor = 3 # oversampling factor
		fwhmi = 4  # fwhm of image Gaussian
		fwhmk = 3  # fwhm of kernel Gaussian
		khpw = 2*fwhmk

		m = SymmetricGaussian2D(fwhm=fwhmi)
		m.oversample_factor(factor)

		k = SymmetricGaussian2D(fwhm=fwhmk)
		m.set_kernel(k,khpw)

		kx,ky = m._k.shape

		return self.assertTrue(kx == factor*(2*khpw+1) and kx == ky)


	def test_gaussian_convolution_of_gaussian_image(self):
		"""Convolve a Gaussian image with a Gaussian kernel, and measure
		the sigma of the final image.  A Guassian convolved with a Gaussian
		is a Gaussian with a variance that is the sum of the two:

		http://www.tina-vision.net/docs/memos/2003-003.pdf
		"""

		# image
		s = (31,61)
		fwhmi = 7
		factor = 5
		i = SymmetricGaussian2D(x_0=(s[1]-1)/2,y_0=(s[0]-1)/2,fwhm=fwhmi)
		i.oversample_factor(factor)

		# assign Gaussian kernel
		fwhmk = 5
		khpw = 2*fwhmk        # kernel half-width
		k = SymmetricGaussian2D(fwhm=fwhmk)
		i.set_kernel(k,khpw)

		# evaluate model
		x,y = testimage.xy_data(s[0],s[1])
		m = i(x,y)

		# fit Gaussian2D to the convolved model
		g_init = models.Gaussian2D(amplitude=m.max(), 
			x_mean=(s[1]-1)/2, y_mean=(s[0]-1)/2,
			x_stddev=fwhmi, y_stddev=fwhmi)
		fit_g = fitting.LevMarLSQFitter()
		g = fit_g(g_init, x, y, m)

		# calculate the the theoretical standard deviation for convolution 
		# of two Gaussians in 1D.
		stddevi,stddevk = fwhmi/2.35,fwhmk/2.35
		stddevt = np.sqrt(stddevi**2 + stddevk**2)

		# make sure that it is within 10%
		rel_dev = (g.x_stddev.value - stddevt)/stddevt

		return self.assertTrue(np.abs(rel_dev) < 0.1)


if __name__ == '__main__' :
	unittest.main()
