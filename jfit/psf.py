import numpy as np
from astropy.modeling import Parameter
from models import ImageModel

__all__ = ['SymmetricGaussian2D','SymmetricMoffat2D']

class PSF(ImageModel):
    """
    Abstract PSF class.
    
    http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/PSFsAndSampling.pdf
    """

    def kernel(self,khpw):
        """
        Create a kernel of size (2*khpw+1,2*khpw+1) with the *shape* parameters 
        of the PSF. It is implicitly assumed that the evaluate method of the
        deriving classes are defined as

            evaluate(x,y,amplitude,x_0,y_0,*args)

        The kernel will be normalized and the center of the PSF profile will 
        land in the center of the central pixel.
        """
        nelem = 2*khpw+1
        y,x = np.mgrid[:nelem, :nelem]
        d = self.evaluate(x,y,1.,khpw,khpw,*self.parameters[3:])

        return d/d.sum()


    def save_kernel(self,khpw,kfname):
        """
        Save PSF as a normalized kernel of size (2*khpw+1,2*khpw+1) to 
        the FITS file 'kfname'.
        """
        d = self.kernel(khpw)
        hdu = fits.PrimaryHDU(d)
        hdu.writeto(kfname,clobber=True)
        hdu.close()
        
        return True


class SymmetricGaussian2D(PSF):
    """Gausian 2D profile (normalized).  The shape is parameterized in terms
    of FWHM rather than sigma for historical reasons."""
    amplitude = Parameter(name='amplitude',default=1.)
    x_0       = Parameter(name='x_0',default=0.)
    y_0       = Parameter(name='y_0',default=0.)
    fwhm      = Parameter(name='fhwm',default=1.)
    
    @staticmethod
    def _evaluate(x, y, amplitude, x_0, y_0, fwhm):
        """The true 'evaluate' method without oversampling"""
        sigma = 0.5*fwhm / np.sqrt(2.*np.log(2))
        rr_gg = ((x - x_0)**2 + (y - y_0)**2) / sigma ** 2
        return amplitude / (2*np.pi*sigma**2) * np.exp(-0.5 * rr_gg )
    
    def evaluate(self, x, y, amplitude, x_0, y_0, fwhm):
        return self._oversample_model(x,y,None,amplitude,x_0,y_0,fwhm)
    
    
class SymmetricMoffat2D(PSF):
    """Moffat 2D profile (normalized).  The shape is parametrized in terms
    of FWHM and alpha rather than alpha and gamma for historical reasons."""
    amplitude = Parameter(name='amplitude',default=1.)
    x_0       = Parameter(name='x_0',default=0.)
    y_0       = Parameter(name='y_0',default=0.)
    fwhm      = Parameter(name='fhwm',default=1.)
    alpha     = Parameter(name='alpha',default=3.)
    
    @fwhm.validator
    def fwhm(self, value):
        # Remember, the value can be an array
        if np.any(value <= 0.):
            raise InputParameterError(
                    "parameter 'fwhm' must be greater than zero ")

    @alpha.validator
    def alpha(self, value):
        # Remember, the value can be an array
        if np.any(value <= 1.):
            raise InputParameterError(
                    "parameter 'alpha' must be greater than 1")            
            
    @staticmethod
    def _evaluate(x,y,amplitude,x_0,y_0,fwhm,alpha):
        """The true 'evaluate' method without oversampling"""
        gamma = 0.5 * fwhm/np.sqrt(2**(1./alpha) - 1.)
        rr_gg = ((x - x_0) ** 2 + (y - y_0) ** 2) / gamma ** 2    
        return amplitude * (alpha - 1.)/(np.pi*gamma*gamma) * (1 + rr_gg)**(-alpha)

    def evaluate(self,x,y,amplitude,x_0,y_0,fwhm,alpha):
        return self._oversample_model(x,y,None,amplitude,x_0,y_0,fwhm,alpha)
