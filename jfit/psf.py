import numpy as np
from astropy.modeling import Parameter
from models import ImageModel
from scipy.interpolate import interp1d

__all__ = ['SymmetricGaussian2D','SymmetricMoffat2D']

class PSF(ImageModel):
    """
    Abstract PSF class.
    
    http://web.ipac.caltech.edu/staff/fmasci/home/astro_refs/PSFsAndSampling.pdf
    """
    def ee(self,radius,sample_factor=10):
        """
        Method for calculate the encircled energy (between 0 and 1) for the radius 1.
        """
        # if there is an analytic solution this should be overridden in the classes
        # deriving from PSF()
        rr = np.asarray(radius).flatten()

        # extend radius array with +/- 0.5 pixel
        rr = np.sort( np.append(rr,[rr.min()-0.5,rr.max()+0.5]) )

        # setup pixel coordinate grid
        hpw = int(np.ceil(rr.max())) + 1
        nelem = 2*hpw+1
        y,x = np.mgrid[:nelem, :nelem]

        # over sampled pixel coordinates
        xs,ys = self._oversample_input(x,y,sample_factor)
        rs = np.sqrt((xs-hpw)**2 + (ys-hpw)**2)

        # evaluate and normalize the PSF
        p = self._evaluate(xs,ys,1.,hpw,hpw,*self.parameters[3:])
        dx,dy = xs[0,1] - xs[0,0],ys[1,0] - ys[0,0]        
        p *= dx*dy

        # calculate encircled energy for all given radii
        e = []
        for r in rr:
            m = rs <= r
            e.append(p[m].sum())

        # setup interpolation object
        ee = interp1d(rr,np.array(e),kind='linear')

        return ee(np.asarray(radius))



    def kernel(self,khpw,sample_factor=1):
        """
        Create a kernel of size (2*khpw+1,2*khpw+1) with the *shape* parameters 
        of the PSF. It is implicitly assumed that the evaluate method of the
        deriving classes are defined as

            evaluate(x,y,amplitude,x_0,y_0,*args)

        The kernel will be normalized and the center of the PSF profile will 
        land in the center of the central pixel.

        If a sample_factor > 1 is specified an over sampled PSF will be 
        returned with the given factor and the center of the PSF will be
        located in 

           ((khpw+0.5)*sample_factor - 0.5,(khpw+0.5)*sample_factor - 0.5)

        and it will have dimensions of

           (sample_factor*(2*khpw+1),sample_factor*(2*khpw+1))

        """
        nelem = 2*khpw+1
        y,x = np.mgrid[:nelem, :nelem]

        if sample_factor == 1:
            d = self.evaluate(x,y,1.,khpw,khpw,*self.parameters[3:])
        elif sample_factor > 1:
            d  = self._oversampled_model(
                x,y,sample_factor,1.,khpw,khpw,*self.parameters[3:])
        else:
            raise ValueError("sample_factor < 1")

        return d/d.sum()


    def save_kernel(self,khpw,kfname,sample_factor=1):
        """
        Save PSF as a normalized kernel of size (2*khpw+1,2*khpw+1) to 
        the FITS file 'kfname'.
        """
        d = self.kernel(khpw,sample_factor)
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
