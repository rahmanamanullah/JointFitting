import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from astropy.modeling import Fittable2DModel

class ImageModel(Fittable2DModel):
    """
    Abstract class for 2D shape models, from simple point-spread-functions
    (PSF) to more complex models.
    
    The class is deriving from the astropy.modeling class Fittable2DModel 
    which facilitates using the astropy.modeling framework for fitting the 
    model.  See the astropy.modeling documentation for details.
    
    The class supports over sampling when calculating the model, which is
    very useful for images with undersampled PSF.
    
    TODO: implement the possibility of instantiating the model with a PSF
          that will be used to convolve the (oversampled) model.
    """

    def __init__(self, **kwargs):
        # the default over sampling factor
        self.sample_factor = 1        
        super(SkyModel, self).__init__(**kwargs)    

        
    def _oversample_factor(self):
        """Can be overwridden to calculate the oversampling factor based
        on shape parameters"""
        return self.sample_factor

    
    def _oversample_input(self,x,y,sample_factor):
        """Over sample the input coordinates"""
        
        xs,ys = x,y
        
        # determine the dimensions, sometimes we might want to calculate
        # e.g. PSF profiles, and will then input 1D arrays for x and y.
        if x.ndim == 1 :
            onedim = True
        elif x.ndim == 2 :
            nx,ny = x.shape
            if nx == 1 or ny == 1 :
                onedim = True
            else :
                onedim = False
        else :
            raise InputParameterError("Input dimensions of 'x' and 'y' must be <= 2")

        # interpolate the coordinate arrays for an array that is sample_factor times
        # longer over each dimension.
        if onedim:
            i = np.arange(len(x))
            xf = interp1d(i,x,kind='linear')
            yf = interp1d(i,y,kind='linear')
            
            ii = np.linspace(0,i.max(),sample_factor*len(x))
            
            xs,ys = xf(ii),yf(ii)
        else :
            # i,j are integer index arrays that have the same
            # lenth as the array dimensions of x and y
            ix,iy = x.shape
            i,j = np.arange(ix),np.arange(iy)
            fx = RectBivariateSpline(i,j,x,kx=1,ky=1)
            fy = RectBivariateSpline(i,j,y,kx=1,ky=1)

            # calculate the x and y arrays for index arrays that
            # span the same range (0,ix-1) and (0,iy-1) but with
            # higher sampling.
            ii = np.linspace(0.,ix-1.,sample_factor*ix)
            jj = np.linspace(0.,iy-1.,sample_factor*iy)

            xs,ys = fx(ii,jj),fy(ii,jj)

        return xs,ys
    
    def _oversample_model(self,x,y,sample_factor,*args):
        """Calculate the model after first oversampling it and then
        rebinning it back to the original resolution."""

        if sample_factor is None :
            sample_factor = self._oversample_factor()
        
        # over sample the input vectors
        if sample_factor > 1:
            xs,ys = self._oversample_input(x,y,sample_factor)
        else :
            xs,ys = x,y
            
        a = self._evaluate(xs,ys,*args)
        ms = a
                
        # rebin model back to the original resolution
        shape = x.shape
        if sample_factor > 1:
            sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
            ms = a.reshape(sh).mean(-1).mean(1)

        return ms
            
        
    def _evaluate(self, x, y, *args):
        """The model evaluated at the coordinates (x,y) without oversampling 
        or convolution."""
        return 0. * x


    def evaluate(self, x, y, *args):
        """Returns the evaluated model in the format that can be compared directly
        to data at the coordinates (x,y)."""
        return self._oversample_model(x,y,None,amplitude,x_0,y_0,fwhm,alpha)    
