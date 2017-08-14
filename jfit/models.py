import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline
from astropy.modeling import Fittable2DModel, Parameter
from astropy.convolution import convolve, convolve_fft

__all__ = ['ImageModel','Sersic2D']

class ImageModel(Fittable2DModel):
    """
    Abstract class for 2D shape models, from simple point-spread-functions
    (PSF) to more complex models.
    
    The class is deriving from the astropy.modeling class Fittable2DModel 
    which facilitates using the astropy.modeling framework for fitting the 
    model.  See the astropy.modeling documentation for details.
    
    The class supports over sampling when calculating the model, which is
    very useful for images with undersampled PSF.
    """

    def __init__(self, **kwargs):
        # the default over sampling factor
        super(ImageModel, self).__init__(**kwargs)

        # if a kernel is attached it will be used to convolve the calculated 
        # model before returning it, should always be a 2D numpy array.
        self._kernel = None   # the kernel model, should be derived from PSF
        self._k = None        # 2D numpy array generated from self._kernel

        self.oversample_factor(1)


    def get(self,parameter) :
        """Get the given model parameter to the value.  This can also be done
        as model.parameter, but this method is useful for looping over many 
        parameters."""
        return self.parameters[self.param_names.index(parameter)]
                                    
                                        
    def set(self,parameter,value) :
        """Set the given model parameter to the value.  This can also be done
        as model.parameter = value, but this method is useful for looping
        over many parameters."""
        self.parameters[self.param_names.index(parameter)] = value

                                                                                    
    def oversample_factor(self,factor):
        """Set the oversampling factor.  If a kernel is attached, it will
        be re-generated each time this method is called to make sure it always
        matches the oversampling factor."""
        self._sample_factor = factor
        self.set_kernel()

        return True


    def set_kernel(self,kern=None,khpw=None):
        """Set a diffusion kernel that will be used to convolve the
        evaluated model before returning it.  

          'kern' - should be an object instantiated from a subclass to 
                   the PSF class
          'khpw' - is the kernel patch half-width

        The kernel size should be chosen depending on the kernel FWHM,
        it will be automatically scaled internally to match the oversampling
        factor of the model.

        If the method is called without a model, the 2D kernel stored
        internally will be re-generated using the current oversampling
        factor"""

        # set the kernel model
        if kern is not None:
            if khpw is None :
                raise ValueError("Both the kernel model and the kernel size must "
                    "be specified!")

            self._kernel = kern
            self._khpw = khpw

        # generate the internal 2D-kernel
        if self._kernel is not None:
            factor = self._sample_factor
            self._k = self._kernel.kernel(self._khpw,factor)

        return True


    def _convolve(self,z) :
        """Convolve model with an attached kernel. FFT convolution will be
        used for kernel sizes exceeding 3x3."""
        if self._k is not None :
            k = self._k + 0.
        
            # if we are evaluating the model along a line (e.g. radially)
            # z could be one dimentionsional, then we want to convolve it
            # with a 1D kernel
            if z.ndim == 1 :
                nx,ny = self._k.shape
                k = self._k[:,(ny-1)/2] + 0. # assuming it is a radially symmetric kernel

            k /= k.sum()
            if len(k) > 9:
                return convolve_fft(z, k, normalize_kernel=True)
            else :
                return convolve(z, k, normalize_kernel=True)
        else :
            return z

    
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


    def _only_oversampled_model(self,x,y,sample_factor,*args):
        """Evaluate and return the oversampled model"""
        if sample_factor is None :
            sample_factor = self._sample_factor
        
        # over sample the input vectors
        if sample_factor > 1:
            xs,ys = self._oversample_input(x,y,sample_factor)
        else :
            xs,ys = x,y
        
        m = self._evaluate(xs,ys,*args)

        # the amplitude is normalized to the original sampling, and
        # in order to conserve flux we need to scale with the square
        # of the sampling factor
        shape = x.shape
        if len(shape) == 1:
            m /= sample_factor
        elif len(shape) == 2:
            m /= sample_factor*sample_factor
        else:
            raise ValueError("Dimensions of input arrays not supported!")

        return m


    def _oversampled_model(self,x,y,sample_factor,*args):
        """Evaluate and return the oversampled convolved model"""
        m = self._only_oversampled_model(x,y,sample_factor,*args)
        m = self._convolve(m)

        return m

    
    def _oversample_model(self,x,y,sample_factor,*args):
        """Calculate the model after first oversampling it and then
        rebinning it back to the original resolution."""
        if sample_factor is None :
            sample_factor = self._sample_factor

        a = self._oversampled_model(x,y,sample_factor,*args)

        # rebin model back to the original resolution
        shape = x.shape
        if sample_factor > 1:
            if len(shape) == 1:
                sh = shape[0],a.shape[0]//shape[0]
                ms = a.reshape(sh).sum(-1)
            elif len(shape) == 2:
                sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
                # ms = a.reshape(sh).mean(-1).mean(1)
                ms = a.reshape(sh).sum(-1).sum(1)
            else :
                raise ValueError("Dimension of input array not supported!")
        else :
            ms = a

        return ms
            
        
    def _evaluate(self, x, y, *args):
        """The model evaluated at the coordinates (x,y) without oversampling 
        or convolution."""
        return 0. * x


    def evaluate(self, x, y, *args):
        """Returns the evaluated model in the format that can be compared directly
        to data at the coordinates (x,y)."""
        return self._oversample_model(x,y,None,*args)


class Sersic2D(ImageModel):
    '''Two dimensional Sersic surface brightness profile.  This is
    similar to the astropy.modeling implementation but is taking
    advantage of ImageModel.

    Note the different argument order to the evaluate() and 
    __call__() methods compared to the the astropy.modeling 
    implementation.'''
    
    amplitude = Parameter(name='amplitude',default=1.)
    x_0 = Parameter(name='x_0',default=0.)
    y_0 = Parameter(name='y_0',default=0.)
    r_eff = Parameter(name='r_eff',default=1,min=1.e-8)
    n = Parameter(name='n',default=1,bounds=(0.5,10))
    ellip = Parameter(name='ellip',default=0.,bounds=(0.,1-1.e-8))
    theta = Parameter(name='theta',default=0.,bounds=(-np.pi,np.pi))
    _gammaincinv = None

#    def __init__(self, **kwargs):
#        self.sample_factor = 1
#        super(ImageModel, self).__init__(**kwargs)

    @classmethod
    def _evaluate(cls,x,y,amplitude,x_0,y_0,r_eff,n,ellip,theta):
        """Two dimensional Sersic profile function.  
        Stolen from astropy.modeling"""

        if cls._gammaincinv is None:
            try:
                from scipy.special import gammaincinv
                cls._gammaincinv = gammaincinv
            except ValueError:
                raise ImportError('Sersic2D model requires scipy > 0.11.')

        bn = cls._gammaincinv(2. * n, 0.5)
        a, b = r_eff, (1 - ellip) * r_eff
        cos_theta, sin_theta = np.cos(theta), np.sin(theta)
        x_maj = (x - x_0) * cos_theta + (y - y_0) * sin_theta
        x_min = -(x - x_0) * sin_theta + (y - y_0) * cos_theta
        z = np.sqrt((x_maj / a) ** 2 + (x_min / b) ** 2)

        return amplitude * np.exp(-bn * (z ** (1 / n) - 1))

