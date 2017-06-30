from __future__ import print_function
import numpy as np

# these iminuit modules are only used for dynamically 
# finding the function signature
from iminuit import describe
from iminuit.util import make_func_code

__all__ = ['JointCostFunctor']

def _validate_shape(x,y,z=None,w=None):
    """
    Validate that all arrays have the same dimensions and lengths.
    """    
    s = x.shape
    valid = True
    for m in [y,z,w]:
        if m is None:
            continue
        ts = m.shape
        if len(ts) != len(s):
            valid = False
        for n in range(len(s)):
            if ts[n] != s[n]:
                valid = False
    return valid


def _not_in_list(list1,list2):
    """
    Return a list of the parameter names in list1 that are not present in list2.
    """
    list3 = []
    for item in list1:
        if item not in list2:
            list3.append(item)
    return list3


def _determine_nsets(x,z):
    """Determine the number of data sets where 'x' and 'y' has the same meaning
    as in the cost function and both should be passed, since the results depends
    on wheter we are doing 1D or 2D fitting"""
    xx = x.squeeze().shape
    nsets = 1
    if z is None:
        if len(xx) == 1:
            nsets = 1
        elif len(xx) == 2:
            nsets = xx[0]
        else :
            raise ValueError('For 1D fitting the dimensions of the data '
                             'arrays must be <= 2.')
    else:
        if len(xx) == 2:
            nsets = 1
        elif len(xx) == 3:
            nsets = xx[0]
        else:
            raise ValueError('For 2D fitting the dimensions of the data '
                             'arrays must be either 2 or 3.')
    return nsets


def _all_param_names(nsets,param_names,common_names):
    """Return a list with all the parameter names, given the number of data sets and
    a list of common parameters.    
    """
    indiv_names = _not_in_list(param_names,common_names)
    ncommon = len(common_names)
    if ncommon > 0 and nsets == 1:
        raise ValueError("Cannot fit common parameters to a single data set.")
    elif ncommon == 0 and nsets == 1:               # fitting to a single data set
        param_names_full = param_names
    else:
        param_names_full = []
        for n in range(nsets):
            for name in indiv_names:
                param_names_full.append("%s_%d"%(name,n))
        param_names_full += common_names
    return param_names_full
    
    
class JointCostFunctor(object):
    """
    Joint fitting of both common and individual data sets to a data set. 
    The __call__ method calculates the cost function given a set of parameters.
    
    This class is compatible with iminuit although it does not specifically depend on it,
    """
    def __init__(self,f,x,y,z=None,w=None,common_names=[],param_names=None,
                 verbose=False,lamb=0.0):
        """
        The class is initialzed with a model function 'f' and measured 'x' and 'y' data
        (optional 'z' for 2D fitting).  If a weight array is passed this will be taken
        into account when calculating the cost.
        
        The dimensions of the x,y,z and w arrays must be the same.  Several independent
        data series can be passed by, and the first dimension of the arrays will then
        be assumed to run over the different data sets.  
        
        In the default execution all parameters of the model 'f' will be fit to each data 
        set indpedently.  In other words, this is identical to fit the model to each data
        set, one at a time.
        
        However, if some parameters are passed as 'common_names' these will be jointly fit
        to all data series.
        """
        self.f  = f
        self.x  = x
        self.y  = y
        self.z  = z
        self.w  = w
        self.lamb = lamb # for future regularization (not yet implemented)
        self.common_names = common_names
        self.verbose = verbose

        if not _validate_shape(x,y,z,w):
            return ValueError("All data and weight arrays must have the same shape!")
        
        # determine the parameter names if they have not been passed manually
        # if 'f' itself is a function where the parameters are setup dynamically
        # this is likely to fail...
        if param_names is None:
            f_sig = describe(f)
            param_names = f_sig[1:]                     # docking off independent variable
            if len(param_names) == 0:
                raise ValueError("Failed to determine function signature, "
                                 "try to pass it manually to the constructor.")
                    
        # determine the number of datasets
        nsets = _determine_nsets(y,z)
        self.nsets = nsets
                                
        # if we are doing joint fitting, the last 'ncommon' parameters are
        # the same for each data set, while the first nparam-ncommon are fitted
        # individually to each set
        ncommon = len(common_names)
        nparam  = len(param_names)
        
        # setup a list of the parameters that should be fitted
        # to each individual data set
        indiv_names = _not_in_list(param_names,common_names)
        self.indiv_names = indiv_names

        # the full parameter list will contain the individual list first and
        # the common parameters in the end, e.g. a_1,c_1,a_2,c_2,b, but the
        # calling sequence to the fitting model is typically (a,b,c)
        # therefore we need to setup an index vector that can be used to
        # sort the argument list in the correct order once it is pulled out
        # so that in the example above (a_1,c_1,b) becomes (a_1,b,c_1)
        self.index = []
        i1,i2 = 0,len(indiv_names)
        for n,name in enumerate(param_names):
            if name in common_names:
                self.index.append(i2)
                i2 += 1
            else:
                self.index.append(i1)
                i1 += 1
        
        # setup the full parameter list that should be fitted, this will have the
        # length: nsets*(nparam-ncommon) + ncommon
        param_names_full = _all_param_names(nsets,param_names,common_names)
         
        # setup the function signature
        self.func_code = make_func_code(param_names_full)
        self.func_defaults = None                       # this keeps np.vectorize happy

                
    def __call__(self,*arg):
        """
        The cost is calculated given the parameters.  Here the number of parameters will equal
        'nset*(nparam-ncommon) + ncommon', where 'nset' is the number of data sets, 'nparam'
        is the number of parameters of the model and 'ncommon' are the number of parameters 
        that will be jointly fitted.
        """
        nindiv = len(self.indiv_names)
        chisq = 0.
        for n in range(self.nsets):
            
            # identify which parameters in the input that should be used for 
            # calling the model on this specific data set, if we only are
            # working with one data set this is trivial
            if self.nsets == 1:
                targ = arg

                x,y = self.x,self.y
                if self.z is None:
                    measured_vals = y
                else:
                    measured_vals = self.z
                if self.w is not None:
                    w = self.w
            else:
                # pull out the individual parameters that are relevant for
                # this given data set and add the common parameters which are
                # always at the end.
                i = n*nindiv
                targ  = arg[i:i+nindiv]
                targ += arg[self.nsets*nindiv:]
                                
                # now the tuple must be reshuffled to match the call order
                # for the model that we are trying to fit.
                targ = list(targ)
                targ.sort(key=dict(zip(targ,self.index)).get)
                targ = tuple(targ)
                
                x,y = self.x[n,:],self.y[n,:]
                if self.z is None:
                    measured_vals = y
                else:
                    measured_vals = self.z[n,:]
                if self.w is not None:
                    w = self.w[n,:]
                    
            if self.z is None :
                model_vals = self.f(x,*targ)
            else :
                model_vals = self.f(x,y,*targ)            

            if self.w is None :
                chisq += np.sum((measured_vals - model_vals)**2)
            else :
                chisq += np.sum( w * (measured_vals - model_vals)**2 )
            
        return chisq
