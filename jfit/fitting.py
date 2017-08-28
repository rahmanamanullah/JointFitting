from __future__ import print_function

from astropy.modeling.optimizers import (DEFAULT_MAXITER, DEFAULT_EPS, DEFAULT_ACC)
from astropy.modeling.fitting import _validate_model
from cost import JointCostFunctor, _validate_shape, _determine_nsets

__all__ = ['JointMinuitFitter']

"""
A general class for fitting joint fitting
"""
class JointMinuitFitter(object) :
    supported_constraints = ['bounds', 'fixed']

    def __init__(self, verbose=False):
        self.verbose = verbose
        self.minuit = None
        self.m = self.minuit

        
        
    def __call__(self, model, x, y, z=None, weights=None, common_names=[], maxiter=DEFAULT_MAXITER):
        try:
            import iminuit
        except ImportError:
            raise ValueError("Minimization method 'minuit' requires the "
                             "iminuit package")
 
        if not _validate_shape(x,y,z,weights):
            raise ValueError("All data and weight arrays must have the same shape!")

        # the number of data sets
        nsets = _determine_nsets(x,z)        
        
        # check input arguments
        if isinstance(model,list) and nsets != len(model):
            raise ValueError("The number of models in the list must match the length "
                             "of the first dimension of the data arrays.")
        elif not isinstance(model,list) and nsets != 1:
            raise ValueError("If single model is passed, the dimensions of the data "
                             "arrays must be either 1 or 2.")

        # Set up keyword arguments to pass to Minuit initializer.
        kwargs = {}
        modelc = []
        vparam_names = []
        for n in range(nsets):
            if nsets == 1:
                model_copy = _validate_model(model, self.supported_constraints)
            else:
                model_copy = _validate_model(model[n], self.supported_constraints)
            modelc.append(model_copy)
                
            fixed,bounds = model_copy.fixed,model_copy.bounds
            for name in model_copy.param_names:
                if name in common_names and n > 0:
                    continue
                elif name not in common_names and nsets > 1:
                    pname = "%s_%d"%(name,n)
                elif nsets > 1:
                    pname = "%s_0"%(name)
                else:
                    pname = name
                kwargs[pname] = model_copy.parameters[model_copy.param_names.index(name)]  # Starting point.

                vparam_names.append(pname)

                # Fix parameters not being varied in the fit.
                if fixed[name] :
                    kwargs['fix_' + pname] = True
                    kwargs['error_' + pname] = 0.
                    continue
                
                # Bounds and initial step size
                if kwargs[pname] != 0. :
                    step = 0.1 * kwargs[pname]
                else :
                    step = 0.1
            
                b1,b2 = bounds[name]
                if b1 is not None and b2 is not None :
                    kwargs['limit_' + pname] = bounds[name]
                    step = 0.02 * (b2 - b1)
                elif (b1 is None and b2 is not None) or (b1 is not None and b2 is None) :
                    raise ValueError('one-sided bounds not allowed for '
                                     'minuit minimizer')
                kwargs['error_' + pname] = step
                
        if self.verbose:
            print("Initial parameters:")
            for name in vparam_names:
                print(name, kwargs[name], 'step=', kwargs['error_' + name],
                      end=" ")
                if 'limit_' + name in kwargs:
                    print('bounds=', kwargs['limit_' + name], end=" ")
                print() 

        # setup the cost function, the parameter names passed here are the parameter
        # names for the invividual models.
        costfnc = JointCostFunctor(model_copy.evaluate,x,y,z,weights,
                                   common_names=common_names,
                                   param_names=model_copy.param_names)

        m = iminuit.Minuit(costfnc, errordef=1.,
                           print_level=(1 if self.verbose else 0),
                           throw_nan=True, **kwargs)

        d, l = m.migrad(ncall=maxiter)
        if m.ncalls == maxiter :
            raise Warning('Maximum number of calls reached for migrad!')
        self.minuit = m

        # set all model parameters to the fitted values
        for n in range(nsets):
            model_copy = modelc[n]
            for name in model_copy.param_names:
                if nsets > 1 and name not in common_names:
                    pname = "%s_%d"%(name,n)
                if nsets > 1:
                    pname = "%s_0"%(name)
                else:
                    pname = name
                model_copy.parameters[model_copy.param_names.index(name)] = m.values[pname]
                
        if nsets == 1:
            modelc = modelc[0]
         
        return modelc
