Package that provides the structure for joint and simulataneous fitting. That
is, a single model can be jointly fitted to a series of 1D or 2D data where
the user can decide which of the model parameters that should be fitted
individually for each data set (e.g. an intercept or a bias) and which that
should be fitted jointly (e.g. physical parameters).

The package also that provides a class for fitting models from 
astropy.modeling using iminuit.  This can be used both for simple fitting and
for joint fitting.
