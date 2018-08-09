# ------------------------------------------------------------------------------------------------------------------
# This program computes three lower-order logarithmic moments of a relaxation time distribution. The original method has been
# described in [R. Zorn, J. Chem. Phys. 116 (2002) 3204-3209]. It is based on the moment rule of convolution and allows
# one to measure the geometric mean, the geometric standard deviation and a skewness parameter of an underlying relaxation 
# time distribution without first calculating a distribution function proper. Two prerequisites to the method are (i) the 
# sampling of the relaxation function in logarithmically equidistant steps and (ii) its scaling to [1, 0]. With those things
# done, the logarithmic moments are calculated straightforwardly through numerical integration of the relaxation function. 
# For more details, see [O. V. Petrov, S. Stapf, J. Magn. Reson. 279 (2017) 29-38].
# ------------------------------------------------------------------------------------------------------------------
# For data scaling purposes, a model function is fitted to the original data (by default, a triple-exponential function is
# used). The best-fit amplitude and offset parameters of the model function are then used to normalize the data to [1, 0]
# as the method prescribes. Given the best-fit parameters, one can also extrapolate the relaxation function toward zero time
# to provide numerical integrals a better approximation. Although optional, the extrapolation is highly recommended to execute.
#
# The module logarithmic_moment_analysis.py contains the function logmoments(), which returns three logarithmic moments as 
# described above, and two auxiliary functions, decay() and recovery(), which define model functions for data extrapolation. 
# 
# Parameters: x: array-like
#                Input xdata, supposed to be logarithmically equispaced time points
#             y: array-like
#                Input ydata, a signal intensity measured at respective time points
#             y_fit: array-like, optional 
#                A pointer to the best-fit model function to be sought inside logmoments(); a hidden return object for visual 
#                control and plotting in a function-caller (default: [])
#             extrapol: boolean, optional
#                Defines whether to execute data extrapolation; if True, n original data points are extended by n model points
#                at the shorter-time side of the relaxation function (default: True)
#             plotdata: boolean, optional
#                Defines whether to plot relaxation functions and their best-fit curves in course of the log-moment calulation 
#             verbose: boolean, optional
#                Defines whether to output the calculated quantities to the console (default: True)
#             **fit_kws: dictionary, optional
#                Options to pass to the minimizer being used. If lmfit_exponential.py is being used, the dictinary is supposed 
#                to include three options: 'order' , 'reusable' and 'verbose', where order defines the order of exponential fit 
#                function (default: 3); reusable defines whether to cash the best-fit parameters to re-use them as initial values 
#                in a sequential fitting (default: False), verbose means reporting fit results (default: False).
# Returns: 
#             A list of three central lower-order logarithmic moments. By default, the list is also saved in a text file 
#             logmoments.txt in a working diectory. An implicit return is possible via the optional parameter y_fit which 
#             points to the best-fit model of the relaxation function for visual control of data normalization from within 
#             a calling function.
#
# v1.0 August 2017
# Author: Oleg V. Petrov, TU Ilmenau
# E-mail: oleg.petrov@tu-ilmenau.de

import numpy as np 
import scipy.optimize
import scipy.integrate
import matplotlib.pyplot as plt
import lmfit_exponential 

def logmoments(x, y, extrapol=True, plotdata=False, verbose=False, **fit_kws):
    x_orig = np.copy(x)
    y_orig = np.copy(y)

   # fit exponential:
    fit_options = fit_kws.get('fit_options')
    fit_result = lmfit_exponential.minimize(x, y, options=fit_options)
    fit_params = fit_result.params 
    y_fit = y + fit_result.residual

    # extrapolate (double the number of shortest-time points):
    if extrapol == True:
        a = np.log10(x)
        b = a[0] - np.cumsum(np.diff(a)) 
        # c = a[-1] + np.cumsum(np.diff(a)) 
        x = 10**np.concatenate((b[::-1], a))#, c))
        y = np.concatenate((func(fit_params, 10**b[::-1]), y))#, (func(fit_params, 10**c))))
        x_extrap = np.copy(x)
        y_extrap = np.copy(y)

    # normalize the data:
    y -= fit_params['offset']
    y /= fit_params['a1']+fit_params['a2']+fit_params['a3']

    # compute logarithmic moments: 
    res1 = np.log(x[0]) + scipy.integrate.simps(y, x=np.log(x))
    res2 = np.log(x[0])**2 + 2.*scipy.integrate.simps(y*np.log(x), x=np.log(x))
    res3 = np.log(x[0])**3 + 3.*scipy.integrate.simps(y*np.log(x)**2, x=np.log(x))

    logm1 = 0.5772 + res1								# the arithmetic mean on the log scale
    logm2 = res2 - res1**2 - np.pi**2/6.				# the 2nd central moment (the variance)
    logm3 = res3 - 3.*res2*res1 + 2.*res1**3 + 2.404	# the 3rd central moment 

    if verbose == True:
        print "First three central moments of a logarithmic relaxation time disribution:"
        print "   - arithmetic mean on the log-scale: ", logm1
        print "   - 2nd central moment (variance): ", logm2
        print "   - 3rd central moment: ", logm3
        print "   - skewness parameter: ", logm3/logm2**1.5
        print ""
        # np.savetxt('C:/Documents and Settings/Oleg Petrov/My Documents/Projects/IDentIFY/sensitivitytest.dat', [logm1, logm2, logm3])#, 0, logm3/(np.sqrt(logm2)**3)])

        print "Corresponding moments on the linear time scale:"
        print "   - geometric mean, in sec: ", np.exp(logm1)
        print "   - geometric standard deviation: ", np.exp(np.sqrt(logm2))

    # plot the data:
    if plotdata == True:
        f1, ax1 = plt.subplots(1, sharex=False, sharey=False)
        if extrapol == True:
            ax1.semilogx(x_extrap[x_extrap.size/2:], y_extrap[x_extrap.size/2:], 'bo')
            ax1.semilogx(x_extrap[:x_extrap.size/2], y_extrap[:x_extrap.size/2], 'ro')
        else:	
            ax1.semilogx(x_orig, y_orig, 'bo')
        ax1.semilogx(x_orig, y_fit, 'r')
        ax1.set_title('experimental & extrapolated data')
        plt.show()

    return [logm1, logm2, logm3], y_fit

# fit functions used for data extrapolation:
def func(params, x):
    a1 = params['a1']
    a2 = params['a2']
    a3 = params['a3']
    r1 = params['r1']
    r2 = params['r2']
    r3 = params['r3']
    offset = params['offset']
    return a1*np.exp(-r1*x) + a2*np.exp(-r2*x) + a3*np.exp(-r3*x) + offset
