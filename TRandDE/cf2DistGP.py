"""
Original matlab file for TR: Viktor Witkovsky, https://github.com/witkovsky/CharFunTool
Python conversion and modification: Andrej Gajdos, https://github.com/fdslrm
"""

import numpy as np 
import time 
import matplotlib.pyplot as plt 
import warnings
# import math


def ChebPoints(N, interval=None):
    
    if interval is None: 
        interval = []
    
    if len(interval) == 0:
        interval = [-1,1]
    elif len(interval) != 2:
        raise ValueError('Input parameter interval should be a list of length 2, containing lower and upper bound.')
        
    a = interval[0]
    b = interval[1]
    pts = (a + b) / 2 - np.cos(np.pi * (np.arange(0,N+1) / N)) * (b - a) / 2
    
    return(pts)

    
def interpBarycentric(x, fun, xNew=None, options=None): 
    
    if xNew is None: 
        xNew = []
    
    if options is None: 
        options = {}
    
    if len(xNew) == 0: 
        xNew = np.arange(np.min(x), np.max(x))
    
    if 'isChebPts' not in options:
        options.update({'isChebPts': True}) 
        
    x = np.asarray(x)
    x = x.flatten() 
    fun = np.asarray(fun) 
    fun = fun.flatten() 
    
    xNew = np.asarray(xNew)
    sztNew = xNew.shape
    xNew = xNew.flatten()
    nx = len(x)
    nxNew = len(xNew)
    funNew = np.zeros(nxNew) 
    
    w = np.power(-1, np.arange(0,nx))
    w = np.double(w)
    w[0] = w[0] / 2
    w[-1] = w[-1] / 2 
    
    for i in range(0, len(xNew)):
        A = 0
        B = 0 
        for j in range(0, nx):
            if xNew[i] == x[j]:
                exactVal = True
                funNew[i] = fun[j]
            else:
                exactVal = False
                weight = w[j] / (xNew[i] - x[j])
                A = A + weight * fun[j]
                B = B + weight
            
            if exactVal == True:
                break
            else:
                funNew[i] = A / B

    xNew = np.reshape(xNew, sztNew)
    funNew = np.reshape(funNew, sztNew)

    return({'xNew':xNew, 'funNew':funNew})

# ## EXAMPLE 1
# # Barycentric interpolant of the Sine function on (-pi,pi)
# x = ChebPoints(32, [-np.pi, np.pi])
# f = np.sin(x)
# xNew = np.linspace(-np.pi, np.pi, 201)
# fNew = interpBarycentric(x, f, xNew)['funNew']
# plt.plot(xNew, fNew, color='blue', linewidth=2)
# plt.scatter(x, f, color='red', s=20) 
# plt.xlabel('')
# plt.ylabel('')
# plt.title('')
# print({'xNew':xNew, 'fNew':fNew, 'sin(xNew)':np.sin(xNew)})


# ## EXAMPLE 2
# # Barycentric interpolant of the Normal CDF
# from scipy.stats import norm
# x = ChebPoints(21, [-8, 8])
# f = norm.cdf(x)
# xNew = np.linspace(-8, 8, 201)
# fNew = interpBarycentric(x, f, xNew)['funNew']
# plt.plot(xNew, fNew, color='blue', linewidth=2)
# plt.scatter(x, f, color='red', s=20) 
# plt.xlabel('')
# plt.ylabel('')
# plt.title('')
# print({'xNew':xNew, 'fNew':fNew, 'norm.cdf(xNew)':norm.cdf(xNew)})


# ## EXAMPLE 3
# # Barycentric interpolant of the Normal quantile function
# x = ChebPoints(2**7, [-8, 8])
# cdf = norm.cdf(x)
# prob = np.linspace(f, 1 - 1e-4, 101)
# qf = interpBarycentric(cdf, x, prob)['funNew'] 
# plt.plot(prob, qf, color='blue', linewidth=2)
# plt.scatter(cdf, x, color='red', s=10) 
# plt.xlabel('')
# plt.ylabel('')
# plt.title('')
# print({'prob':prob, 'qf':qf, 'norm.ppf':norm.ppf(prob)})


def cfN_Binomial(t=None, n = 10, p = 1/2, cfX=None):
    ## CHECK THE INPUT PARAMETERS
    if t is None: 
        raise ValueError('Enter input parameter t.')
    
    ## Characteristic function of the (compound) Binomial distribution
    t = np.asarray(t)
    szt = t.shape
    t = t.flatten() 
    
    if cfX is None:
        expit = np.exp(1j * t)
    else:
        expit = cfX(t)
    
    cf = (1 - p + p * expit) ** n
    cf[t == 0] = 1 
    
    cf = np.reshape(cf, szt) 
    
    return(cf)

## test 
n = 25
p = 0.3
t = np.linspace(-15, 15, 1001)
x = cfN_Binomial(t=t, n=n, p=p)


def cfN_Poisson(t=None, lamb=1, cfX=None):
    ## CHECK THE INPUT PARAMETERS
    if t is None: 
        raise ValueError('Enter input parameter t.')
    
    ## Characteristic function of the (compound) Poisson distribution
    t = np.asarray(t)
    szt = t.shape
    t = t.flatten()
    
    if cfX is None:
        expit = np.exp(1j * t)
    else:
        expit = cfX(t)
    
    cf = np.exp(lamb * (expit - 1))
    cf[t == 0] = 1 
    
    cf = np.reshape(cf, szt) 
    
    return(cf)

## test 
lamb = 10
t = np.linspace(-10, 10, 501)
x = cfN_Poisson(t=t, lamb=lamb)


def firstOccIdx(x):
    x_unique = np.sort(list(set(x)))
    indices = []
    for i in range(0,len(x_unique)):
        idx = 1
        for j in range(0, len(x)):
            if x[j] == x_unique[i]:
                idx = j
                indices.append(idx)
                break
    
    return(indices)




def cf_Exponential(t=None, lamb=None, coef=None, niid=None):
    ## CHECK THE INPUT PARAMETERS
    if t is None: 
        raise ValueError('Enter input parameter t.')
        
    if lamb is None: 
        lamb = []
    
    if coef is None: 
        coef = []
    
    if niid is None: 
        niid = []
    
    if len(lamb) == 0:
        lamb = [1]
    
    if len(coef) == 0:
        coef = [1]
    
    if len(niid) == 0:
        niid = [1] 
    
    # Equal size of the parameters
    if len(coef) > 0 and len(lamb) == 1 and len(niid) == 0:
        coef = np.sort(coef)
        m = len(coef)
        coef = list(set(coef))
        idx = firstOccIdx(coef)
        lamb <- lamb * np.diff(idx, m + 1) 
        
    l_max = np.max([len(lamb), len(coef)])
    if l_max > 1:
        if(len(lamb) == 1):
            lamb = lamb * l_max
        if len(coef) == 1:
            coef = coef * l_max 
        if len(coef) < l_max or len(lamb) < l_max:
            raise ValueError('Input size mismatch.')
    
    # Special treatment for linear combinations with large number of RVs
    coef = np.asarray(coef)
    szcoefs = coef.shape
    if len(szcoefs) == 1:
        szcoefs = szcoefs[0]
    else:
        szcoefs = szcoefs[0] * szcoefs[1]
    t = np.asarray(t)
    szt = t.shape
    sz = 0
    if len(szt) == 1:
        sz = szt[0]
    else:
        sz = szt[0] * szt[1]      
    szcLimit = np.ceil(1e3 / (sz / (2 ** 16)))
    idc = np.arange(1, int(szcoefs / szcLimit) + 2)
    
    ## Characteristic function
    idx0 = 0
    cf = 1
    lamb = np.asarray(lamb)
    
    for j in range(0,idc[-1]):
        idx1 = np.min([idc[j] * szcLimit, szcoefs])
        idx = np.arange(idx0,idx1)
        idx = idx.astype(int)
        idx0 = idx1 + 1
        aux = np.outer(t, coef[idx] / lamb[idx])
        cf = np.multiply(cf, np.multiply.reduce(1/(1 - 1j * aux), axis=1))
    
    cf = np.reshape(cf, szt)
    cf[t == 0] = 1
    
    if len(niid) > 0:
        if len(niid) == 1:
            cf = cf ** niid
        else:
            raise ValueError('niid should be a scalar (positive integer) value')
    
    return(cf)

  



def cfX_Exponential(t=None, lamb=None, coef=None, niid=None):
    
    if t is None: 
        raise ValueError('Enter input parameter t.')
    
    if lamb is None: 
        lamb = []
    
    if coef is None: 
        coef = []
    
    if niid is None: 
        niid = []
    
    cf = cf_Exponential(t, lamb, coef, niid) 
    
    return(cf)

# ## test 1
# lamb = [5]
# t = np.linspace(-50, 50, 501)
# x = cfX_Exponential(t=t, lamb=lamb)
  
# ## test 2
# coef = 1 / ((np.arange(1,51) - 0.5) * np.pi) ** 2
# lamb = [5]
# t = np.linspace(-100, 100, 201)
# x = cfX_Exponential(t=t, lamb=lamb, coef=coef)

#' @title
#' Evaluating CDF/PDF/QF from CF of a continous distribution F by using the Gil-Pelaez inversion formulae.
#'
#' @description
#' \code{cf2DistGP(cf, x, prob, options)} calcuates the CDF/PDF/QF from the Characteristic Function CF
#' by using the Gil-Pelaez inversion formulae:
#' \deqn{cdf(x) = 1/2 + (1/\pi) * Integral_0^inf imag(exp(-1i*t*x)*cf(t)/t)*dt,}
#' \deqn{pdf(x) = (1/\pi) * Integral_0^inf real(exp(-1i*t*x)*cf(t))*dt.}
#'
#' The FOURIER INTEGRALs are calculated by using the simple TRAPEZOIDAL QUADRATURE method, see below.
#'
#' @family CF Inversion Algorithm
#'
#' @importFrom stats runif
#' @importFrom graphics plot grid
#'
#' @seealso For more details see:
#' \url{https://arxiv.org/pdf/1701.08299.pdf}.
#'
#' @param cf      function handle for the characteristic function CF.
#' @param x       vector of values where the CDF/PDF is computed.
#' @param prob    vector of values from \eqn{[0,1]} for which the quantile function is evaluated.
#' @param options  structure (list) with the following default parameters:
#' \itemize{
#'     \item \code{options$isCompound = FALSE} treat the compound distributions, of the RV \eqn{Y = X_1 + ... + X_N},
#'     where \eqn{N} is discrete RV and \eqn{X\ge0} are iid RVs from nonnegative continuous distribution,
#'     \item \code{options$isCircular = FALSE} treat the circular distributions on \eqn{(-\pi, \pi)},
#'     \item \code{options$isInterp = FALSE} create and use the interpolant functions for PDF/CDF/RND,
#'     \item \code{options$N = 2^10} N points used by FFT,
#'     \item \code{options$xMin = -Inf} set the lower limit of \code{x},
#'     \item \code{options$xMax = Inf} set the upper limit of \code{x},
#'     \item \code{options$xMean = NULL} set the MEAN value of \code{x},
#'     \item \code{options$xStd = NULL} set the STD value of \code{x},
#'     \item \code{options$dt = NULL} set grid step \eqn{dt = 2*\pi/xRange},
#'     \item \code{options$T = NULL} set upper limit of \eqn{(0,T)}, \eqn{T = N*dt},
#'     \item \code{options$SixSigmaRule = 6} set the rule for computing domain,
#'     \item \code{options$tolDiff = 1e-4} tol for numerical differentiation,
#'     \item \code{options$isPlot = TRUE} plot the graphs of PDF/CDF,
#'
#'     \item options$DIST                   list with information for future evaluations,
#'                                         \code{options$DIST} is created automatically after first call:
#'     \itemize{
#'         \item \code{options$DIST$xMin} the lower limit of \code{x},
#'         \item \code{options$DIST$xMax} the upper limit of \code{x},
#'         \item \code{options$DIST$xMean} the MEAN value of \code{x},
#'         \item \code{options$DIST$cft} CF evaluated at \eqn{t_j} : \eqn{cf(t_j)}.
#'         }
#'     }
#'
#' @return
#' \item{result}{structure (list) with CDF/PDF/QF amd further details.}
#'
#' @details
#' If \code{options.DIST} is provided, then \code{cf2DistGP} evaluates CDF/PDF based on
#' this information only (it is useful, e.g., for subsequent evaluation of
#' the quantiles). \code{options.DIST} is created automatically after first call.
#' This is supposed to be much faster, bacause there is no need for further
#' evaluations of the characteristic function. In fact, in such case the
#' function handle of the CF is not required, i.e. in such case do not specify \code{cf}.
#'
#' The required integrals are evaluated approximately by using the simple
#' trapezoidal rule on the interval \eqn{(0,T)}, where \eqn{T = N * dt} is a sufficienly
#' large integration upper limit in the frequency domain.
#'
#' If the optimum values of \eqn{N} and \eqn{T} are unknown, we suggest, as a simple
#' rule of thumb, to start with the application of the six-sigma-rule for
#' determining the value of \eqn{dt = (2*\pi)/(xMax-xMin)}, where \eqn{xMax = xMean +}
#' \eqn{6*xStd}, and \eqn{xMin = xMean - 6*xStd}, see \code{[1]}.
#'
#' Please note that THIS (TRAPEZOIDAL) METHOD IS AN APPROXIMATE METHOD:
#' Frequently, with relatively low numerical precision of the results of the
#' calculated PDF/CDF/QF, but frequently more efficient and more precise
#' than comparable Monte Carlo methods.
#'
#' However, the numerical error (truncation error and/or the integration
#' error) could be and should be properly controled!
#'
#' CONTROLING THE PRECISION:
#' Simple criterion for controling numerical precision is as follows: Set \eqn{N}
#' and \eqn{T = N*dt} such that the value of the integrand function
#' \eqn{Imag(e^(-1i*t*x) * cf(t)/t)} is sufficiently small for all \eqn{t > T}, i.e.
#' \eqn{PrecisionCrit = abs(cf(t)/t) <= tol},
#' for pre-selected small tolerance value, say \eqn{tol = 10^-8}. If this
#' criterion is not valid, the numerical precission of the result is
#' violated, and the method should be improved (e.g. by selecting larger \eqn{N}
#' or considering other more sofisticated algorithm - not considered here).
#'
#' @references
#' [1] WITKOVSKY, V.: On the exact computation of the density and
#' of the quantiles of linear combinations of t and F random variables.
#' Journal of Statistical Planning and Inference, 2001, 94, 1-13.
#'
#' [2] WITKOVSKY, V.: Matlab algorithm TDIST: The distribution
#' of a linear combination of Student's t random variables.
#' In COMPSTAT 2004 Symposium (2004), J. Antoch, Ed., Physica-Verlag/Springer 2004,
#' Heidelberg, Germany, pp. 1995-2002.
#'
#' [3] WITKOVSKY, V., WIMMER, G., DUBY, T. Logarithmic Lambert W x F
#' random variables for the family of chi-squared distributions
#' and their applications. Statistics & Probability Letters, 2015, 96, 223-231.
#'
#' [4] WITKOVSKY, V.: Numerical inversion of a characteristic function:
#' An alternative tool to form the probability distribution
#' of output quantity in linear measurement models. Acta IMEKO, 2016, 5(3), 32-44.
#'
#' [5] WITKOVSKY, V., WIMMER, G., DUBY, T. Computing the aggregate loss distribution
#' based on numerical inversion of the compound empirical characteristic function
#' of frequency and severity. ArXiv preprint, 2017, arXiv:1701.08299.
#'
#' @note Ver.: 16-Sep-2018 17:59:07 (consistent with Matlab CharFunTool v1.3.0, 22-Sep-2017 11:11:11).
#'
#' @example R/Examples/example_cf2DistGP.R
#'
#' @export
#'

def cf2DistGP(cf, x=None, prob=None, options=None): 
    
    ## CHECK THE INPUT PARAMETERS
    start_time = time.time() 
    
    cf = cf 
    x = x  
    prob = prob 
    options = options 
    
    if x is None: 
        x = []

    if prob is None: 
        prob = []      

    if options is None: 
        options = {} 

    if 'isCompound' not in options: 
        options.update({'isCompound': False}) 

    if 'isCircular' not in options: 
        options.update({'isCircular': False})

    if 'N' not in options:
        if options['isCompound'] == True:
            options.update({'N': 2**14})
        else: 
            options.update({'N': 2**10})
    
    if 'xMin' not in options:
        if options['isCompound'] == True: 
            options.update({'xMin': 0}) 
        else:
            options.update({'xMin': -np.inf})
    
    if 'xMax' not in options: 
        options.update({'xMax': np.inf})
    
    if 'xMean' not in options: 
        options.update({'xMean': []})

    if 'xStd' not in options: 
        options.update({'xStd': []})

    if 'dt' not in options: 
        options.update({'dt': []})
    
    if 'T' not in options: 
        options.update({'T': []})

    if 'SixSigmaRule' not in options: 
        if options['isCompound'] == True:
            options.update({'SixSigmaRule': 10})
        else:
            options.update({'SixSigmaRule': 6}) 
        
    if 'tolDiff' not in options: 
        options.update({'tolDiff': 1e-04})

    if 'crit' not in options: 
        options.update({'crit': 1e-12}) 

    if 'isPlot' not in options: 
        options.update({'isPlot': True})

    if 'DIST' not in options: 
        options.update({'DIST': {}})

    # Other options parameters
    if 'qf0' not in options: 
        options.update({'qf0': np.real((cf(1e-4) - cf(-1e-4)) / (2e-4 * 1j))})
    
    if 'maxiter' not in options: 
        options.update({'maxiter': 1000})

    if 'xN' not in options: 
        options.update({'xN': 101})

    if 'chebyPts' not in options: 
        options.update({'chebyPts': 2**9})

    if 'correctedCDF' not in options:
        if options['isCircular'] == True:
            options.update({'correctedCDF': True})
        else: 
            options.update({'correctedCDF': False})

    if 'isInterp' not in options: 
        options.update({'isInterp': False})

    ## GET/SET the DEFAULT parameters and the OPTIONS
    cfOld = []

    if len(options['DIST']) > 0: 
        xMean = options['DIST']['xMean']
        cft = options['DIST']['cft']
        xMin = options['DIST']['xMin']
        xMax = options['DIST']['xMax']
        cfLimit = options['DIST']['cfLimit']
        xRange = xMax - xMin
        dt = 2 * np.pi / xRange
        N = len(cft) 
        t = np.arange(1,N+1) * dt 
        xStd = []
    else: 
        N = options['N']
        dt = options['dt']
        T = options['T']
        xMin = options['xMin']
        xMax = options['xMax']
        xMean = options['xMean']
        xStd = options['xStd']
        SixSigmaRule = options['SixSigmaRule']
        tolDiff = options['tolDiff']

      # Special treatment for compound distributions. If the real value of CF at infinity (large value)
      # is positive, i.e. cfLimit = real(cf(Inf)) > 0. In this case the
      # compound CDF has jump at 0 of size equal to this value, i.e. cdf(0) =
      # cfLimit, and pdf(0) = Inf. In order to simplify the calculations, here we
      # calculate PDF and CDF of a distribution given by transformed CF, i.e.
      # cf_new(t) = (cf(t)-cfLimit) / (1-cfLimit); which is converging to 0 at Inf,
      # i.e. cf_new(Inf) = 0. Using the transformed CF requires subsequent
      # recalculation of the computed CDF and PDF, in order to get the originaly
      # required values: Set pdf_original(0) =  Inf & pdf_original(x) = pdf_new(x) * (1-cfLimit),
      # for x > 0. Set cdf_original(x) =  cfLimit + cdf_new(x) * (1-cfLimit).
      
        cfLimit = np.real(cf(1e300)) 
        cfOld = cf # skontrolovat spravnost priradenia 
        cf2 = cf # skontrolovat spravnost priradenia 
        if options['isCompound'] == True:
            if cfLimit > 1e-13:
                cf2 = lambda t : (cf(t) - cfLimit) / (1 - cfLimit)
            options.update({'isNonnegative': True})

        cft = cf2(tolDiff * np.arange(1,5))
        cftRe = np.real(cft)
        cftIm = np.imag(cft)
        
        if len(xMean) == 0:
            if options['isCircular'] == True:
                xMean.append(np.angle(cf2(1)))
            else:
                xMean = (8 * cftIm[0] / 5 - 2 * cftIm[1] / 5 + 8 * cftIm[2] / 105 - 2 * cftIm[3] / 280) / tolDiff 
        
        if len(xStd) == 0:
            if options['isCircular'] == True: 
                # see https://en.wikipedia.org/wiki/Directional_statistics 
                xStd = np.sqrt(-2 * np.log(abs(cf2(1))))
            else:
                xM2 = (205 / 72 - 16 * cftRe[0] / 5 + 2 * cftRe[1] / 5 - 16 * cftRe[2] / 315 + 2 * cftRe[3] / 560) / (tolDiff ** 2)
                xStd = np.sqrt(xM2 - xMean ** 2) 
        
        if np.isfinite(xMin) and np.isfinite(xMax):
            xRange = xMax - xMin 
        elif len(dt) > 0:
            xRange = 2 * np.pi / dt 
        elif len(T) > 0:
            xRange = 2 * np.pi / (T / N)
        else:
            if options['isCircular'] == True:
                xMin = -np.pi
                xMax = np.pi
            else:
                if np.isfinite(xMin):
                    xMax = xMean + SixSigmaRule * xStd
                elif np.isfinite(xMax):
                    xMin = xMean - SixSigmaRule * xStd
                else:
                    xMin = xMean - SixSigmaRule * xStd
                    xMax = xMean + SixSigmaRule * xStd
            xRange = xMax - xMin

        dt = 2 * np.pi / xRange
        t = np.arange(1,N+1) * dt
        cft = cf2(t)
        cft[-1] = cft[-1] / 2

        options['DIST']['xMin'] = xMin
        options['DIST']['xMax'] = xMax
        options['DIST']['xMean'] = xMean
        options['DIST']['cft'] = cft
        options['DIST']['cfLimit'] = cfLimit

    # ALGORITHM ---------------------------------------------------------------

    #print(x)
    if np.isscalar(x): 
        x = np.array([x])
        
    if len(x) == 0:
        x = np.linspace(xMin, xMax, options['xN'])
    
    if options['isInterp'] == True:
        xOrg = x
        #Chebysev points
        x = (xMax - xMin) * (-np.cos(np.pi * (np.arange(0,options['chebyPts']+1)) / options['chebyPts']) + 1) / 2 + xMin
    else:
        xOrg = []
    
    #WARNING: OUT of range
    if any(x[i] < xMin for i in range(0,len(x))) or any(x[i] > xMax for i in range(0,len(x))):
        warnings.warn('CharFunTool: cf2DistGP - x out of range (the used support): [xMin, xMax] = ['+str(xMin)+','+str(xMax)+']!')

    # Evaluate the required functions
    
    x = np.asarray(x)
    szx = x.shape
    x = x.flatten()
    t = np.asarray(t)
    E = np.exp(np.outer((-1j) * x.T, t))

    # CDF estimate computed by using the simple trapezoidal quadrature rule
    cft = np.asarray(cft)
    cdf = (xMean - x) / 2 + np.imag(np.inner(E, (cft / t).T))
    cdf = 0.5 - (cdf * dt) / np.pi

    # Correct the CDF (if the computed result is out of (0,1))
    # This is useful for circular distributions over intervals of length 2*pi,
    # as e.g. the von Mises distribution
    corrCDF = 0
    if options['correctedCDF'] == True:
        if np.min(cdf) < 0:
            corrCDF = np.min(cdf)
            cdf = cdf - corrCDF
        if np.max(cdf) > 1:
            corrCDF = np.max(cdf) - 1
            cdf = cdf - corrCDF
    
    cdf = np.reshape(cdf, szx)

    # PDF estimate computed by using the simple trapezoidal quadrature rule
    pdf = 0.5 + np.real(np.inner(E, cft.T))
    pdf = (pdf * dt) / np.pi
    pdf[pdf < 0] = 0 
    pdf = np.reshape(pdf, szx)
    x = np.reshape(x, szx)

    # REMARK:
    # Note that, exp(-1i*x_i*0) = cos(x_i*0) + 1i*sin(x_i*0) = 1. Moreover,
    # cf(0) = 1 and lim_{t -> 0} cf(t)/t = E(X) - x. Hence, the leading term of
    # the trapezoidal rule for computing the CDF integral is CDFfun_1 = (xMean- x)/2,
    # and PDFfun_1 = 1/2 for the PDF integral, respectively.

    # Reset the transformed CF, PDF, and CDF to the original values 
    if options['isCompound']:
        cf2 = cfOld 
        cdf = cfLimit + cdf * (1 - cfLimit) 
        pdf = pdf * (1 - cfLimit)
        pdf[x == 0] = np.inf
        pdf[x == xMax] = 'NaN'
    
    # Calculate the precision criterion PrecisionCrit = abs(cf(t)/t) <= tol,
    # PrecisionCrit should be small for t > T, smaller than tolerance
    # options$crit
    PrecisionCrit = np.abs(cft[-1] / t[-1])
    isPrecisionOK = (PrecisionCrit <= options['crit'])

    # QF evaluated by the Newton-Raphson iterative scheme
    if len(prob) > 0:
        isPlot = options['isPlot']
        options['isPlot'] = False
        isInterp = options['isInterp']
        options['isInterp'] <- False
        prob = np.asarray(prob)
        szp = prob.shape
        prob = prob.flatten()
        maxiter = options['maxiter']
        crit = options['crit']
        qf = options['qf0']
        criterion = True
        count = 0
        res = cf2DistGP(cf2, qf, options = options)
        cdfQ = res['cdf']
        pdfQ = res['pdf']
        
        while criterion == True:
            count = count + 1
            correction = (cdfQ - corrCDF - prob) / pdfQ
            qf = np.maximum(xMin, np.minimum(xMax, qf - correction)) 
            
            res = cf2DistGP(lambda x : cf2(x), x = qf, options = options)
            cdfQ = res['cdf']
            pdfQ = res['pdf']
            
            if np.isscalar(qf): 
                qf = np.array([qf])
            criterion = any(np.abs(correction)[i] > (crit * np.abs(qf))[i] for i in range(0,len(qf))) and np.max(np.abs(correction)) > crit and count < maxiter
        
        qf = np.reshape(qf, szp)
        prob = np.reshape(prob, szp)
        options['isPlot'] = isPlot
        options['isInterp'] = isInterp
    else:
        qf = []
        count = []
        correction = []
    
    if options['isInterp'] == True:
        idd = np.isfinite(pdf)
        PDF = lambda xNew : np.maximum(0, interpBarycentric(x[idd], pdf[idd], xNew)['funNew']) # skontrolovat
        idd = np.isfinite(cdf) 
        CDF = lambda xNew : np.maximum(0, np.minimum(1, interpBarycentric(x[idd], cdf[idd], xNew)['funNew'])) # skontrolovat 
        QF = lambda prob : interpBarycentric(cdf[idd], x[idd], prob)[[2]] # skontrolovat 
        RND = lambda n : QF(np.random.rand(n))

    if len(xOrg) > 0:
        x = xOrg
        cdf = CDF(x)
        pdf = PDF(x)

    # Reset the correct value for compound PDF at 0
    if options['isCompound'] == True: 
        pdf[x == 0] = np.inf

    ## Result
    result = {'Description': 'CDF/PDF/QF from the characteristic function CF', 'x': x, 'cdf': cdf, 'pdf': pdf, 'prob': prob, 
              'qf': qf, 'cf': cfOld, 'isCompound': options['isCompound'], 'isCircular': options['isCircular'], 'isInterp': options['isInterp'], 
              'SixSigmaRule': options['SixSigmaRule'], 'N': N, 'dt': dt, 'T': t[-1], 'PrecisionCrit': PrecisionCrit, 
              'myPrecisionCrit': options['crit'], 'isPrecisionOK': isPrecisionOK, 'xMean': xMean, 'xStd': xStd, 'xMin': xMin, 
              'xMax': xMax, 'cfLimit': cfLimit, 'cdfAdjust': correction, 'nNewtonRaphsonLoops': count, 'options': options}

    if options['isInterp']: 
        result['PDF'] = PDF
        result['CDF'] = CDF
        result['QF'] = QF
        result['RND'] = RND
      
    end_time = time.time()
    options.update({'tictoc': end_time - start_time})

    # PLOT the PDF / CDF
    
    if np.isscalar(x): 
        x = np.array([x])
    if len(x) == 0:
        options['isPlot'] = False
    
#     if options['isPlot'] == True:
#         plt.plot(x, pdf, color='blue', linewidth=2)
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('pdf', fontsize=12)
#         plt.title('PDF Specified by the CF', fontsize=20)
#         plt.grid()
#         plt.show()
        
#         plt.plot(x, cdf, color='blue', linewidth=2)
#         plt.xlabel('x', fontsize=12)
#         plt.ylabel('cdf', fontsize=12)
#         plt.title('CDF Specified by the CF', fontsize=20)
#         plt.grid()
#         plt.show()

    return(result)


# ## EXAMPLE1
# # Calculate CDF/PDF of N(0,1) by inverting its CF
# def cf(t):
#     return(np.exp(-np.power(t, 2) / 2))
    
# #result = cf2DistGP(cf)


## EXAMPLE2
# PDF/CDF of the compound Binomial-Exponential distribution
# n = 25
# p = 0.3
# lamb = [5]
# cfX = lambda t : cfX_Exponential(t=t, lamb=lamb)
# cf = lambda t : cfN_Binomial(t, n, p, cfX)
# x = np.linspace(0, 5, 101)
# prob = [0.9, 0.95, 0.99]
# options = {'isCompound':True}
# result = cf2DistGP(cf=cf, x=x, prob=prob, options=options)


## EXAMPLE3
# PDF/CDF of the compound Poisson-Exponential distribution
# lamb1 = [10]
# lamb2 = [5]
# cfX = lambda t : cfX_Exponential(t, lamb2)
# cf = lambda t : cfN_Poisson(t, lamb1, cfX)
# x = np.linspace(0, 8, 101)
# prob = [0.9, 0.95, 0.99]
# options = {'isCompound':True}
# result = cf2DistGP(cf, x, prob, options)


