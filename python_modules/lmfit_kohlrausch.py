import numpy as np 
import lmfit

_cached_params = None
def minimize(x, y, options={'reusable':False, 'verbose':False}):
    global _cached_params
    
    if options is not None:
        reusable, verbose = map(options.get, ('reusable', 'verbose'))
    else:
        reusable, verbose = (False, False)

    if reusable and _cached_params:
        params = _cached_params
        perturb_params(params) 
        params['offset'].set(min=0.1*params['offset'], max=1.1*params['offset']) 

    else:
        params = initialize_params(x, y)
        perturb_params(params) 

    result = lmfit.minimize(residual, params, args=(x, y), method='powell', options={'xtol':1e-5,'ftol':1e-5})
    
    lmfit.printfuncs.report_fit(result.params)
    if reusable:
        _cached_params = result.params

    return result # the optimization result

def initialize_params(x, y):#, mode):
   # solve Az = b:
    A = np.array((np.ones(x.size/2), x[0:x.size/2]))
    b = np.log(np.abs(y[0:x.size/2]-np.mean(y[-2:]))) 
    z = np.linalg.lstsq(A.T, b)
    ampl = np.exp(z[0][0])
    rate = -z[0][1]
    offset = np.min(y)
    beta = 1.0

    params = lmfit.Parameters()
    params.add('a', value=1.0*ampl, min=0.)
    params.add('r', value=1.0*rate, min=0.)
    params.add('beta', value=beta, min=0., max=1.0)
    params.add('offset', value=offset, min=0.)

    return params

def perturb_params(params):
    eps = 1e-2 
    for key in params:
        if params[key].vary == False: 
            continue
        new_value = params[key].value*np.random.uniform(low=1-eps, high=1+eps)
        params[key].set(value=new_value, min=params[key].min, max=params[key].max) 
    return None

def residual(params, x, y):
    p = params.valuesdict()
    if y[0] > y[-1]:
        model = p['a']*np.exp(-(p['r']*x)**p['beta']) + p['offset']
    else:
        model = p['a'] - p['a']*np.exp(-(p['r']*x)**p['beta']) + p['offset']
    return model - y

