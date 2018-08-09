import numpy as np 
import lmfit

_cached_params = None
def minimize(x, y, options={'order':3, 'reusable':False, 'verbose':False}):
    global _cached_params
    
    if options is not None:
        order, reusable, verbose = map(options.get, ('order', 'reusable', 'verbose'))
        if order not in [1, 2, 3]:
            order = 3
            print "A default value of order is being used (order=3)"
    else:
        order, reusable, verbose = (3, False, False)

    if reusable and _cached_params:
        params = _cached_params
        perturb_params(params) 
        params['offset'].set(min=0.1*params['offset'], max=1.1*params['offset']) 
    else:
        params = initialize_params(x, y, order)
        perturb_params(params) 

    result = lmfit.minimize(residual, params, args=(x, y), method='powell', options={'xtol':1e-5,'ftol':1e-5})
    
    if verbose:
        lmfit.printfuncs.report_fit(result.params)
    if reusable:
        _cached_params = result.params

    return result # the optimization result

def initialize_params(x, y, order):
   # solve Az = b:
    A = np.array((np.ones(x.size/2), x[0:x.size/2]))
    b = np.log(np.abs(y[0:x.size/2]-np.mean(y[-2:]))) 
    z = np.linalg.lstsq(A.T, b)
    ampl = np.exp(z[0][0])
    rate = -z[0][1]
    offset = np.mean(y[-2:])

    params = lmfit.Parameters()
    if order == 3:
        if y[0] > y[-1]:
            params.add('a1', value=0.3333*ampl, min=0.)#, max=1.*ampl)
            params.add('a2', value=0.3333*ampl, min=0.)#, max=1.*ampl) 
            params.add('a3', value=0.3333*ampl, min=0.)#, max=1.*ampl)
        else:
            params.add('a1', value=-0.3333*ampl, max=0.)#, max=1.*ampl)
            params.add('a2', value=-0.3333*ampl, max=0.)#, max=1.*ampl) 
            params.add('a3', value=-0.3333*ampl, max=0.)#, max=1.*ampl)
        params.add('r1', value=1.0*rate, min=0., max=1.*rate)
        params.add('r2', value=1.0*rate, min=0., max=10.*rate)
        params.add('r3', value=1.0*rate, min=0., max=100.*rate)
    if order == 2:
        if y[0] > y[-1]:
            params.add('a1', value=0.5*ampl, min=0.)#, max=10.*amplitude)
            params.add('a2', value=0.5*ampl, min=0.)#, max=10.*amplitude) 
        else:
            params.add('a1', value=-0.5*ampl, max=0.)#, max=10.*amplitude)
            params.add('a2', value=-0.5*ampl, max=0.)#, max=10.*amplitude) 
        params.add('r1', value=1.0*rate, min=0., max=1.*rate)
        params.add('r2', value=1.0*rate, min=0., max=10.*rate) 
        params.add('a3', value=0., vary=False)
        params.add('r3', value=0., vary=False)
    if order == 1:
        if y[0] > y[-1]:
            params.add('a1', value=1.0*ampl, min=0.)#, max=10.*amplitude)
        else:
            params.add('a1', value=-1.0*ampl, max=0.)#, max=10.*amplitude)      
        params.add('r1', value=1.0*rate, min=0.)
        params.add('a2', value=0., vary=False)
        params.add('a3', value=0., vary=False)
        params.add('r2', value=0., vary=False)
        params.add('r3', value=0., vary=False)
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
    model = p['offset']  \
          + p['a1']*np.exp(-p['r1']*x) \
          + p['a2']*np.exp(-p['r2']*x) \
          + p['a3']*np.exp(-p['r3']*x)
    return model - y

