import functools
import math
from matplotlib import pyplot
from numpy import arange, cumprod, cumsum, argmax, argmin

from incentives import replicator, linear_fitness_landscape, logit, fermi
from heatmap import heatmap_for_function

import matplotlib
font = {'weight' : 'bold', 'size': 20}
matplotlib.rc('font', **font)

## Plotting helpers ##

def function_values_to_plot(f, a, b, steps=100):
    delta = (b - a) / float(steps)
    xs, ys = [], []
    for i in range(steps):
        x = a + delta * i
        y = f(x)
        xs.append(x)
        ys.append(y)
    return xs, ys

def points_for_incentive(func, lower=0.001, upper=10., step=0.01):
    points = []
    for r in arange(lower, upper, step):
        phi = func(r)
        points.append((r, phi))
    return points

## Comparison test functions ##

def quad_fixation(r, N=2):
    """For comparison, this is the closed formula."""
    return float(1. + 1./r)**(1.-N)

def moran_fixation(r, N=2):
    """For comparison, this is the closed formula."""
    if r == 1:
        return 1./N
    return float(1. - 1./r) / float(1. - 1./r**N)

## Fixation computations ##

def fixation_for_incentive(incentive, N):
    """Compute the fixation probability numerically."""
    ratios = []
    for i in range(1, N):
        i1,i2 = incentive((i, N-i))
        ratios.append(i*i2 / (i1 * (N-i)))
    t = cumprod(ratios)
    s = cumsum(t)
    return 1./(1+s[-1])

def fixation_func_for_incentive(N, incentive_func, kwargs):
    """Returns a function that gives a function for the fixation probability for the given incentive and parameters."""
    def func(r):
        m = [[r,r],[1,1]]
        landscape = linear_fitness_landscape(m)
        incentive = incentive_func(landscape, **kwargs)
        phi = fixation_for_incentive(incentive, N)
        return phi
    return func

def parameter_comparison(N=10, R=6, incentive_func=replicator, params=('q', [-1., 0., 0.5, 1., 2.])):
    """Plot fixation probabilities for given list of parameters."""
    for p in params[-1]:
        func = fixation_func_for_incentive(N, incentive_func, {params[0]:p})
        points = points_for_incentive(func, upper=R)
        pyplot.plot([x for (x,y) in points], [y for (x,y) in points], linewidth=2.)

def q_logit_comparison(N=10, R=6, incentive_func=logit, eta=1., params=('q', [0.1, 0.5, 1.])):
    """Plot fixation probabilities for given list of parameters."""
    for p in params[-1]:
        func = fixation_func_for_incentive(N, incentive_func, {params[0]:p, 'eta':eta})
        points = points_for_incentive(func, upper=R)
        pyplot.plot([x for (x,y) in points], [y for (x,y) in points])

def neutral_fixation(Nmin=2, Nmax=20, incentive_func=replicator, qs=[0,1,2]):
    for q in qs:
        points = []
        for N in range(Nmin, Nmax+1):
            phi = fixation_func_for_incentive(N, incentive_func, {'q':q})
            points.append(phi(1))
        pyplot.plot(range(Nmin, Nmax+1), points, linewidth=2)
    pyplot.xlabel('Population Size')
    pyplot.ylabel('Fixation Probability')
    pyplot.ylim((-0.01, 0.51))

def q_iss(r, q):
    if q == 1:
        if r > 1:
            return 0
        return 1
    if r == 0:
        if q < 1:
            return 0
        return 1
    return 1./(1. + r**(1./(q-1)))

def q_fermi_iss(r, q, beta=1.):
    if q == 1:
        if r > 1:
            return 0
        return 1
    if r == 0:
        if q < 1:
            return 0
        return 1
    r = math.e**(beta*(r-1))
    return 1./(1. + r**(1./(q-1)))

def q_logit_iss(r, beta=1., q=1.):
    if q == 1:
        if r > 1:
            return 0
        return 1
    if r == 0:
        if q < 1:
            return 0
        return 1
    r = math.e**(beta*(r-1))
    return 1./(1. + r**(1./(q-1)))

def q_fixation(N, q, incentive_func=replicator, r=1.):
    phi = fixation_func_for_incentive(N, incentive_func, {'q':q})
    return phi(r)

def real_quadratic_roots(a, b, c):
    if a == 0:
        return [-c / float(b)]
    a, b, c = (1., b/float(a), c/float(a))
    disc = b*b - 4*a*c
    if disc < 0:
        return []
    elif disc == 0:
        return [-b / (2. * a)]
    else:
        d = math.sqrt(disc)
        print b, disc, a
        return sorted([x for x in [(-b + d) / (2. * a), (-b - d) / (2. * a)] if x >= 0 and x <= 1])

def iss_comparison(N, a, b, c, d):
    # q == 1
    xs = range(2, N+1)
    ys = []
    for M in xs:
        if a -c + d - b == 0:
            ys.append(0)
        else:
            M = float(M)
            ys.append(((d - b) + (a - d) / M) / (a -c + d - b))
    pyplot.scatter(xs, ys, c='b')
    # q == 0
    p1 = []
    p0 = []
    for M in xs:
        M = float(M)
        A = a + c - d - b
        B = d + 2 * b -a - (a + d)/M
        C = a/M - b
        roots = real_quadratic_roots(A, B, C)
        print M, A, B, C, roots
        try:
            p0.append((M, roots[0]))
        except IndexError:
            pass
        try:
            p1.append((M, roots[1]))
        except IndexError:
            pass
    for z in [p0,p1]:
        pyplot.scatter([x for (x,y) in z], [y for (x,y) in z], c='r', marker='+')    
    #q == 2
    p1 = []
    p0 = []
    for M in xs:
        M = float(M)
        # q = 2
        A = a + c - d - b
        B = b + 2*d - c - (a + d)/M
        C = d / M - d

        print M, A, B, C
        roots = real_quadratic_roots(A, B, C)
        print M, A, B, C, roots
        try:
            p0.append((M, roots[0]))
        except IndexError:
            pass
        try:
            p1.append((M, roots[1]))
        except IndexError:
            pass
    for z in [p0,p1]:
        pyplot.scatter([x for (x,y) in z], [y for (x,y) in z], c='g', marker='d')
    pyplot.xlim(0,N+1)
    pyplot.ylim(0,1.1)
    pyplot.xlabel('Population size N')
    pyplot.ylabel('ISS location i/N')        

def paper_figures():
    ## Figure 2 (left)
    #Nmin=2
    #Nmax=80
    #neutral_fixation(Nmin=Nmin, Nmax=Nmax, qs=[0, 0.5, 0.8, 0.9, 1., 2.])
    #pyplot.show()

    # Figure 2 (right)
    pyplot.figure()
    qs = [round(x, 3) for x in arange(-1., 3.0, 0.01)]
    Ns = arange(2, 51, 1)
    #func = functools.partial(q_fixation, r=3.)
    data, plot_obj = heatmap_for_function(q_fixation, Ns, qs, cmap_name="jet")
    print data[-1]
    pyplot.xlabel("Population Size")
    pyplot.ylabel("Parameter q")
    pyplot.show()

    ## Figure 3
    #N=10
    #R=5
    #pyplot.figure()
    #parameter_comparison(N=N, R=R, incentive_func=logit, params=('beta', [0.1, 0.5, 1., 2., 10.]))
    #pyplot.ylabel("Fixation Probability")
    #pyplot.xlabel("Relative fitness r")
    ##pyplot.figure()
    ##parameter_comparison(N=N, R=R, incentive_func=fermi, params=('beta',[0.1, 0.5, 1., 2., 10.]))
    #pyplot.show()
    
    ## Figure 4
    #iss_comparison(20, 20., 1., 7., 10. )
    ##iss_comparison(200, 1,2,3,1 )
    #pyplot.show()

    ## Figure 5 (left)
    ### q_iss heatmap
    #qs = arange(0., 3.0, 0.01)
    #rs = arange(0., 3.0, 0.01)
    #data, plot_obj = heatmap_for_function(q_iss, rs, qs, cmap_name="jet")
    #pyplot.ylabel('Parameter q for q-replicator')
    #xticks = arange(0., 3.0, 0.2)
    #pyplot.xticks(rotation=45)
    #pyplot.xlabel('Relative fitness r')    
    #pyplot.show()

    ## Figure 5 (right)
    ### q_iss heatmap
    #qs = arange(0., 3.0, 0.01)
    #rs = arange(0., 3.0, 0.01)
    #data, plot_obj = heatmap_for_function(q_fermi_iss, rs, qs, cmap_name="jet")
    #pyplot.ylabel('Parameter q for q-replicator')
    #xticks = arange(0., 3.0, 0.2)
    #pyplot.xticks(rotation=45)
    #pyplot.xlabel('Relative fitness r')    
    #pyplot.show()

if __name__ == '__main__':
    paper_figures()
    exit()

    