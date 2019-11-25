# -*- coding: utf-8 -*-
"""
Python implementation of Frank van den Bosch's N-body simulation IC generation
code. Draws (x,y,z) and (vx,vy,vz) for an NFW halo of concentration c.
Units adopted such that G = M = r_s = 1.
Output is in binary format for use with Go Ogiya's gputree code.

Sheridan Beckwith Green
sheridan.green@yale.edu
Nov 2019
"""

from numba import njit, jit, prange
from root_finding import brentq
from scalar_maximization import brent_max
from numpy.random import rand, randint, seed
from numpy import log, log10, abs, exp, inf
from numpy import zeros, pi, sqrt, sin, cos, savetxt
from numpy import repeat, any, sum, column_stack
from scipy.optimize import ridder, brent

@njit(fastmath=True)
def NFWf(c):
    return log(1. + c) - c/(1. + c)

@njit(fastmath=True)
def NFWpotential(r):
    return log(1. + r) / r

@njit(fastmath=True)
def fr_fx(xlgr, enc_m):
    return NFWf(10.**xlgr) - enc_m

@njit(fastmath=True, parallel=True)
def find_r(enc_m, rtrunc):
    Npart = len(enc_m)
    lg_rtrunc = log10(rtrunc)
    lgrads = zeros(Npart)
    for i in prange(0, Npart):
        lgrads[i] = brentq(fr_fx, -5., lg_rtrunc, xtol=1e-4, args=(enc_m[i],)).root

    return 10**lgrads

@njit(fastmath=True)
def dist_func(e):
    # Fitting function for an NFW
    # distribution function using Table 2
    # and eqn A2 from Widrow (2000)
    F0 = 9.1968E-2
    q = -2.7419
    p1 = 0.3620
    p2 = -0.5639
    p3 = -0.0859
    p4 = -0.4912

    e2 = 1. - e
    pp = p1 * e + p2 * e**2 + p3 * e**3 + p4 * e**4
    fac1 = e**(1.5) / e2**(2.5)
    fac2 = (-1.*log(e)/e2)**q
    fac3 = exp(pp)
    return F0 * fac1 * fac2 * fac3

@njit(fastmath=True)
def fevsq(v, p):
    fe = dist_func(p - v**2 / 2.)
    if(fe == 0 or v == 0):
        return -inf
    else:
        return -1. / (fe * v**2)

@njit(fastmath=True, parallel=True)
def find_pmax(vesc, psi):
    Npart = len(vesc)
    pmax = zeros(Npart)
    vmax = zeros(Npart)
    # find maximum of distribution function * v**2
    # for each particle's r
    for i in prange(0, Npart):
        vmax[i], pmax[i], num = brent_max(fevsq, 1E-8, .99999*vesc[i], args=(psi[i],), xtol=1.0E-4)
    return -1.001 / pmax, vmax


def nbodipy(Npart, conc, rtrunc, oname='output.dat', sd=randint(2**32)):
    fc = NFWf(conc)
    Mc = NFWf(rtrunc) / fc
    mpart = Mc / float(Npart)
    mpart = repeat([mpart], Npart)

    # generate Npart random numbers
    # translated to enclosed masses
    # all within the cutoff radius
    seed(sd)
    prob = rand(Npart) * fc * Mc
    r = find_r(prob, rtrunc)

    # generate phi and cos(theta)
    # for random positions at each radius
    phi = rand(Npart) * 2. * pi
    costheta = 2.*rand(Npart) - 1.
    sintheta = sqrt(1. - costheta**2.)

    x = r * sintheta * cos(phi)
    y = r * sintheta * sin(phi)
    z = r * costheta

    psi = NFWpotential(r)
    vesc = sqrt(2. * abs(psi))
    pmax, vmax = find_pmax(vesc, psi)

    # run rejection algorithm
    # Explained in S4.1 of Kuijken and Dubinski ( 1994)
    msk = repeat([True], Npart)
    vvals = zeros(Npart)
    total = 0
    while(any(msk)): # while there are still rejections
        this_try = sum(msk)
        total += this_try
        # sample uniformly in 0 to vesc
        vvals[msk] = rand(this_try) * vesc[msk]
        ff = dist_func(psi[msk] - vvals[msk]**2 / 2.) * vvals[msk]**2
        # keep the randomly sampled point if a second uniformly
        # sampled point in range of 0 to max(f(E)*v**2) is
        # below the value of f(E)*v**2 for our sampled v
        msk[msk] = rand(this_try) > ff/pmax[msk]

    energy = psi - vvals**2 / 2.

    # generate random orientations for velocities
    phi = rand(Npart) * 2. * pi
    costheta = 2. * rand(Npart) - 1.
    sintheta = sqrt(1. - costheta**2)

    vx = vvals * sintheta * cos(phi)
    vy = vvals * sintheta * sin(phi)
    vz = vvals * costheta

    # we're done!
    # convert units, etc. and output
    print("Generated %d particles with %d total tries." % (Npart, total))

    vx = vx / sqrt(fc)
    vy = vy / sqrt(fc)
    vz = vz / sqrt(fc)
    vmax = vmax / sqrt(fc)

    out = column_stack((mpart, x, y, z, vx, vy, vz))
    savetxt(oname, out, header='%d' % Npart, comments='')
    print("Output ICs to %s" % oname)


if __name__ == "__main__":
    import sys
    Npart = int(sys.argv[1])
    conc = float(sys.argv[2])
    rtrunc = float(sys.argv[3])
    if(len(sys.argv) == 5): # there is a name
        nbodipy(Npart, conc, rtrunc, sys.argv[4])
    else:
        nbodipy(Npart, conc, rtrunc)

