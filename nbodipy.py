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

from numba import njit
from numpy.random import rand, randint, seed
from numpy import log, log10, abs, exp, inf
from numpy import zeros, pi, sqrt, sin, cos
from numpy import repeat, any, sum
from scipy.optimize import ridder, brent

@njit
def NFWf(c):
    return log(1. + c) - c/(1. + c)

@njit
def NFWpotential(r):
    return log(1. + r) / r

#@njit
def find_r(enc_m, rtrunc):
    Npart = len(enc_m)
    lg_rtrunc = log10(rtrunc)
    lgrads = zeros(Npart)
    for i in range(0, Npart):
        fx = lambda xlgr: NFWf(10.**xlgr) - enc_m[i]
        lgrads[i] = ridder(fx, -5., lg_rtrunc, xtol=1e-4)
    return 10**lgrads

@njit
def dist_func(e):
    F0 = 9.1968E-2
    q = -2.7419
    p1 = 0.3620
    p2 = -0.5639
    p3 = -0.0859
    p4 = -0.4912

    e2 = 1-e
    pp = p1 * e + p2 * e**2 + p3 * e**3 + p4 * e**4
    fac1 = e**(1.5) / e2**(2.5)
    fac2 = (-1.*log(e)/e2)**q
    fac3 = exp(pp)
    return F0 * fac1 * fac2 * fac3

@njit
def fevsq(v, p):
    fe = dist_func(p - v**2 / 2.)
    if(fe == 0 or v == 0):
        return inf
    else:
        return 1. / (fe * v**2)

def find_pmax(vesc, psi):
    Npart = len(vesc)
    pmax = zeros(Npart)
    for i in range(0, Npart):
        _, pmax[i], _, fc = brent(fevsq, args=(psi[i],), brack=(1E-8, vesc[i]), tol=1.0E-4, full_output=True)
    return 1.001 / pmax


def nbodipy(Npart, conc, rtrunc, oname='output.dat', sd=randint(2**32)):
    fc = NFWf(conc)
    Mc = NFWf(rtrunc) / fc
    mpart = Mc / float(Npart)

    # generate Npart random numbers
    # translated to enclosed masses
    # all within the cutoff radius
    seed(sd)
    prob = rand(Npart) * fc * Mc
    r = find_r(prob, rtrunc)

    # generate phi and cos(theta)
    phi = rand(Npart) * 2. * pi
    costheta = 2.*rand(Npart) - 1.
    sintheta = sqrt(1. - costheta**2.)

    x = r * sintheta * cos(phi)
    y = r * sintheta * sin(phi)
    z = r * costheta

    psi = NFWpotential(r)
    vesc = sqrt(2. * abs(psi))
    pmax = find_pmax(vesc, psi)

    msk = repeat([True], Npart)
    vvals = zeros(Npart)
    total = 0
    while(any(msk)): # while some have failed
        this_try = sum(msk)
        total += this_try
        vvals[msk] = rand(this_try) * vesc[msk]
        ff = dist_func(psi[msk] - vvals[msk]**2 / 2.) * vvals[msk]**2
        msk[msk] = rand(this_try) > ff/pmax[msk]

    energy = psi - vvals**2 / 2.

    # random orientations in v-sapce
    phi = rand(Npart) * 2. * pi
    costheta = 2. * rand(Npart) - 1.
    sintheta = sqrt(1. - costheta**2)

    vx = vvals * sintheta * cos(phi)
    vy = vvals * sintheta * sin(phi)
    vz = vvals * costheta

    # we're done!
    # convert units, etc.
    print("Generated %d particles with %d total tries." % (Npart, total))




if __name__ == "__main__":
    import sys
    Npart = int(sys.argv[1])
    conc = float(sys.argv[2])
    rtrunc = float(sys.argv[3])
    if(len(sys.argv) == 5): # there is a name
        nbodipy(Npart, conc, rtrunc, sys.argv[4])
    else:
        nbodipy(Npart, conc, rtrunc)

