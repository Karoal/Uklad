#!/usr/bin/python2
import numpy as np
from numpy import log, isnan, isinf
import scipy.misc
import sys
# for performance analysis
# import cProfile

ITER_NUM = 250
EPS = 0.01
BAILOUT = 2

xmin, xmax = (-0.7, 0.3)
ymin, ymax = (-0.3j, 1.3j)
c = 0.345 - 0.45j
abs_c = abs(c)
# c = -0.5 + 0.25j
# c = -0.123 + 0.745j
# Exponent in the Julia set
p = 2
# Number of 'trailing' elements when calculating Triangle Inequality Average
m = 10

width = 2500
yrange = np.abs(ymax - ymin)
xrange = xmax - xmin
height = np.int(yrange * width / xrange)


def julia(z):
    zs = np.empty(ITER_NUM, dtype='complex64')
    for i in range(ITER_NUM):
        z = z ** 2 + c
        zs[i] = z
        if abs(z) >= BAILOUT:
            return i, zs
    return ITER_NUM, zs


# Functions for "Smooth Iteration Count"
def smooth_iter(z, iters):
    # return iters + 1 - log(log(abs(z))) / log(m)
    return iters + 1 + log(log(BAILOUT) / log(abs(z))) / log(p)


# Functions for Triangle Inequality Average method for colouring fractals
# Pre: zs has at least two elements
def t(zn_minus1, zn, const):
    abs_zn_minus1 = abs(zn_minus1 ** p)

    mn = abs(abs_zn_minus1 - abs(const))
    Mn = abs_zn_minus1 + abs(const)
    return (abs(zn) - mn) / (Mn - mn)


# to be implemented later
def avg_sum(zs, i, numelems, const):
    if i - numelems == 0:
        return np.inf
    return (sum(t(zs[n - 2], zs[n - 1], const) for n in range(numelems, i)) /
            (i - numelems))


def lin_inp(zs, d, i, num_elems, const=c):
    last_iters_num = i if i < num_elems else num_elems
    return (d * avg_sum(zs, i, last_iters_num, const) +
            (1 - d) * avg_sum(zs[:-1], i, last_iters_num, const))


if __name__ == '__main__':
    if len(sys.argv) == 2:
        m = int(sys.argv[1])
    path = 'img_edit/julia-header/m' + str(m) + '.png'

    xaxis = np.linspace(xmin, xmax, width)
    yaxis = np.linspace(ymin, ymax, height)

    bitmap = np.zeros((height, width))

    for row in range(width):
        for col in range(height):
            cplx_param = xaxis[row] + yaxis[col]
            numiters, iterated_zs = julia(cplx_param)
            smooth_count = smooth_iter(iterated_zs[numiters - 1], numiters)

            index = lin_inp(iterated_zs, smooth_count % 1.0,
                            numiters, m, c)
            if isnan(index) or isinf(index):
                index = 0

            bitmap[col][row] = index
        if row % 10 == 0:
            print str(m) + ': ' + str(row)
    scipy.misc.imsave(path, bitmap)

# if __name__ == '__main__':
    # cProfile.run('main()')
