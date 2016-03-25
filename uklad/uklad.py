#!/usr/bin/python2
import numpy as np
import scipy.misc
import sys

ITER_NUM = 250
EPS = 0.01
BAILOUT = 2

xmin, xmax = (-0.7, 0.3)
ymin, ymax = (-0.3j, 1.3j)
c = -0.123 + 0.745j
abs_c = abs(c)

# Todo: implement these variables as arguments
# Todo: Move to Julia set class?
# Exponent in the Julia set
p = 2
# Number of 'trailing' elements when calculating Triangle Inequality Average
m = 3

width = 30
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
    return iters + 1 + np.log(np.log(BAILOUT) / np.log(abs(z))) / np.log(p)


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
        path = sys.argv[1]

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
            if np.isnan(index) or np.isinf(index):
                index = 0

            bitmap[col][row] = index
    scipy.misc.imsave(path, bitmap)
