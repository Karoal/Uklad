#!/usr/bin/python2
import numpy as np
import scipy.misc
import sys

ITER_NUM = 250
EPS = 0.01
BAILOUT = 2

xmin, xmax = (-1., 1.)
ymin, ymax = (-1.j, 1.j)

width = 30
yrange = np.abs(ymax - ymin)
xrange = xmax - xmin

height = np.int(yrange * width / xrange)

path = 'out.png'


class Julia:
    # Exponent in the Julia set
    p = 2
    BAILOUT = 2.0

    def __init__(self, c, m=3):
        self.c = c
        self.m = m

    def julia(self, z):
        zs = np.empty(ITER_NUM, dtype='complex64')
        for i in range(ITER_NUM):
            z = z ** 2 + self.c
            zs[i] = z
            if abs(z) >= BAILOUT:
                return i, zs
        return ITER_NUM, zs

    # Shading functions
    def smooth_iter(self, z, iters):
        return iters + 1 + np.log(np.log(BAILOUT) /
                                  np.log(abs(z))) / np.log(self.p)

    def t(self, zn_minus1, zn, const):
        abs_zn_minus1 = abs(zn_minus1 ** self.p)

        mn = abs(abs_zn_minus1 - abs(const))
        Mn = abs_zn_minus1 + abs(const)
        return (abs(zn) - mn) / (Mn - mn)

    def avg_sum(self, zs, i, num_elems, const):
        if i - num_elems == 0:
            return np.inf
        return (
            sum(self.t(zs[n-2], zs[n-1], const) for n in range(num_elems, i)) /
            (i - num_elems))

    def lin_inp(self, zs, d, i):
        last_iters_num = i if i < self.m else self.m
        return (d * self.avg_sum(zs, i, last_iters_num, self.c) +
                (1 - d) * self.avg_sum(zs[:-1], i, last_iters_num, self.c))

if __name__ == '__main__':
    if len(sys.argv) == 2:
        path = sys.argv[1]

    xaxis = np.linspace(xmin, xmax, width)
    yaxis = np.linspace(ymin, ymax, height)

    bitmap = np.zeros((height, width))
    julia = Julia(-0.8 + 0.156j)

    for row in range(width):
        for col in range(height):
            cplx_param = xaxis[row] + yaxis[col]
            num_iters, iterated_zs = julia.julia(cplx_param)
            smooth_count = julia.smooth_iter(iterated_zs[num_iters - 1], num_iters)

            index = julia.lin_inp(iterated_zs, smooth_count % 1.0,
                                  num_iters)
            if np.isnan(index) or np.isinf(index):
                index = 0

            bitmap[col][row] = index
    scipy.misc.imsave(path, bitmap)
