#!/usr/bin/python2
import numpy as np
import scipy.misc
import sys

EPS = 0.01

xmin, xmax = (-1., 1.)
ymin, ymax = (-1.j, 1.j)

width = 30
yrange = np.abs(ymax - ymin)
xrange = xmax - xmin

height = np.int(yrange * width / xrange)

path = 'out.png'


class Julia:
    # Exponent in the Julia set
    pow = 2
    BAILOUT = 2.0
    ITER_NUM = 250

    def __init__(self, c, m=3):
        self.c = c
        self.m = m

    def julia(self, z):
        zs = np.empty(self.ITER_NUM, dtype='complex64')
        for i in range(self.ITER_NUM):
            z = z ** self.pow + self.c
            zs[i] = z
            if abs(z) >= self.BAILOUT:
                return i, zs
        return self.ITER_NUM, zs

    # Shading functions
    def smooth_iter(self, z, iters):
        return iters + 1 + np.log(np.log(self.BAILOUT) /
                                  np.log(abs(z))) / np.log(self.pow)

    def _addend(self, zn_minus1, zn):
        abs_zn_minus1 = abs(zn_minus1 ** self.pow)

        mn = abs(abs_zn_minus1 - abs(self.c))
        Mn = abs_zn_minus1 + abs(self.c)
        return (abs(zn) - mn) / (Mn - mn)

    def _avg_sum(self, zs, i, num_elems):
        if i == num_elems:
            # Case when the point is part of the Julia set
            return np.inf

        f = lambda n: self._addend(zs[n - 2], zs[n - 1])
        return sum(f(n) for n in range(num_elems, i)) / (i - num_elems)

    def interpolation(self, zs, decimal_part, iters_no):
        """Performs linear interpolation. Usually applied on each point of a 2d array"""
        last_iters_num = iters_no if iters_no < self.m else self.m

        return (decimal_part * self._avg_sum(zs, iters_no, last_iters_num) +
                (1 - decimal_part) * self._avg_sum(zs[:-1], iters_no, last_iters_num))

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
            smooth_count = julia.smooth_iter(iterated_zs[num_iters - 1],
                                             num_iters)

            index = julia.interpolation(iterated_zs, smooth_count % 1.0,
                                        num_iters)
            if np.isnan(index) or np.isinf(index):
                index = 0

            bitmap[col][row] = index
    scipy.misc.imsave(path, bitmap)
