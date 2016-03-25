#!/usr/bin/python2
import numpy as np
from numpy import isnan, isinf
import scipy.misc
import sys
import os
from julia import t, avg_sum, lin_inp

ITERATION_FILE = 'big-mbrot.npy'
OUTPUT_FILE = 'img_edit/mandelbrot_-1/out.png'
ITERS_PER_POINT = 200
log2 = np.log(2)


# Interpolation functions: generate colour index
def iteration_index(iteration):
    return float(iteration) / ITERS_PER_POINT
iteration_index = np.vectorize(iteration_index)


def smooth_index(iteration, escaped):
    return iteration + 1 - np.log(log2 / np.log(np.abs(escaped))) / log2

def greyscale(index):
    colour = np.floor(255 * index)
    return colour, colour, colour

if __name__ == '__main__':
    if len(sys.argv) == 2:
        num_avg_elems = int(sys.argv[1])
    else:
        num_avg_elems = 10
    path = 'img_edit/mandelbrot_huge/m' + str(num_avg_elems) + '.png'

    file_data = np.load(ITERATION_FILE)
    file_data.dtype.names = ('cplx_consts', 'iters', 'iter_val_arrays')
    argand_points = file_data['cplx_consts']
    iterations = file_data['iters']
    zs_arrays = file_data['iter_val_arrays']

    half_height, width = file_data.shape
    # Scipy saves bitmaps in form (height, width, colour_channels)
    bitmap_height = half_height * 2 - 1
    bitmap = np.empty((bitmap_height, width))

    middle_row = half_height - 1
    for row in range(half_height):
        for col in range(width):
            point = argand_points[row][col]
            numiters = iterations[row][col]
            iterated_zs = zs_arrays[row][col]

            smooth_count = smooth_index(numiters, iterated_zs[numiters - 1])
            if col % 10 == 0: print smooth_count
            index = smooth_count / ITERS_PER_POINT

            # index = lin_inp(iterated_zs, smooth_count % 1.0,
            #                 numiters, num_avg_elems, point)

            if isnan(index) or isinf(index):
                index = 0

            bitmap[middle_row + row][col] = index
            bitmap[middle_row - row][col] = index

    dir_path = os.path.dirname(OUTPUT_FILE)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    scipy.misc.imsave(OUTPUT_FILE, bitmap)
