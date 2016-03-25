#!/usr/bin/python2
import numpy as np
from numpy import abs

iterations_per_point = 200
EPSILON = 0.01
power = 2

DATA_FILE = 'big-mbrot.npy'
# ESCAPED_FILE = 'escaped_mi.npy'

x_min, x_max = (-2.1, 0.7)
y_min, y_max = (-1.2j, 1.2j)
gen_y_min, gen_y_max = (0, y_max)

# Width of resulting image in pixels
width = 2500

# Preserving proportions in pixel-height
y_range = np.abs(y_max - y_min)
x_range = x_max - x_min
height = np.int(y_range * width / x_range)
# Uses vertical symmetry in Mandelbrot set to save memory
half_height = int(height / 2) + 1


def iterations_to_escape(c):
    """Computes the Mandelbrot sequence for complex c"""
    zs = np.empty(iterations_per_point, dtype='complex64')
    z = 0
    if c == 0 and power < 0:
        zs[0] = c
        return 0, zs
    for i in range(iterations_per_point):
        z = np.power(z, power) + c
        zs[i] = z
        if abs(z) >= 2: break
    return c, i, z

# Allows the function to be performed element-wise
iterations_to_escape = np.vectorize(iterations_to_escape)

if __name__ == '__main__':
    x_axis = np.linspace(x_min, x_max, width)
    y_axis = np.linspace(gen_y_min, gen_y_max, half_height)
    xs, ys = np.meshgrid(x_axis, y_axis)

    data_type = 'complex64, int, {}complex64'.format(iterations_per_point)
    output_data = np.empty((half_height, width), dtype=data_type)

    for row in range(half_height):
        for col in range(width):
            output_data[row][col] = iterations_to_escape(x_axis[col] + y_axis[row])
        if row % 20 == 0:
            print row

    # Data saved in triples (complex_param, num_of_iters, iterated_values)
    np.save(DATA_FILE, output_data)
