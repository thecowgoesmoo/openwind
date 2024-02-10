#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2023, INRIA
#
# This file is part of Openwind.
#
# Openwind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Openwind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Openwind.  If not, see <https://www.gnu.org/licenses/>.
#
# For more informations about authors, see the CONTRIBUTORS file

"""
This example shows how to use your own instrument geometries in OpenWind.
In OpenWind, an instrument is described by its bore, i.e., the radius of the
main pipe along its length.
"""
import os

import matplotlib.pyplot as plt

from openwind import InstrumentGeometry

path = os.path.dirname(os.path.realpath(__file__))

# %% Conical parts

# For instruments consisting only of conical parts, the information can consist
# in a list of coordinates (x,r), with x the abscissa along the linearized bore
# length (measured from the mouth-end of the instrument), and r the radius at
# this abscissa, in meters.

# Then, use the :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
# class to create the OpenWind model of your instrument and visualize the instrument.
#
# a.From python list
# ^^^^^^^^^^^^^^^^^^^^^^

# For example, you can write the geometry directly in the OpenWind code :
        #   x[meters]   r[meters]
my_bore = [[0.0      ,  8.4e-03   ],
           [1.2e-03  ,  2.4e-03   ],
           [1.8e-03  ,  1.83e-03  ],
           [8.4e-03  ,  3.0e-03   ],
           # to specify a discontinuity of section you repeat the same x with two values of r
           [8.4e-03  ,  4.2e-03   ],
           [30.3e-03 ,  5.8e-03   ],
           [102.3e-03,  7.4e-03   ],
           [129.8e-03,  14.0e-03  ],
           [141.9e-03,  61.0e-03  ]]

instrument_1 = InstrumentGeometry(my_bore)

print('INSTRUMENT 1:')
print(instrument_1) # display geometry information

fig1 = plt.figure(1)  # create new figure
instrument_1.plot_InstrumentGeometry(figure=fig1)  # plot the instrument
plt.title('a simple instrument')  # add a title to the plot

# b.From file
# ^^^^^^^^^^^^^^^^^^^

# OpenWind can therefore also import text files to make a model of the instrument.
# First column is the linearized absissa, second column is the radius.
#
# .. warning::
#     The files parsed by this methods must be in .csv or .txt format,
#     with columns separated by spaces and/or tabs. Comments start with #.
#     Empty lines are ignored.
#
# .. hint::
#    To open the files with a spreadsheet editor (like excel) choose the
#    separators "Tab" and "Space" and turn on the option "merged delimiters"
#
# It is particularly interesting for large and/or detailed instruments, for
# which this list can grow very fast, such as the instrument given in the
# example file 'Ex1_instrument_2.txt' with the content:
#
# .. code-block:: shell
#
#    ##  x[meters]          r[meters]
#       0.0000000e-01   8.4000000e-03
#       2.0000000e-04   8.4000000e-03
#       6.0000000e-04   6.6262909e-03
#       1.0000000e-03   3.1069360e-03
#       1.4000000e-03   1.9895242e-03
#       1.7734052e-03   1.8250000e-03
#           ...             ...

file = os.path.join(path, 'Ex1_instrument_2.txt')
instrument_2 = InstrumentGeometry(file)

fig2 = plt.figure(2)  # create new figure
instrument_2.plot_InstrumentGeometry(figure=fig2)  # plot the instrument
plt.title('instrument from file')  # add a title to the plot

# %% Complexe shape

# Some shapes are not easily described by conical parts. OpenWind supports
# different types of shapes, that can easilly be mixed together in the same
# instrument.
# Each of these shape need to be defined in a precise way ('Formatting') indicating
# in the following order `[x_0, x_1, r_0, r_1, type, param]`
# * `x_0` is the start position of the shape (in meter)
# * `x_1` is the end position of the shape (in meter)
# * `r_0` is the start radius of the shape (in meter)
# * `r_1` is the end radius of the shape (in meter)
# * `type` the type of shape
# * `param` additional optional parameters necessary for some type of shape
#
# The different types are :
# - 'cone' : conical portion (same as above), no additional parameter.
#   Draws a straight line between [x0, r0] and [x1, r1]
# - 'circle' : an arc of a circle, additional parameters `R` the radius of curvature.
#   Draws an arc between [x0, r0] and [x1, r1] with radius R
# - 'exponential' : no additional parameter. Draws an exponential line between [x0, r0] and [x1, r1]
# - 'Bessel' : Additional parameter `alpha`.
#   Draws a line based on a "Bessel horn" function, where alpha is the expansion rate of the horn (=power)
# - 'spline' : Smooth C2 function with control points. Additional parameters
#   `[x2, x3,..., xN, r2, r3,..., rN]`. Draws a smooth line between [x0, r0] and [x1, r1], passing
#   through the control points (x2 r2) ... (xN, rN)
#
# You can either write the geometry as a list of list :

my_complex_bore = [[0.0,	0.0009,	0.0087,	0.0046,	'circle',	-0.01],
                   [0.0009,	0.0014,	0.0046,	0.0024,	'circle',	0.007],
                   [0.0014,	0.01,	0.0024,	0.003,	'cone'],
                   # the discontinuity of section is here done by fixing r_0 different to the previous r_1
                   [0.01,	0.03,	0.0042,	0.005,	'cone'],
                   [0.03,	0.1,    0.005,	0.005,	'spline',	0.04,	0.07,	0.006,	0.004],
                   [.1, 	.12,	0.005,	0.01,	'exponential'],
                   [.12,	.14,	0.01,	0.05,	'bessel',	0.8]]

instrument_3 = InstrumentGeometry(my_complex_bore)
print('INSTRUMENT 3:')
print(instrument_3) # display geometry information

fig3 = plt.figure(3)  # create new figure
instrument_3.plot_InstrumentGeometry(figure=fig3)  # plot the instrument
plt.title('a more complicated and smooth instrument')  # add a title

# or load the instrument from the file. The file 'Ex1_instrument_3.csv'
# corresponding to the same geometry, has the following content:
#
# .. code-block:: shell
#
#     ##  x_0  	    x_1    	r_0    	r_1    	type   	param
#        0      	0.0009 	0.0087 	0.0046 	Circle	-0.01
#        0.0009 	0.0014 	0.0046 	0.0024 	Circle	0.007
#        0.0014 	0.01   	0.0024 	0.003  	Cone
#        0.01   	0.03   	0.0042 	0.005  	Cone
#        0.03   	0.1    	0.005  	0.005  	Spline	0.04   	0.07   	0.006  	0.004
#        0.1    	0.12   	0.005  	0.01   	Exponential
#        0.12   	0.14   	0.01   	0.05   	Bessel	0.8
#
# The following command writes a file 'Ex1_instrument_3_test_MainBore.csv'
# having the same content than 'Ex1_instrument_3.csv'.
instrument_3.write_files(os.path.join(path, 'Ex1_instrument_3_test'), extension='csv')

# %% Mixing

# You can also mix the two methods:
my_mixed_bore = [[0.0,	0.0009,	0.0087,	0.0046,	'circle',	-0.01],
                 [0.0009,	0.0014,	0.0046,	0.0024,	'circle',	0.007],
                 [0.01,	0.003],
                 [0.01,	0.0042],
                 [0.03,	0.005],
                 [0.03,	.1,	    0.005,	0.005,	'spline',	0.04,	0.07,	0.006,	0.004],
                 [.1, 	.12,	0.005,	0.01,	'exponential'],
                 [.12,	.14,	0.01,	0.05,	'bessel',	0.8]]

instrument_4 = InstrumentGeometry(my_mixed_bore)
instrument_4.plot_InstrumentGeometry(figure=fig3, linestyle=':', color='k')


# Handling instrument side holes is the topic of Example 2 !
plt.show()
