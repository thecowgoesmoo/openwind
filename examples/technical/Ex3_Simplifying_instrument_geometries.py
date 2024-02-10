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
This example shows how to simplify a complicated instrument geometry
"""
import os

import matplotlib.pyplot as plt

from openwind import InstrumentGeometry
from openwind.technical import AdjustInstrumentGeometry

path = os.path.dirname(os.path.realpath(__file__))

# If you wish to import a real existing instrument to OpenWind, you probably
# have access to it's bore, either via measurements or directly from the
# maker's plan.
# If you measured a real instrument, you have noted the radius of the bore at
# different points of the instrument - the more precise the measurement, the
# smaller the distance between the measurement points and therefore the more
# data.

# In OpenWind, calculating the acoustic behaviour of an instrument with a large
# amount of parts can take a (very) long time.
# Moreover, these data points can be noisy (due to measurements errors).
# This is why it is interesting to simplify a complicated instrument
# geometry.


# we load a complex instrument with a lot of segments
complex_instr = InstrumentGeometry(os.path.join(path, 'Ex3_complex_instrument.txt'))

fig1 = plt.figure(1)  # create new figure
complex_instr.plot_InstrumentGeometry(figure=fig1)  # plot the instrument
plt.title('A complex and noisy measurement with a lot of segments')

# The geometry simplification is done by an optimization process. We need to
# give OpenWind a starting shape and set the values we want to be optimized
# by writing them between '' and with a tilde ~ in front.

# Here we try to simplify our instrument down to 8 linear segments, where the
# lengths of the segments are fixed, and the radii are to be optimized.
simplified_bore = [[0.0, 0.2, '~0.005', '~0.005', 'linear'],
                   [0.2, 0.4, '~0.005', '~0.005', 'linear'],
                   [0.4, 0.6, '~0.005', '~0.005', 'linear'],
                   [0.6, 0.8, '~0.005', '~0.005', 'linear'],
                   [0.8, 1.0, '~0.005', '~0.005', 'linear'],
                   [1.0, 1.2, '~0.005', '~0.005', 'linear'],
                   [1.2, 1.4, '~0.005', '~0.005', 'linear'],
                   [1.4, 1.5, '~0.005', '~0.005', 'linear']]

simplified_instr = InstrumentGeometry(simplified_bore)

print("Simplified instrument with cones:")
# the AdjustInstrumentGeometry is instanciated from the two Instrument Geometries
adjustment = AdjustInstrumentGeometry(simplified_instr, complex_instr)
# the optimization process is carried out
adjusted_instr = adjustment.optimize_geometry(iter_detailed=False, max_iter=100)

fig2 = plt.figure(2)
complex_instr.plot_InstrumentGeometry(figure=fig2, linestyle=':')
adjusted_instr.plot_InstrumentGeometry(figure=fig2)
plt.title('A complex instrument (dotted line) and the simplification by' +
          ' conical segments (solid line)')


# As you can see, the simplification works well for the conical parts in the
# original instrument, but the round parts are not well approximated by
# conical segments.

# We need to help OpenWind a bit more.

# -----------------------------------------------------------------------------
print("\n-------------\n")

# By looking at the bore of the instrument we can make better guesses for the
# simplification.

# Straight parts are well approximated by linear segments, curved parts can
# at least be simplified to splines (although exp or bessel may give better
# results)
better_simpl_bore = [[0.0, 0.015, '~0.005', '~0.005', 'spline', 0.005, 0.01, '~0.005', '~0.005'],
                     [0.015, 0.1, '~0.005', '~0.005', 'linear'],
                     [0.1, 0.3, '~0.005', '~0.005', 'spline', 0.2, '~0.005'],
                     [0.3, 0.5, '~0.005', '~0.005', 'linear'],
                     [0.5, 1.0, '~0.005', '~0.005', 'linear'],
                     [1.0, 1.5, '~0.005', '~0.005', 'spline', 1.2, 1.3, '~0.005', '~0.005']]

better_simpl_instr = InstrumentGeometry(better_simpl_bore)

print("Simplified instrument with complex shapes:")
# the AdjustInstrumentGeometry is instanciated from the two Instrument Geometries
better_adjust = AdjustInstrumentGeometry(better_simpl_instr, complex_instr)
# the optimization process is carried out
better_adjusted_instr = better_adjust.optimize_geometry(iter_detailed=False,
                                                   max_iter=100)


fig3 = plt.figure(3)
complex_instr.plot_InstrumentGeometry(figure=fig3, linestyle=':')
better_adjusted_instr.plot_InstrumentGeometry(figure=fig3)
plt.title('A complex instrument (dotted line) and the simplification by' +
          ' well-chosen different shapes (solid line)')

# Much better ! Some errors remain, for instance at the end of the horn. This
# is a matter of tweaking and trying out different shapes and parameters.

# -----------------------------------------------------------------------------

# With this method a complex instrument described by 1500 sections is
# simplified to 6 parts, which greatly decreases computation time.


# The new simplified instrument can be saved in a file :
better_adjusted_instr.write_files(os.path.join(path, 'simplified_instrument'))

plt.show()
