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
This example shows how to add side holes to your instrument
"""
import os

import matplotlib.pyplot as plt

from openwind import InstrumentGeometry


path = os.path.dirname(os.path.realpath(__file__))
# In Ex1, we have learned how to import a geometry from a file :
my_instrument = InstrumentGeometry(os.path.join(path, "Ex2_instrument.txt"))

# %% Side holes

# Most wind instruments have side holes. In OpenWind, you can define your own
# side holes for your instrument.
# This can be done either directly in the code or in a independent file having
# both the same structure.
# * A first line/list with the columns names (in arbritrary order)
#   - 'label' indicating the column with the name of the holes
#   - 'position': the column with the location of the holes on the main bore (in meter)
#   - 'radius': the column with the radius of the holes (in meter)
#   - 'chimney': the column with the chimney height of the holes (in meter)
# * A line per hole with the right data
#
# .. note::
#       Currently, only cylindrical holes can be specified this way in OW.
#
# The file corresponding to an instrument with 9 side holes, has a structure
# similar to "Ex2_holes.txt":
#
# .. code-block:: shell
#
#       label	position  radius	chimney
#       ## ----------------------------------------
#       hole1	0.14	  0.0016    0.005
#       hole2	0.17	  0.0017    0.005
#       hole3	0.21	  0.0025    0.005
#       hole4	0.26	  0.0030    0.005
#       hole5	0.30	  0.0027    0.005
#       hole6	0.33	  0.0025    0.005
#       hole7	0.38	  0.0030    0.005
#       hole8	0.41	  0.0040    0.005
#       hole9	0.47	  0.0060    0.005
#
# Directly in python, it gives:
my_holes = [['label',   'position', 'radius', 'chimney'],
            ['hole1',	0.14,	    0.0016,    0.005],
            ['hole2',	0.17,	    0.0017,    0.005],
            ['hole3',	0.21,	    0.0025,    0.005],
            ['hole4',	0.26,	    0.0030,    0.005],
            ['hole5',	0.30,	    0.0027,    0.005],
            ['hole6',	0.33,	    0.0025,    0.005],
            ['hole7',	0.38,	    0.0030,    0.005],
            ['hole8',	0.41,	    0.0040,    0.005],
            ['hole9',	0.47,	    0.0060,    0.005]]

# To add holes to your instrument, simply add the file with the holes info (or the list)
# in your :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`.
# Make sure the file with the holes is second after the main bore geometry.

instrument_with_holes = InstrumentGeometry(os.path.join(path, "Ex2_instrument.txt"), my_holes)

fig1 = plt.figure(1)
instrument_with_holes.plot_InstrumentGeometry(figure=fig1)
plt.suptitle('wind instrument with side holes')

# %% Fingering Chart

# Side holes are useful for calculating the impedance or simulating the sound
# of your instrument for a given note, i.e., fingering. For this you need to
# specify which holes are open and which are closed.
# You can add a 'fingering chart' file to your instrument to make this step
# easier. It is a table in which
# * each column correspond to a fingering (the first one indicating the holes label)
# * each line correspond to one hole (the first one indicating the notes names)
#
# For each note, `x` indicates a closed hole and `o` an open one.
#
# .. warning::
#   The labels of the holes indicating in the first column must correspond to the
#   the ones given in the hole file!
#
# A fingering chart file with 8 notes associated to the instrument above
# is given in "Ex2_fingering_chart.txt". It has the following content
#
# .. code-block:: shell
#
#       label     C     D     E     F     G     A     B     C2
#       hole1     x     x     x     x     x     x     x     o
#       hole2     x     x     x     x     x     x     o     x
#       hole3     x     x     x     x     x     o     o     o
#       hole4     x     x     x     x     o     o     o     o
#       hole5     x     x     x     o     o     o     o     o
#       hole6     x     x     o     x     o     o     o     o
#       hole7     x     x     x     x     x     x     x     x
#       hole8     x     o     o     o     o     o     o     o
#       hole9     o     o     o     o     o     o     o     o
#
# The corresponding list is (quite heavy):
my_fing_chart = [['label', 'C', 'D', 'E', 'F', 'G', 'A', 'B', 'C2'],
                 ['hole1', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'o'],
                 ['hole2', 'x', 'x', 'x', 'x', 'x', 'x', 'o', 'x'],
                 ['hole3', 'x', 'x', 'x', 'x', 'x', 'o', 'o', 'o'],
                 ['hole4', 'x', 'x', 'x', 'x', 'o', 'o', 'o', 'o'],
                 ['hole5', 'x', 'x', 'x', 'o', 'o', 'o', 'o', 'o'],
                 ['hole6', 'x', 'x', 'o', 'x', 'o', 'o', 'o', 'o'],
                 ['hole7', 'x', 'x', 'x', 'x', 'x', 'x', 'x', 'x'],
                 ['hole8', 'x', 'o', 'o', 'o', 'o', 'o', 'o', 'o'],
                 ['hole9', 'o',	'o', 'o', 'o', 'o', 'o', 'o', 'o'] ]

# Simply add the fingering chart as third file for the
# :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`:
complete_instrument = InstrumentGeometry(os.path.join(path, "Ex2_instrument.txt"),
                                         os.path.join(path, "Ex2_holes.txt"),
                                         os.path.join(path, "Ex2_fingering_chart.txt"))

print(complete_instrument) # Display informations on the instrument

# With a fingering chart, you can plot the instrument for a given note :

fig2 = plt.figure(2)
complete_instrument.plot_InstrumentGeometry(figure=fig2, note='E')
plt.suptitle('wind instrument with side holes (closed holes are filled)')


# This instrument is now fully ready to be used in simulations !

# %% Write files

# It is possible to write files in the right format from constructed
# :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`.
# The following line command writes the files "Ex2_test_MainBore.csv",
# "Ex2_test_Holes.csv", "Ex2_test_FingeringChart.csv"
# Different options can be given to specify the unit, the number of digit etc.
# (for more details, see :py:func:`InstrumentGeometry.write_files() <openwind.technical.instrument_geometry.InstrumentGeometry.write_files>`).

complete_instrument.write_files(os.path.join(path, "Ex2_test"), extension='.csv')

plt.show()
# %% Use a single file

# Since Openwind 0.9.1, it is also possible to use a single file concatenating
# the three previous files: main bore, holes and fingering chart. This file can be written using the method:

complete_instrument.write_single_file(os.path.join(path, 'Ex2_single'), unit='mm', comments='this is a comment written in the file')

# to use this file, just give the right path:

instru_from1file = InstrumentGeometry(os.path.join(path, 'Ex2_single.ow'))