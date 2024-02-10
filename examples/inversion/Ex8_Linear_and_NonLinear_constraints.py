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
How to use constrained optimization with openwind
"""

import numpy as np

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)


# 1. Global options

frequencies = np.linspace(100, 500, 10)
temperature = 25
losses = True
player = Player()

# 2. Targets definitions
# For this example we use simulated data instead of measurement
# The geometry is 0.5m conical pipe with radii within of [2mm, 20mm].
target_geom = [[0, 0.5, 2e-3, 20e-3, 'linear']]
target_computation = ImpedanceComputation(frequencies, target_geom,
                                          temperature=temperature,
                                          losses=losses)

# The impedance used in target must be normalized
Ztarget = target_computation.impedance/target_computation.Zc

# noise is added to simulate measurement
noise_ratio = 0.01
Ztarget = Ztarget*(1 + noise_ratio*np.random.randn(len(Ztarget)))

# %% Length constraint

print('***** Length Constraints *****')

# In addition of the bounds on a given design parmeters, it can be usefull to
# include also constraints depending on several parameters.

# If we try to optimize the following trouble an Error is raised:

inverse_geom = [[0, '0.25', 2e-3, 10e-3, 'linear'],
                ['0.25', '~0.26', 20e-3, 10e-3, 'linear']]
try:
    instru_geom = InstrumentGeometry(inverse_geom)
    instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
    inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)
    result = inverse.optimize_freq_model(iter_detailed=True)
except AssertionError as e:
    print('The following error occurs:')
    print(e)
    print('\n')

# The length of the second pipe becomes negatives. This could be solve changing
# the low bound of the right position : ".25<~.26", but sometimes, it is more convenient
# to constrain the length of this pipe to be positive.
# the length being defined in OW as x1-x0, it is a linear combination of design parameters
# and the constraint must be treated like this.

instru_geom = InstrumentGeometry(inverse_geom)
instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)

# The length of a specific part can be constrained by using
# :py:meth:`DesignShape.create_length_constraint() <openwind.design.design_shape.DesignShape.create_length_constraint>`:

my_shape = instru_geom.main_bore_shapes[1] # get the shape on which apply the constrain
my_shape.create_length_constraint(Lmin=0, Lmax=np.inf) # create the constrain with the appropriate method

# Lmin and Lmax are optional, by default they are 0 and +inf.
# It is also possible to constrain similarly the length of all the pipe of an instrument by using
# :py:meth:`InstrumentGeometry.constrain_parts_length() <openwind.technical.instrument_geometry.InstrumentGeometry.constrain_parts_length>`:

instru_geom.constrain_parts_length()

# the constraints are stored in the :py:class:`OptimizationParameters <openwind.design.design_parameter.OptimizationParameters>` object and can be displayed:

print(instru_geom.optim_params)

# .. warning:: With these constraints only the 'trust-constr' and 'SLSQP' algorithms can be used
#
# Now the inversion converged without error:

result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP')



# %% Constrain the distances between the nodes of a spline

print('\n***** Spline Constraints *****')

# Similar constraint can be applied to the node of a spline, indeed, to avoid
# trouble, the order of the nodes must be conserved during the optimization process.
# This condition is often violated during the optimization process.
# let's try we the following measurement


inverse_geom = [[0, '0<~0.25', 2e-3, 7.5e-3, 'linear'],
                ['0<~0.25', '~.5', 7.5e-3, 20e-3, 'spline', '~0.26', '~0.28',17e-3, 10e-3]]

try:
    instru_geom = InstrumentGeometry(inverse_geom)
    instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
    inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)
    result = inverse.optimize_freq_model(iter_detailed=True)
except Exception as e:
    print('The following error occurs:')
    print(e)
    print('\n')

# Here again it is possible to add constrain by treating specificaly this part
# of the main bore, or directly from the `InstrumentGeometry` object.
# In this second case, the minimal distance between nodes is se to global minimal
# length divided by the number of internode

instru_geom = InstrumentGeometry(inverse_geom) # instanciate the InstruGeom
instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses) # instanciate InstruPhy
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget) # instanciate the InverseFreqResp

my_spline = instru_geom.main_bore_shapes[1] # get the Main Bore shape on which apply the constrain
my_spline.create_nodes_distance_constraints(Dmin=1e-3) # create the constrain => this is automatically added to the OptimParam object
print(instru_geom.optim_params) # the constraint is now displayed

instru_geom.constrain_parts_length(Lmin=1e-3) # creation from the `InstrumentGeometry` class
print(instru_geom.optim_params)


result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP') # optimization with SLSQP algo

# %% Constrain the conicity

print("\n***** Conicity *****")

# For conical parts, it is also possible to constrain their conicity.
# We define here the conicity as the local slope (for a cone: Delta R/L)
# It is possible to bounds the concity value with `Cmin` and `Cmax` or impose
# to keep the same value as initial with the keyword argument `keep_constant=True`

inverse_geom = [[0, '0.25', 2e-3, 10e-3, 'linear'],
                ['0.25', '~0.37', 10e-3, '~16e-3', 'linear']]

instru_geom = InstrumentGeometry(inverse_geom)
instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)


instru_geom.constrain_parts_length() # constrain the length to be positive
my_cone = instru_geom.main_bore_shapes[1] # get the shape to constrain
my_cone.create_conicity_constraint(Cmin=-np.inf, Cmax=np.inf, keep_constant=True) # create the constraint on conicity

# This time both linear and non-linear constraints are display from the `OptimizationParameters` object
print(instru_geom.optim_params) # all the constraints can be display from the `optim_params` object
print('Initial conicity {:.2g}'.format(my_cone.get_conicity_at(0)))


result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP')

print(instru_geom.optim_params)
print('Final conicity {:.2g}'.format(my_cone.get_conicity_at(0)))



# %% Constrain the radius and position of hole

print("\n***** Holes' position and radius *****")
# When the radius and the position of the hole are defined relatively to the main bore
# (cf Ex.4), the constraint of its position and radius can be done only by using non linear constraint
# it is defined only be adding bounds to the definition of the geometry

# 1. Target definition
geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
target_hole = [['label', 'position', 'radius', 'chimney'],
               ['hole1', .25, 3e-3, 5e-3],
               ['hole2', .35, 4e-3, 7e-3]]
fingerings = [['label', 'A', 'B', 'C', 'D'],
              ['hole1', 'x', 'x', 'o', 'o'],
              ['hole2', 'x', 'o', 'x', 'o']]
noise_ratio = 0.01


target_computation = ImpedanceComputation(frequencies, geom, target_hole,
                                          fingerings,
                                          temperature=temperature,
                                          losses=losses)
notes = target_computation.get_all_notes()

Ztargets = list()
for note in notes:
    target_computation.set_note(note)
    Ztargets.append(target_computation.impedance/target_computation.Zc
                    * (1 + noise_ratio*np.random.randn(len(frequencies))))

# 2. Definition of the initial geometry with constraints

inverse_geom =  [[0, '0.05<~0.3', 2e-3, '0<~2e-3', 'linear']]
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '.05<~0.1%<.27', '1e-3<~1.75e-3%<2e-3', 5e-3],
                ['hole2', '~0.2%', '~1.75e-3%', 7e-3]]
instru_geom = InstrumentGeometry(inverse_geom, inverse_hole, fingerings)

# this time non-linear constraints are displayed
print(instru_geom.optim_params)

instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztargets, notes=notes)
result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP')

print(instru_geom.optim_params)
# Unlike in Ex.4, this time the "target geom" is not reached due to the constraint.
# We can see that one constrained is active at the end of the optimization process.
# The radius of Hole1 is 2mm.


# %% Constrain the distance between holes

# The distance between the holes can be constrained with the methode:
# :py:meth:`InstrumentGeometry.constrain_all_holes_distance() <openwind.technical.instrument_geometry.InstrumentGeometry.constrain_all_holes_distance>`
# or :py:meth:`InstrumentGeometry.constrain_2_holes_distance() <openwind.technical.instrument_geometry.InstrumentGeometry.constrain_2_holes_distance>`

# %%% Hole centers distance
inverse_geom =  [[0, 0.5, 2e-3, 10e-3, 'linear']]
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '~0.1%', '~3e-3%', 5e-3],
                ['hole2', '~0.12%', '~4e-3%', 7e-3]]
instru_geom = InstrumentGeometry(inverse_geom, inverse_hole, fingerings)

# the distance between all hole centers is imposed to be below 5cm
instru_geom.constrain_all_holes_distance(Lmax=0.05)

# It is also possible to constrain a given holes couple (not necessary adjacent)
instru_geom.constrain_2_holes_distance('hole1', 'hole2', Lmax=0.05)

# this time non-linear constraints are displayed
print(instru_geom.optim_params)

instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)

inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztargets, notes=notes)
result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP')

print(instru_geom.optim_params)
# The "target geom" is not reached due to the constraint.
# We can verify the respect of the contraint:
distance = instru_geom.holes[1].position.get_value() - instru_geom.holes[0].position.get_value()
print(f'Distance between the holes: {distance*100:.2f}cm <= 5cm')

# %%% Hole edges distance

instru_geom = InstrumentGeometry(inverse_geom, inverse_hole, fingerings)

# This time we constrain the distance between the edges of the hole with the keyword "edges"
instru_geom.constrain_all_holes_distance(Lmax=0.05, edges=True)
print(instru_geom.optim_params)
instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztargets, notes=notes)
result = inverse.optimize_freq_model(iter_detailed=True, algorithm='SLSQP')
# The "target geom" is not reached due to the constraint.
# We can verify the respect of the contraint:
distance = instru_geom.holes[1].position.get_value() - instru_geom.holes[0].position.get_value()
print(f'Distance between the hole centers: {distance*100:.2f}cm <= 5cm')

distance_edges = instru_geom.holes[1].position.get_value() - instru_geom.holes[0].position.get_value() + instru_geom.holes[1].shape.get_radius_at(0) - instru_geom.holes[0].shape.get_radius_at(0)
print(f'Distance between the hole edges: {distance_edges*100:.2f}cm <= 5cm')
