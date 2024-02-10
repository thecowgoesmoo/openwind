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
How to chose the observable used in the optimization process and how to define its own.
"""

import numpy as np
import matplotlib.pyplot as plt

from openwind.inversion import InverseFrequentialResponse

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      InstrumentPhysics)


# %% Chose the observable

# The principle of inversion is to minimize the deviation between target and simulated impedance trough an observable.
# The choice of the observable is crucial in the optimization process, it can influence the attraction bassins and the final result.
# This example illustrate this influence.

# Targets definition: for this example we use simulated data
geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
target_hole = [['label', 'position', 'radius', 'chimney'],
               ['hole1', .25, 3e-3, 5e-3],
               ['hole2', .35, 4e-3, 7e-3]]
fingerings = [['label', 'A', 'B', 'C', 'D'],
              ['hole1', 'x', 'x', 'o', 'o'],
              ['hole2', 'x', 'o', 'x', 'o']]
noise_ratio = 0.01

frequencies = np.linspace(50, 2000, 100)
temperature = 20
losses = True

target_computation = ImpedanceComputation(frequencies, geom, target_hole,
                                          fingerings,
                                          temperature=temperature,
                                          losses=losses, note='B')
# normalize and noised impedance
Ztarget = (target_computation.impedance/target_computation.Zc
           * (1 + noise_ratio*np.random.randn(len(frequencies))))

# Construction of the inverse problem:
inverse_hole = [['label', 'position', 'radius', 'chimney'],
                ['hole1', '~.25', 3e-3, 5e-3],
                ['hole2', .35, 4e-3, 7e-3]]
instru_geom = InstrumentGeometry(geom, inverse_hole, fingerings)
instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)

# By default the observable is the reflection function, you can specify it
# when you instanciate the InverseFrequentialResponse with the keyword:
# 'observable'
inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget,
                                     notes='B', observable='reflection')
# or modify it a posteriori
inverse.set_observation('impedance')
# it is necessary to update the target w.r. to the new observable
inverse.set_targets_list(Ztarget, 'B')

# you can chose between the predefined observables:
observablaes = ['impedance',  # the impedance, the observable is the identity
                'impedance_modulus',  # the module of the impedance
                'impedance_phase',  # the impedance angle
                'reflection',  # the reflection function (Z-Zc)/(Z+Zc)
                'reflection_modulus',  # the reflec. func. modulus
                'reflection_phase',  # the reflec. func. angle
                'reflection_phase_unwraped']  # the unwrapped angle

# we observe the evolution of the cost functions associated to each of this
# observables w.r. to the hole1 radius.
values = np.linspace(7e-2, 0.34, 50).tolist()
impedance = np.zeros(len(values), dtype=float)
impedance_modulus = np.zeros(len(values), dtype=float)
impedance_phase = np.zeros(len(values), dtype=float)
reflection = np.zeros(len(values), dtype=float)
reflection_modulus = np.zeros(len(values), dtype=float)
reflection_phase = np.zeros(len(values), dtype=float)
reflection_phase_unwrap = np.zeros(len(values), dtype=float)


for k, value in enumerate(values):
    print('Value {}/{}'.format(k+1, len(values)))
    inverse.set_observation('impedance')
    inverse.set_targets_list(Ztarget, 'B')
    # We use it the method giving the cost, gradient and hessian vor a given
    # set of values for the design varaible.
    impedance[k], grad, hessian = inverse.get_cost_grad_hessian([value])

    inverse.set_observation('impedance_modulus')
    inverse.set_targets_list(Ztarget, 'B')
    # By default, the function keep the same value for the design variable
    impedance_modulus[k], grad, hessian = inverse.get_cost_grad_hessian()

    inverse.set_observation('impedance_phase')
    inverse.set_targets_list(Ztarget, 'B')
    impedance_phase[k], grad, hessian = inverse.get_cost_grad_hessian()

    inverse.set_observation('reflection')
    inverse.set_targets_list(Ztarget, 'B')
    reflection[k], grad, hessian = inverse.get_cost_grad_hessian()

    inverse.set_observation('reflection_modulus')
    inverse.set_targets_list(Ztarget, 'B')
    reflection_modulus[k], grad, hessian = inverse.get_cost_grad_hessian()

    inverse.set_observation('reflection_phase')
    inverse.set_targets_list(Ztarget, 'B')
    reflection_phase[k], grad, hessian = inverse.get_cost_grad_hessian()

    inverse.set_observation('reflection_phase_unwraped')
    inverse.set_targets_list(Ztarget, 'B')
    reflection_phase_unwrap[k], grad, hessian = inverse.get_cost_grad_hessian()

plt.figure()
plt.semilogy(np.asarray(values), impedance, label='Impedance')
plt.semilogy(np.asarray(values), impedance_modulus, label='Impedance Modulus')
plt.semilogy(np.asarray(values), impedance_phase, label='Impedance Phase')
plt.semilogy(np.asarray(values), reflection, label='Reflection')
plt.semilogy(np.asarray(values), reflection_modulus,
             label='Reflection Modulus')
plt.semilogy(np.asarray(values), reflection_phase, label='Reflection Phase')
plt.semilogy(np.asarray(values), reflection_phase_unwrap,
             label='Reflection Unwrap')
plt.grid(True)
plt.legend()
plt.xlabel('Hole1 radius')
plt.ylabel('Cost')

# We can see here that except the wraped reflection function phase all the cost
# function are smooth but they show local minimums which can be problematic.
# For the reflection function and the impedance phase, it is less pronounced
# %% Define your own observable


# It is possible to define your own observable. To do that you need to define
# a function which return this observation from Z
def my_own_observable(Z):
    """
    This method return my observable: here the (impedance angle)+10

    Parameters
    ----------
    Z : np.array
        The impedance

    Returns
    -------
    np.array()
    """
    return np.angle(Z)+10


# You need also to define a function which return the derivative of this
# observable which respect to Z and the derivative of the conjugate of this
# observable w.r. to Z.


# .. warning::
#     Z is a complex vector: :math:`d/dZ = 0.5(d/d(real(Z)) -jd/d(imag(Z)))`

def diff_my_own_observable_wrZ(Z):
    """
    Return the derivative of the observable and of the conjugate of this
    observable with respect to Z

    Parameters
    ----------
    Z : np.array
        The impedance

    Returns
    -------
    diff_obs : np.array
        The derivative of the obs. w.r. to Z
    diff_conj_obs : np.array
        The derivative of the conjugate of the obs. w.r. to Z

    """
    diff_obs = -1j * Z.conjugate() / (2 * np.abs(Z)**2)
    # the phase being a purely real number the two derivatives are equal
    diff_conj_obs = diff_obs
    return diff_obs, diff_conj_obs


# you can now use this new observable on the previous problem
inverse.set_observation((my_own_observable, diff_my_own_observable_wrZ))
inverse.set_targets_list(Ztarget, 'B')

# and compute the corresponding cost
my_own = np.zeros(len(values), dtype=float)
for k, value in enumerate(values):
    my_own[k], grad, hessian = inverse.get_cost_grad_hessian([value])


plt.semilogy(np.asarray(values), my_own,
             label='My own observable', color='black', linestyle=':')
plt.legend()

plt.show()
