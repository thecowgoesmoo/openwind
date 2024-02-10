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

import warnings
import os

import numpy as np
import matplotlib.pyplot as plt

from openwind import InstrumentGeometry, InstrumentPhysics, Player, FrequentialSolver
from openwind import ImpedanceComputation
from openwind.continuous import Physics
from openwind.impedance_tools import read_impedance, write_impedance, plot_impedance



def ComputeTransferMatrix(geom, freq, temperature, losses=True, **kwargs):
    """
    Computes the transfer matrix for a given geometry and for a frequency range

    Parameters
    ----------
    geom : string or list
        path to instrument geometry file ; or list describing the geometry.
    freq : ndarray or list
        frequency range.
    temperature : float
        The temperature in Celcius degree.
    losses : Boolean, optional
        Defines if losses are taken into account in computation.
        The default is True.
    **kwargs : keyword argument
        Any option wich can be used with:
        :py:class:`InstrumentPhysics<openwind.technical.instrument_geometry.InstrumentGeometry>`,
        :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`,
        :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`,
        :py:func:`FrequentialSolver.solve() <openwind.frequential.frequential_solver.FrequentialSolver.solve()>`,

    Returns
    -------
    a, b, c, d : 4 ndarray with same length as freq
        The four coefficients of the transfer matrix

    """

    kwargs_geom, kwargs_phy, kwargs_freq, kwargs_solve = ImpedanceComputation._check_kwargs(kwargs)
    kwargs_phy.pop('radiation_category', None)
    kwargs_solve.pop('interp', None)
    kwargs_solve.pop('interp_grid', None)


    instr_geom = InstrumentGeometry(geom, **kwargs_geom)
    player = Player()

    closed_instr = InstrumentPhysics(instr_geom,
                                     temperature=temperature,
                                     player=player,
                                     losses=losses,
                                     radiation_category='closed',
                                     **kwargs_phy)

    open_instr = InstrumentPhysics(instr_geom,
                                   temperature=temperature,
                                   player=player,
                                   losses=losses,
                                   radiation_category='perfectly_open',
                                   **kwargs_phy)

    closed_model = FrequentialSolver(closed_instr, freq, **kwargs_freq)

    open_model = FrequentialSolver(open_instr, freq, **kwargs_freq)

    closed_model.solve(interp=True, **kwargs_solve)
    open_model.solve(interp=True, **kwargs_solve)

    # closed end : u2 = 0 ; a = p1/p2 ; c = u1 / p2
    # open end :   p2 = 0 ; b = p2/u2 ; d = u1 / u2
    p_closed = np.array(closed_model.pressure)
    u_closed = np.array(closed_model.flow)
    p_open = np.array(open_model.pressure)
    u_open = np.array(open_model.flow)
    a = p_closed[:, 0]/p_closed[:, -1]
    b = p_open[:, 0]/u_open[:, -1]
    c = u_closed[:, 0]/p_closed[:, -1]
    d = u_open[:, 0]/u_open[:, -1]

    if not np.all(np.isclose(a*d - c*b, 1)):
        f_min = np.min(np.array(freq)[~(np.isclose(a*d - c*b, 1))])
        f_max = np.max(np.array(freq)[~(np.isclose(a*d - c*b, 1))])
        warnings.warn('determinant of transfer matrix is not close to 1 ' +
                       'for frequencies in [{:.2f}, {:.2f}] Hz.'.format(f_min, f_max))

    return a, b, c, d



def remove_adaptor(adaptor_geom, filenames, temperature, humidity=.5, write_files=True, display=False, **kwargs):
    """
    Estimate the impedance "after" an adaptor with known geoemtry.

    If :math:`Z_1, Z_2` are respectively the measured impedance and the impedance
    after the adaptor, with have:

    .. math::
        Z_2 = \\frac{B - DZ_1}{CZ_1 - A}

    with :math:`A,B,C,D` the coefficients of the transfer matrix of the adaptor.

    Parameters
    ----------
    adaptor_geom : list
        Geometry of the adaptor in openwind format.
    filenames : str or list of str
        the filenames (+path) of all the measurements which should be treated.
    temperature : float
        the measurement temperature.
    humidity : float, optional
        the humidity rate. The default is .5.
    write_files : booelan, optional
        If true, write the corrected impedance at the original path, with prefix
        "adaptor_removed". Default: True
    display: boolean, optional
        If true plot the impedances. Default: False

     Returns
     -------
     freq_cor : list of array
         The list of the frequency axis for each measurement.
     Z_cor : list of array
         The list of the corrected impedances for each measurement.
    """
    my_geom = InstrumentGeometry(adaptor_geom)
    R0 = my_geom.main_bore_shapes[0].get_radius_at(0)
    R1 = my_geom.main_bore_shapes[-1].get_radius_at(1)

    my_phy = Physics(temperature, humidity=humidity)
    rho, c = my_phy.get_coefs(0, 'rho', 'c')
    Zc_adaptor = rho*c/(np.pi*R0**2)
    Zc_instru = rho*c/(np.pi*R1**2)

    freq_adapt = np.array([])

    if isinstance(filenames, str):
        filenames = [filenames]


    Z_cor = list()
    freq_cor = list()
    for file in filenames:
        freq, Zraw = read_impedance(file)

        if len(freq_adapt)!=len(freq) or not np.all(freq_adapt==freq):
            # print('recompute transfer matrix for new freq axis')
            a,b,c,d = ComputeTransferMatrix(adaptor_geom, freq, temperature, losses=True, humidity=humidity, **kwargs)
            freq_adapt=freq


        Zraw_dim = Zraw*Zc_adaptor
        Z_wo_adaptor_dim = (b - d*Zraw_dim)/(c*Zraw_dim - a)
        Z_wo_adaptor = Z_wo_adaptor_dim/Zc_instru
        Z_cor.append(Z_wo_adaptor)
        freq_cor.append(freq)

        path, filename = os.path.split(file)
        if display:
            fig = plt.figure()
            plot_impedance(freq, Zraw, label='raw', figure=fig)
            plot_impedance(freq, Z_wo_adaptor, label='wo adaptor', figure=fig)
            plt.suptitle(filename)

        if write_files:
            write_impedance(freq, Z_wo_adaptor, os.path.join(path, 'adaptor_removed_' + filename))

    return freq_cor, Z_cor
