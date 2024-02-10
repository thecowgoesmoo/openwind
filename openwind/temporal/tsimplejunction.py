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


"""Numerical scheme for a junction of two pipes."""

import numpy as np

from openwind.temporal import TemporalComponent

class TemporalSimpleJunction(TemporalComponent):
    # TODO: the same, with any number of pipes?
    """
    A model of junction, for use in temporal simulation.

    Interacts with two pipe end.
    Ensures pressure continuity, and flow conservation (Kirchhoff conditions).
    Does not store energy.

    .. important::
        Only 'PH1' convention supported yet.

    Parameters
    ----------
    dt : float
        time step increment
    pipe_ends: list of 2 :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`
        the two pipe ends to connect
    t_solver : :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance
    """

    def __init__(self, simplejunction, pipe_ends, t_solver):
        super().__init__(simplejunction, t_solver)
        self.pipe_end_1, self.pipe_end_2 = pipe_ends

    def set_dt(self, dt):
        self.dt = dt
        self._should_recompute_coefficients = True

    def reset_variables(self):
        # Nothing to do!
        pass

    def _precompute_coefficients(self):
        alpha_1 = self.pipe_end_1.get_alpha()
        alpha_2 = self.pipe_end_2.get_alpha()
        self._dp_to_flow = 1/(alpha_1 + alpha_2)

        self._should_recompute_coefficients = False


    def one_step(self):
        if self._should_recompute_coefficients:
            self._precompute_coefficients()

        # p_corr_1 + dt/(2 m_end_1) * lambda = p_corr_2 - dt/(2 m_end_2) * lambda
        p_corr_1 = self.pipe_end_1.get_p_no_flow()
        p_corr_2 = self.pipe_end_2.get_p_no_flow()
        flow = self._dp_to_flow * (p_corr_2 - p_corr_1)

        self.pipe_end_1.update_flow(-flow)
        self.pipe_end_2.update_flow(flow)

    def __str__(self):
        return 'TSimpleJunction({},{})'.format(self.pipe_end_1, self.pipe_end_2)

    def __repr__(self):
        return self.__str__()

    def energy(self):
        return 0

    def dissipated_last_step(self):
        return 0

    def get_maximal_dt(self):
        return np.infty
