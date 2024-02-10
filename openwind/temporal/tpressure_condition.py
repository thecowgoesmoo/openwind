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


"""Temporal version of a pressure constraint."""

import numpy as np

from openwind.temporal import TemporalComponentExit


class TemporalPressureCondition(TemporalComponentExit):
    """
    Constrain the value of the pressure P to zero at the end of the pipe.

    See Also
    --------
    :py:class:`RadiationPerfectlyOpen<openwind.continuous.physical_radiation.RadiationPerfectlyOpen>`
        The continuous version of this radiation condition.


    Parameters
    ----------
    phy_radiation : :py:class:`RadiationPerfectlyOpen<openwind.continuous.physical_radiation.RadiationPerfectlyOpen>`
            The continuous radiation model, which must be a perfectly open
            condition (:math:`Z_r = 0`)
    pipe_ends : tuple of 1 :py:class:`TemporalPipeEnd <openwind.temporal.tpipe.TemporalPipeEnd>`
        the pipe end from which to radiate
    t_solver: :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance
    """

    def __init__(self, phy_radiation, pipe_ends, t_solver):
        super().__init__(phy_radiation, t_solver)
        self._pipe_end, = pipe_ends

    def one_step(self):
        p_no_flow = self._pipe_end.get_p_no_flow()
        alpha = self._pipe_end.get_alpha()
        # p^{n+1/2} = p_no_flow - alpha * w must be zero
        w = p_no_flow / alpha
        self._pipe_end.update_flow(w)
        super().remember_flow_and_pressure(self._pipe_end)

    def set_dt(self, dt):
        pass  # nothing to do

    def get_maximal_dt(self):
        return np.infty  # no CFL

    def reset_variables(self):
        pass  # nothing to do

    def energy(self):
        return 0

    def dissipated_last_step(self):
        return 0

    def __repr__(self):
        return "TPressure(0, {})".format(str(self._pipe_end))
