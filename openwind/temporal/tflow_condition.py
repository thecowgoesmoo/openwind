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


"""Temporal version of a flow constraint."""

import numpy as np

from openwind.temporal import TemporalComponentExit


class TemporalFlowCondition(TemporalComponentExit):
    """Constrain the value of the flow V at the end of the pipe.

    This boundary condition may be used for closed pipes, but
    its main purpose is for simulating the response of an instrument to an
    impulse, a chirp tone, or any such input signal.

    .. important::
        Only 'PH1' convention is supported for now.

    Parameters
    ----------
    flow : float or :py:class:`Flow<openwind.continuous.excitator.Flow>`
        If float, the constant value of the flow. Either, the object must have
        an attribute :py:attr:`input_flow<openwind.continuous.excitator.Flow.input_flow>`
        with a methode
        :py:meth:`get_value(t)<openwind.continuous.excitator.ExcitatorParameter.get_value>`
        returning the value of the flow at time t
    pipe_end : :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`
        The PipeEnd on which this condition applies
    t_solver: :py:class:`TemporalSolver <openwind.temporal.temporal_solver.TemporalSolver>`
        The associated solver
    """

    def __init__(self, flow, pipe_ends, t_solver):
        super().__init__(flow, t_solver)
        self._pipe_end, = pipe_ends
        self._flow = flow  # Continuous model
        self._input_flow = flow.input_flow

    def one_step(self):
        flow_value = self._input_flow.get_value(self.get_current_time())
        self._pipe_end.update_flow(flow_value/self._flow.scaling.get_scaling_flow())
        super().remember_flow_and_pressure(self._pipe_end)

    def set_dt(self, dt):
        pass # nothing to do

    def get_maximal_dt(self):
        return np.infty # no CFL

    def reset_variables(self):
        pass # nothing to do

    def energy(self):
        """
        Amount of energy currently stored in element

        As a TemporalFlowCondition does not store energy, return zero.
        It may be a source of energy into the system, but the
        amount cannot be known in advance.

        Returns
        -------
        0
            The stored energy
        """
        return 0

    def dissipated_last_step(self):
        """
        Amount of energy dissipated by this component during the last time step.

        A nonzero flow condition is an external source of energy.
        It does not count as dissipation. However let us use this method
        abusively to also represent a source term.

        Returns
        -------
        float
            The "dissipated" energy
        """
        return self.get_exit_flow() * self.get_exit_pressure() * self._t_solver.get_dt() / self._flow.scaling.get_scaling_power()

    def __str__(self):
        return "TFlow({}, {})".format(str(self._flow), str(self._pipe_end))

    def __repr__(self):
        return self.__str__()
