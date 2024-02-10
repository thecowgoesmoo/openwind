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


"""Numerical scheme for a radiating end."""

import numpy as np

from openwind.continuous import RadiationPade
from openwind.temporal import TemporalComponentExit


class TemporalRadiation(TemporalComponentExit):
    """A model of radiation, for use in temporal simulation.

    Implements the radiation model in an energy-consistent way,
    using one internal variable.

    .. warning::
        The radiation model must be compatible with the temporal scheme: it
        must be a :py:class:`RadiationPade\
        <openwind.continuous.physical_radiation.RadiationPade>`

    It interacts with one pipe end. It accumulates energy in one
    internal variable, and dissipates energy according to the pressure
    at the end of the pipe.

    .. important::
        Only PH1 convention supported yet.

    Parameters
    ----------
    phy_radiation : :py:class:`RadiationPade<openwind.continuous.physical_radiation.RadiationPade>`
            The continuous radiation model, which must be written as a Padé
            development.
    pipe_end : :py:class:`TemporalPipeEnd <openwind.temporal.tpipe.TemporalPipeEnd>`
        the pipe end from which to radiate
    t_solver: :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance

    Attributes
    ----------
    alpha, beta, Zplus: float
        The coefficients of the physical radiation model
    m_end : float
        The inertial term at the pipe end due to waves propagation.
    """

    def __init__(self, phy_radiation, pipe_ends, t_solver):
        super().__init__(phy_radiation, t_solver)
        self.pipe_end, = pipe_ends
        assert isinstance(phy_radiation, RadiationPade)
        self._rad_model = phy_radiation
        self.reset_variables()
        self._opening_factor = 1.0
        self.update_coefficients()

    def reset_variables(self):
        # Internal variable
        self._zeta = 0.0

    def set_dt(self, dt):
        self._dt = dt
        self.half_dt = dt/2
        self._should_recompute_coefs = True

    def update_coefficients(self):
        """Get radiation coefficients alpha, beta and Zplus from the model"""
        alpha, beta, Zplus = self._rad_model.\
            compute_temporal_coefs(*self.pipe_end.get_physical_params(),
                                   self._opening_factor)
        self.alpha = alpha
        self.beta = beta
        self.Zplus = Zplus
        self._should_recompute_coefs = True

    def set_opening_factor(self, opening_factor):
        """Control opening and closing of the hole.

        Can be used during simulation. Energy-safe, as it only modifies
        the strength of the interaction with the tube.

        Parameters
        ----------
        opening_factor : float
            1 for an open hole, 0 for a closed hole, or any value in-between
            for a partly-closed hole
        """
        # Only update the factor if it changed significantly : too expensive (5,92 sec pour set_op contre 4,44)
        if not (abs(opening_factor - self._opening_factor) < 1e-5):
        #    np.isclose(opening_factor, self._opening_factor, atol=1e-5):
            self._opening_factor = opening_factor
            old_Zplus = self.Zplus
            self.update_coefficients()
            assert self.Zplus == old_Zplus  # Zplus is not allowed to change!


    def _precompute_coefficients(self):
        alpha = self.alpha
        beta = self.beta
        self.m_end = m_end = self._dt / (2*self.pipe_end.get_alpha())
        m_end_rad = m_end + self.half_dt * beta/self.Zplus
        rt_alpha = np.sqrt(alpha)
        # Influence of zeta on itself
        Z_dt = self.Zplus + self.half_dt**2 * alpha / m_end_rad
        self._step = self.Zplus / Z_dt
        self._infl = -self.half_dt * rt_alpha/Z_dt * m_end/m_end_rad
        # Influence of zeta on the flow
        self._zeta_to_flow = rt_alpha * m_end / m_end_rad
        # Influence of p_corr on the flow
        self._p_to_flow = -beta/self.Zplus * m_end/m_end_rad

        self._should_recompute_coefs = False


    def one_step(self):
        """Advance one time step.

        Update internal variable ``_zeta`` and the flux of the
        pipe_end.
        """
        if self._should_recompute_coefs:
            self._precompute_coefficients()

        p_corr = self.pipe_end.get_p_no_flow()
        zeta_n = self._zeta
        zeta_b = self._step * zeta_n + self._infl * p_corr
        self._zeta = 2 * zeta_b - zeta_n
        flow = self._zeta_to_flow * zeta_b + self._p_to_flow * p_corr
        # Flow calculated with old convention : bad sign
        self.pipe_end.update_flow(-flow)

        super().remember_flow_and_pressure(self.pipe_end)
        q_nph = self.pipe_end.get_q_nph()
        self._dissipated = self._dt * self.beta/self.Zplus * q_nph**2


    def get_exit_flow(self):
        return super().get_exit_flow()

    def get_exit_pressure(self):
        return super().get_exit_pressure()


    def __str__(self):
        return 'TRadiation({})'.format(self.pipe_end)

    def __repr__(self):
        return self.__str__()

    def energy(self):
        """
        Compute the amount of energy stored in the radiating element.

        Includes only the stored energy,
        i.e. the part that can be brought back into the pipe.
        The dissipated energy is ignored, but is guaranteed to be positive.

        Returns
        -------
        float
        """
        zeta = self._zeta
        return self.Zplus/2 * zeta**2

    def dissipated_last_step(self):
        return self._dissipated

    def get_maximal_dt(self):
        return np.infty
