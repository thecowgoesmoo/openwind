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
Numerical scheme for a junction of three pipes.
"""

import numpy as np
from numpy.linalg import inv
from scipy.sparse import diags

from openwind.temporal import TemporalComponent


class TemporalJunction(TemporalComponent):
    """
    Junction of three Pipes for time-domain simulation.

    .. warning::
        uses modified acoustic masses, that do not correspond exactly \
        to the model from [Chaigne--Kergomard]. Instead of a negative \
        acoustic mass, we use a small positive one, computed by \
        :py:meth:`JunctionTjoint.compute_passive_masses()\
        <openwind.continuous.junction.JunctionTjoint.compute_passive_masses>`.

    Parameters
    ----------
    junct : :py:class:`JunctionTjoint<openwind.continuous.junction.JunctionTjoint>`
        The physics of the associated T-junction
    ends: list of 3 :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`
        The three pipe ends the junction connects in the order:

        1. main bore: upstream pipe
        2. main bore: downstream pipe
        3. side pipe (hole)

    t_solver : :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance

    Attributes
    ----------
    m11, m12, m22 : np.array
        the passive masses
    dt: float
        the time step
    Step: np.array
        Matrix used to update gamma by half a step
    Infl: np.array
        Influence of each pipe's pressure and velocity on the update of gamma
    """

    def __init__(self, junct, ends, t_solver):
        super().__init__(junct, t_solver)
        self._end1, self._end2, self._end3 = ends

        r_main, rho = self._end1.get_physical_params()[0:2]
        r_side = self._end3.get_physical_params()[0]
        self.m11, self.m12, self.m22 = junct.compute_passive_masses(r_main, r_side, rho)

        self.reset_variables()
        self._should_recompute_constants = True

    def set_dt(self, dt):
        self.dt = dt
        self._should_recompute_constants = True

    def reset_variables(self):
        self._gamma = np.zeros(2)
        self._P_corr = np.zeros(3)

    def _precompute_constants(self):
        """Compute constant matrices used in the update"""
        m_end1 = self.dt / (2 * self._end1.get_alpha())
        m_end2 = self.dt / (2 * self._end2.get_alpha())
        m_end3 = self.dt / (2 * self._end3.get_alpha())

        M_J = np.array([[self.m11, self.m12],
                        [self.m12, self.m22]])
        invM_end = diags([1/m_end1, 1/m_end2, 1/m_end3])
        T_J = np.array([[1, 0, -1],
                        [0, 1, -1]])

        # Matrix of self-contribution of gamma trough the evolution of pressure
        A_g = T_J @ invM_end @ T_J.T
        # Matrix M_{J,dt}, to invert for performing a half-timestep
        M_Jdt = M_J + self.dt**2/4 * A_g
        invM_Jdt = np.array(inv(M_Jdt))
        # Matrix used to update gamma by half a step
        self.Step = invM_Jdt @ M_J
        # Influence of each pipe's pressure and velocity on the update of gamma
        self.Infl = -self.dt/2 * invM_Jdt @ T_J
        # print(self.Step, type(self.Step), self.Infl, type(self.Infl))

        self._should_recompute_constants = False

    def _read_end_values(self):
        # Predicted value of P at time n+1/2
        self._P_corr[0] = self._end1.get_p_no_flow()
        self._P_corr[1] = self._end2.get_p_no_flow()
        self._P_corr[2] = self._end3.get_p_no_flow()

    def _update_flows(self, gamma_b):
        # Flow calculated with old convention : inverted sign
        self._end1.update_flow(-gamma_b[0])
        self._end2.update_flow(-gamma_b[1])
        self._end3.update_flow(gamma_b[0] + gamma_b[1])

    def one_step(self):
        """
        Advance one time step

        Computes the evolution of internal variables ``gamma``,
        and updates the flow on the three connected
        :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`.
        """
        if self._should_recompute_constants:
            self._precompute_constants()

        self._read_end_values()
        gamma = self._gamma
        gamma_b = self.Step @ gamma + self.Infl @ self._P_corr
        self._gamma = 2*gamma_b - gamma
        self._update_flows(gamma_b)

    def __str__(self):
        name = "TJunction({}, {}, {})"
        return name.format(self._end1, self._end2, self._end3)
    def __repr__(self):
        return self.__str__()

    def energy(self):
        """
        Compute the amount of energy stored in the junction at time step.

        Returns
        -------
        float
            The stored energy
        """
        gamma = self._gamma
        energy_1 = self.m11/2 * gamma[0]**2
        energy_2 = self.m22/2 * gamma[1]**2
        energy_cross = self.m12 * gamma[0] * gamma[1]
        return energy_1 + energy_2 + energy_cross

    def dissipated_last_step(self):
        """
        Amount of energy dissipated by the junction during the last time step.

        The junction does not dissipate energy.

        Returns
        -------
        0
        """
        return 0

    def get_maximal_dt(self):
        return np.infty


class TemporalJunctionDiscontinuity(TemporalComponent):
    """
    Junction of two Pipes with acoustic mass related to cross-section
    discontinuity for time-domain simulation.

    Parameters
    ----------
    junct : :py:class:`JunctionDiscontinuity<openwind.continuous.junction.JunctionDiscontinuity>`
        The continuous version of the junction

    ends : list of 2 :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`
        The two pipe end objects the junction connects

    t_solver : :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance

    Attributes
    ----------
    m_disc : float
        The acoustic mass due to the cross section discontinuity at the junction
    dt: float
        the time step
    Step: np.array
        Matrix used to update gamma by half a step
    Infl: np.array
        Influence of each pipe's pressure and velocity on the update of gamma
    """

    def __init__(self, junct, ends, t_solver):
        super().__init__(junct, t_solver)
        self._end1, self._end2 = ends

        r_1, rho = self._end1.get_physical_params()[0:2]
        r_2 = self._end2.get_physical_params()[0]
        self.m_disc = junct.compute_mass(r_1, r_2, rho)

        self.reset_variables()
        self._should_recompute_constants = True

    def set_dt(self, dt):
        self.dt = dt
        self._should_recompute_constants = True

    def reset_variables(self):
        """
        Reinitialize all variables to start the simulation over.
        """
        self._gamma = np.zeros(1)
        self._P_corr = np.zeros(2)

    def _precompute_constants(self):
        """Compute constant matrices used in the update"""
        m_end1 = self.dt / (2 * self._end1.get_alpha())
        m_end2 = self.dt / (2 * self._end2.get_alpha())

        M_J = np.array([[self.m_disc]])
        invM_end = diags([1/m_end1, 1/m_end2])
        T_J = np.array([[-1, 1]])

        # Matrix of self-contribution of gamma trough the evolution of pressure
        A_g = T_J @ invM_end @ T_J.T
        # Matrix M_{J,dt}, to invert for performing a half-timestep
        M_Jdt = M_J + self.dt**2/4 * A_g
        invM_Jdt = np.array(inv(M_Jdt))
        # Matrix used to update gamma by half a step
        self.Step = invM_Jdt @ M_J
        # Influence of each pipe's pressure and velocity on the update of gamma
        self.Infl = -self.dt/2 * invM_Jdt @ T_J

        self._should_recompute_constants = False


    def _read_end_values(self):
        # Predicted value of P at time n+1/2
        self._P_corr[0] = self._end1.get_p_no_flow()
        self._P_corr[1] = self._end2.get_p_no_flow()

    def _update_flows(self, gamma_b):
        self._end1.update_flow(gamma_b[0])
        self._end2.update_flow(-gamma_b[0])

    def one_step(self):
        """Advance one time step

        Computes the evolution of internal variable ``gamma``,
        and updates the flow on the two connected
        :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`.
        """
        if self._should_recompute_constants:
            self._precompute_constants()

        self._read_end_values()
        gamma = self._gamma
        gamma_b = self.Step @ gamma + self.Infl @ self._P_corr
        self._gamma = 2*gamma_b - gamma
        self._update_flows(gamma_b)

    def __str__(self):
        name = "TJunctionDiscontinuity({}, {})"
        return name.format(self._end1, self._end2)
    def __repr__(self):
        return self.__str__()

    def energy(self):
        """
        Compute the amount of energy stored in the junction at time step

        .. math::
            \mathcal{E} = \\frac{1}{2} m_{disc} \\gamma^2

        Returns
        -------
        float : the stored energy

        """
        energy  = self.m_disc/2 * self._gamma[0]**2
        return energy

    def dissipated_last_step(self):
        return 0

    def get_maximal_dt(self):
        return np.infty


class TemporalJunctionSwitch(TemporalComponent):
    """
    Switch Junction between 3 Pipes with acoustic mass related to cross-section
    discontinuity for time-domain simulation.

    Parameters
    ----------
    junct : :py:class:`JunctionSwitch<openwind.continuous.junction.JunctionSwitch>`
        The continuous version of the junction

    ends : list of 3 :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`
        The three pipe end objects the junction connects in the order

        - 1 the always connected end
        - 2 the end connected when the switch is "open"
        - 3 the end connected when the switch is "closed"

    t_solver : :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The solver instance

    Attributes
    ----------
    M_J : 2D array
        The  mass matrix related to the acoustic masses due to cross section
        discontinuity at the junction
    T_J : 2D array
        Transmission matrix.
    dt: float
        the time step
    """

    def __init__(self, junct, ends, t_solver):
        super().__init__(junct, t_solver)
        self._end1, self._end2, self._end3 = ends
        self._junct = junct
        self._opening_factor = 1.0
        self.update_coefficients()
        self.reset_variables()
        self._should_recompute_constants = True

    def set_dt(self, dt):
        self.dt = dt
        self._should_recompute_constants = True

    def reset_variables(self):
        """
        Reinitialize all variables to start the simulation over.
        """
        self._gamma = np.zeros(1)
        self._P_corr = np.zeros(3)

    def update_coefficients(self):
        """
        Update junction coefficients when the opening factor is modified.

        Update :py:attr:`M_J` and :py:attr:`T_J`.
        """
        r1, rho = self._end1.get_physical_params()[0:2]
        r2 = self._end2.get_physical_params()[0]
        r3 = self._end3.get_physical_params()[0]
        self.M_J, self.T_J = self._junct.compute_masses(r1, r2, r3, rho,
                                                        self._opening_factor)
        if self.M_J != 0:
            raise ValueError('The brass valve junction does not accept yet '
                              'discontinuity mass in the temporal domain.'
                              ' Please set the option "discontinuity_mass" to False.')
        self._should_recompute_constants = True

    def set_opening_factor(self, opening_factor):
        """Control opening and closing of the switch.

        Can be used during simulation. Energy-safe.

        Parameters
        ----------
        opening_factor : float
            1 for a raised valve (open switch) connect pipe 1 to 2,
            0 for a depressed valve  (closed switch) connect pipe 1 to 3,
            or any value in-between for a partly-closed switch
        """
        # Only update the factor if it changed significantly : too expensive (5,92 sec pour set_op contre 4,44)
        if not (abs(opening_factor - self._opening_factor) < 1e-5):
            self._opening_factor = opening_factor
            self.update_coefficients()

    def _precompute_constants(self):
        """Compute constant matrices used in the update"""
        m_end1 = self.dt / (2 * self._end1.get_alpha())
        m_end2 = self.dt / (2 * self._end2.get_alpha())
        m_end3 = self.dt / (2 * self._end3.get_alpha())
        invM_end = diags([1/m_end1, 1/m_end2, 1/m_end3])

        # Matrix of self-contribution of gamma trough the evolution of pressure
        A_g = self.T_J @ invM_end @ self.T_J.T
        # Matrix M_{J,dt}, to invert for performing a half-timestep
        M_Jdt = self.M_J + self.dt**2/4 * A_g
        invM_Jdt = np.array(inv(M_Jdt))
        # Matrix used to update gamma by half a step
        self.Step = invM_Jdt @ self.M_J
        # Influence of each pipe's pressure and velocity on the update of gamma
        self.Infl = -self.dt/2 * invM_Jdt @ self.T_J
        self._should_recompute_constants = False

    def _read_end_values(self):
        # Predicted value of P at time n+1/2
        self._P_corr[0] = self._end1.get_p_no_flow()
        self._P_corr[1] = self._end2.get_p_no_flow()
        self._P_corr[2] = self._end3.get_p_no_flow()

    def _update_flows(self, gamma_b):
        u = -self.T_J*gamma_b[0] #flow convention
        self._end1.update_flow(u[0, 0])
        self._end2.update_flow(u[0, 1])
        self._end3.update_flow(u[0, 2])

    def one_step(self):
        """Advance one time step

        Computes the evolution of internal variable ``gamma``,
        and updates the flow on the three connected
        :py:class:`TemporalPipeEnd<openwind.temporal.tpipe.TemporalPipeEnd>`.
        """
        if self._should_recompute_constants:
            self._precompute_constants()

        self._read_end_values()
        gamma = self._gamma
        gamma_b = self.Step @ gamma + self.Infl @ self._P_corr
        self._gamma = 2*gamma_b - gamma
        self._update_flows(gamma_b)

    def __str__(self):
        name = "TJunctionSwitch({}, {})"
        return name.format(self._end1, self._end2)

    def __repr__(self):
        return self.__str__()

    def energy(self):
        """
        Compute the amount of energy stored in the junction at time step

        Returns
        -------
        float : the stored energy

        """
        energy  = self.M_J[0]/2 * self._gamma[0]**2
        return energy

    def dissipated_last_step(self):
        return 0

    def get_maximal_dt(self):
        return np.infty
