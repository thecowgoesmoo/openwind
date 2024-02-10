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

"""Numerical theta-scheme for a 1-DOF reed model."""

from math import sqrt
import numpy as np

from openwind.temporal import TemporalComponentExit

def _positive_part(y):
    return y * (y > 0)

def _negative_part(y):
    return y * (y < 0)

def _abs_negative_part(y):
    return -y * (y < 0)

#NUMERICAL THETA SCHEME
class TemporalReed1dof(TemporalComponentExit):
    """Simulate the interaction with reed (lips or cane) at the end of the pipe.

    We use the one-degree-of-freedom model from [Bil09]_.
    The scheme used is a theta-scheme.
    TODO Add documentation for the scheme.

    References
    ----------
    .. [Bil09] Bilbao, S. (2009). Direct simulation of reed wind instruments.
       Computer Music Journal, 33(4), 43-55.

    Parameters
    ----------
    reed1dof : :py:class:`Reed1dof<openwind.continuous.excitator.Reed1dof>`
        Continuous model with parameters
    pipe_ends : tuple of 1 :py:class:`TemporalPipeEnd \
    <openwind.temporal.tpipe.TemporalPipeEnd>`
        The tuple of the single pipe end this reed is connected to
    t_solver : :py:class:`TemporalSolver\
        <openwind.temporal.temporal_solver.TemporalSolver>`
        The TemporalSolver this object is a part of
    theta : float
        Parameter of the theta-scheme

    """

    def __init__(self, reed1dof, pipe_ends, t_solver, theta):
        super().__init__(reed1dof, t_solver)
        self.pipe_end, = pipe_ends
        self.reed1dof = reed1dof
        self.theta = theta
        print('Using Theta scheme, theta = ' + str(self.theta))

    def set_dt(self, dt):
        self.dt = dt
        self.reset_variables()
        self._precompute_constants()

    def _precompute_constants(self):
        radius, rho, c = self.pipe_end.get_physical_params()
        self.Zc = rho*c/(np.pi*radius**2)
        self.rho = rho

    def one_step(self):
        """Advance one time step.

        Computes the evolution of internal variables `y`, and updates the flux
        on the connected :py:class:`TemporalPipeEnd \
        <openwind.temporal.tpipe.TemporalPipeEnd>`

        It uses a thea-scheme.
        """

        rho = self.rho
        p_no_flow = self.pipe_end.get_p_no_flow() * self.reed1dof.scaling.get_scaling_pressure()

        dt = self.dt
        theta = self.theta
        m_end_tilde = -1 / self.pipe_end.get_alpha() / self.reed1dof.scaling.get_impedance() # 2/ delta t * m_end

        last_y = self._last_y  # y^{n-1}
        this_y = self._this_y  # y^{n}
        prev_z = self._prev_z  # z^{n-\half}

        # get control parameters at the time t
        (Sr, Mr, g, omega02, w, y0, epsilon, p_m, omega_nl, alpha_nl) = self.reed1dof.get_dimensionfull_values(self.get_current_time(), self.Zc, self.rho)
        nl_exp = alpha_nl + 1 # the scheme has been implemented with another contact force: K*abs(y-)^(nl_exp-1)
        # Compute constantes :
        ynminus = _abs_negative_part(this_y)
        Gprime = -1*nl_exp*0.5 * ynminus**(nl_exp/2-1) * np.sqrt(2*omega_nl**nl_exp
                                                              / (nl_exp*y0**(nl_exp-2)))


        NL_fact = Gprime**2/(4)
        AB_fact = 1/(1/dt**2  + g/(2*dt) + omega02*theta + NL_fact)
        B1 = (2/dt**2 - omega02*(1-2*theta))
        B2 = (-2/dt**2 - 2*omega02*theta)
        B3 = omega02*y0 - prev_z*Gprime
        A = AB_fact * (epsilon*Sr/Mr)
        B = AB_fact * (B1*this_y + B2*last_y + B3)
        C = -m_end_tilde
        D = m_end_tilde * (p_m - p_no_flow)

        # Compute DELTA P
        alpha = C + epsilon * (Sr/(2*dt) * A)
        beta = D + epsilon * (Sr/(2*dt) * B)
        gamma = w * _positive_part(this_y) * np.sqrt(2/rho)
        discr = gamma**2 + 4 * alpha * abs(beta)
        root = (-gamma + sqrt(discr)) / (2*alpha)
        Delta_P = -np.sign(beta) * root**2

        # Compute LAMBDA
        flow_lambda = C * Delta_P + D

        # Compute y^(n+1)
        temp = A * Delta_P + B
        next_y = temp + last_y
        next_z = prev_z + Gprime*(next_y-last_y)*0.5

        # update values
        self.pipe_end.update_flow(flow_lambda / self.reed1dof.scaling.get_scaling_flow())
        self._this_y = next_y
        self._last_y = this_y
        self._last_last_y = last_y
        self._prev_z = next_z
        self._last_Delta_P = Delta_P
        self._last_lambda = flow_lambda
        self._last_pm = p_m
        # self.t += self.dt
        super().remember_flow_and_pressure(self.pipe_end)  # For recording
        super().remember_y(last_y)

    def get_maximal_dt(self):
        """This numerical scheme has a CFL condition."""
        theta = self.theta
        if theta >= 0.25:
            dt_max = np.inf
        else:
            # CFL is calculated from the value at t=0
            # TODO Which parameters are allowed to vary?
            omega02 = self.reed1dof.pulsation.get_value(0)**2
            dt_max = np.sqrt(np.abs(1 / ((0.25 - theta) * omega02)))
        return dt_max

    def reset_variables(self):
        # _y is y^{n+1/2}
        self.y = self.reed1dof.opening.get_value(0)

        self._last_last_y = self.y  # needed for energy dissipation
        self._last_y = self.y
        self._this_y = self.y

        # linearly imlpicit scheme for NL term
        # (it is assumed that y is far from contact at initial time)
        self._prev_z = 0.0 # z^{n+1}
        self._next_z = 0.0 # z^n

        # self.t = self.dt/2

    def __str__(self):
        return "TReed1dof({})".format(self.pipe_end)
    def __repr__(self):
        return self.__str__()

    def energy(self):
        """Compute the amount of energy stored in the reed."""

        (Sr, mr, g, omega02, w, y0, epsilon, p_m, omega_nl, nl_exp) = self.reed1dof.get_dimensionfull_values(self.get_current_time(), self.Zc, self.rho)

        z_n = (self._this_y - self._last_y)/self.dt
        y_bar = (self._this_y + self._last_y)/2

        spring_energy = mr/2 * omega02 * (y_bar - y0)**2 + 0.5*mr*self._prev_z**2

        kinetic_energy = mr/2 * (1 +
                                 self.dt**2 *
                                 (self.theta - 1/4) *
                                 omega02) * z_n**2
        return (spring_energy + kinetic_energy) / self.reed1dof.scaling.get_scaling_power()


    def dissipated_last_step(self):
        (Sr, mr, g, omega02, w, y0, epsilon, p_m, omega_nl, nl_exp) = self.reed1dof.get_dimensionfull_values(self.get_current_time(), self.Zc, self.rho)

        z_bar = (self._this_y - self._last_last_y)/(2 * self.dt)

        spring_damping = mr  * g * z_bar**2
        bernoulli_dissip = w * sqrt(2/self.rho) * _positive_part(self._last_y) * abs(self._last_Delta_P)**(3/2)
        NL_dissip = 0
        # we add the term coming from the mouth pressure
        forcing_term = self._last_pm * self._last_lambda
        return (spring_damping + bernoulli_dissip + NL_dissip + forcing_term) * self.dt  / self.reed1dof.scaling.get_scaling_power()
