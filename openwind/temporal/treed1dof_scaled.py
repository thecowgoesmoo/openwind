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
class TemporalReed1dofScaled(TemporalComponentExit):
    """Simulate the interaction with reed (lips or cane) at the end of the pipe.

    We use the one-degree-of-freedom model from [Bil09.]_. The scheme used is a
    theta-scheme. More details are available here [Scheme_Report]_.

    References
    ----------
    .. [Bil09.] Bilbao, S. (2009). Direct simulation of reed wind instruments.
       Computer Music Journal, 33(4), 43-55.
    .. [Scheme_Report] https://files.inria.fr/openwind/supp_files/scheme_reed_dimless_contact.pdf

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
    contact_quadratization_cst : float
        Constant used in the quadratization of the contact force.

    """

    def __init__(self, reed1dof, pipe_ends, t_solver, theta, contact_quadratization_cst):
        super().__init__(reed1dof, t_solver)
        self.pipe_end, = pipe_ends
        self.reed1dof = reed1dof
        self.theta = theta
        self.contact_quad_cst = contact_quadratization_cst
        print('Using Theta scheme, theta = ' + str(self.theta))

    def set_dt(self, dt):
        self.dt = dt
        self.reset_variables()
        self._precompute_constants()

    def _precompute_constants(self):
        # nothing yet
        radius, rho, c = self.pipe_end.get_physical_params()
        self.Zc = rho*c/(np.pi*radius**2)
        self.rho = rho

    def one_step(self):
        """Advance one time step.

        Computes the evolution of internal variables `y`, and updates the flux
        on the connected :py:class:`TemporalPipeEnd \
        <openwind.temporal.tpipe.TemporalPipeEnd>`

        It uses a theta-scheme.
        """

        theta = self.theta
        gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon = self.reed1dof.get_dimensionless_values(self.get_current_time(), self.Zc, self.rho)
        Pclosed = self.reed1dof.get_Pclosed(self.get_current_time()) # necessary to rescale the quantities from/toward the pipe
        dt_scaled = omegar*self.dt

        last_y = self._last_y  # y^{n-1}
        this_y = self._this_y  # y^{n}
        prev_z = self._prev_z  # z^{n-\half}

        # scaling coef to communicate with the pipe: rescaling from the pipe and scaling for this source
        p_scaling = Pclosed/self.reed1dof.scaling.get_scaling_pressure()
        flow_scaling = Pclosed/self.Zc/self.reed1dof.scaling.get_scaling_flow()

        # coef from the pipe
        p_no_flow = self.pipe_end.get_p_no_flow()/p_scaling # the pressure from the pipe needs to be rescaled
        A_lambda = self.Zc / (self.pipe_end.get_alpha() * self.reed1dof.scaling.get_impedance()) # alpha = dt /(2* m_end)

        # coef for y
        # arbitrary constant
        yn_minus = _abs_negative_part(this_y)
        # Quadratization of the conact force
        if self.contact_quad_cst==0:
            H_yn = -.5*np.sqrt(2*Kc*(alpha_c+1))*yn_minus**(alpha_c/2-.5)# for C=0:
        else:
            H_yn = -Kc*yn_minus**alpha_c / np.sqrt(self.contact_quad_cst + 2*Kc/(alpha_c+1)*yn_minus**(alpha_c+1))
            # H_yn = -.5*np.sqrt(2*Kc*(alpha_c+1))*yn_minus**alpha_c / np.sqrt(self.contact_quad_cst + yn_minus**(alpha_c+1))

        A_yn = 1/(1/dt_scaled**2 + 1/(2*Qr*dt_scaled) + theta + .25*H_yn**2)

        B1_y = 2/dt_scaled**2 + 2*theta - 1
        B2_y = -1*(2/dt_scaled**2 + 2*theta)
        B3_y = 1 - H_yn*prev_z
        B_yn = A_yn*(B1_y*this_y +B2_y*last_y + B3_y)

        # coef for Delta_p
        an = A_lambda + A_yn*kappa/(2*dt_scaled)
        bn = _positive_part(this_y)*zeta
        cn = A_lambda*(p_no_flow - gamma) + epsilon*B_yn*kappa/(2*dt_scaled)

        # Compute Delta_p
        discr = bn**2 + 4*an*np.abs(cn)
        root_poly = (-bn + np.sqrt(discr))/(2*an)
        Delta_p = -np.sign(cn)*root_poly**2

        # next step:
        Delta_y = B_yn + epsilon*A_yn*Delta_p
        next_y = Delta_y + last_y
        next_z = .5*H_yn*Delta_y + prev_z
        flow_lambda = A_lambda*(Delta_p + p_no_flow - gamma)

        # update values
        self.pipe_end.update_flow(flow_lambda*flow_scaling) # in the pipe, the equations can be scaled differently
        self._this_y = next_y
        self._last_y = this_y
        self._last_last_y = last_y
        self._prev_z = next_z
        self._last_Delta_P = Delta_p
        self._last_lambda = flow_lambda
        self._last_gamma = gamma
        # self.t += self.dt
        super().remember_flow_and_pressure(self.pipe_end)  # For recording
        super().remember_y(this_y)



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
        self.y = 1 #self.reed1dof.opening.get_value(0)

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
        gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon = self.reed1dof.get_dimensionless_values(self.get_current_time(), self.Zc, self.rho)
        Pclosed = self.reed1dof.get_Pclosed(self.get_current_time())
        dt_scaled = self.dt*omegar
        mu_y = (self._this_y + self._last_y)/2
        delta_y = (self._this_y - self._last_y)/dt_scaled

        spring_energy = .5*(mu_y -1)**2 + .5*self._prev_z**2
        kinetic_energy = .5*(1 + dt_scaled**2*(self.theta - 0.25))*delta_y**2

        # rescaling
        scaling_pipe_power = self.reed1dof.scaling.get_scaling_power()
        scaling_coef =  Pclosed**2*kappa/(omegar *self.Zc) / scaling_pipe_power
        return (spring_energy + kinetic_energy)*scaling_coef


    def dissipated_last_step(self):
        gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon = self.reed1dof.get_dimensionless_values(self.get_current_time(), self.Zc, self.rho)
        Pclosed = self.reed1dof.get_Pclosed(self.get_current_time())

        dt_scaled = omegar*self.dt
        delta_mu_y = (self._this_y - self._last_last_y)/(2 * dt_scaled)

        spring_damping = kappa/Qr*delta_mu_y**2
        bernoulli_dissip = zeta * _positive_part(self._last_y) * abs(self._last_Delta_P)**(3/2)
        NL_dissip = 0
        # we add the term "dissipated" by the mouth pressure (-1*the energy furnished)
        forcing_term = +1* self._last_gamma * self._last_lambda #

        scaling_pipe_power = self.reed1dof.scaling.get_scaling_power()

        return (spring_damping + bernoulli_dissip + NL_dissip + forcing_term) * self.dt*Pclosed**2/self.Zc / scaling_pipe_power
