#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2021, INRIA
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

"""Numerical theta-scheme for a flute model."""

import numpy as np
from warnings import warn
from openwind.temporal import TemporalComponentExit

class TemporalFlute(TemporalComponentExit):
    """
    Simulate the jet-edge interaction at the end of the pipe.

    We use an adaptation of the "jet-drive model" from [Auv14]_. The equations
    are scaled. More details on these equations an the scheme used are available
    here [Scheme_flute]_.

    References
    ----------
    .. [Auv14] R. Auvray, A. Ernoult, B. Fabre and P.-Y. Lagrée 2014.
        "Time-domain simulation of flute-like instruments: Comparison of
        jet-drive and discrete-vortex models". JASA 136(1), p.389–400.
        https://hal.science/hal-01426971/document.

    .. [Scheme_flute] https://files.inria.fr/openwind/supp_files/Scheme_flute_jet-drive.pdf

    Parameters
    ----------
    flute : :py:class:`Flute<openwind.continuous.excitator.Flute>`
        Continuous model with parameters
    pipe_ends : tuple of 1 :py:class:`TemporalPipeEnd \
    <openwind.temporal.tpipe.TemporalPipeEnd>`
        The tuple of the single pipe end this flute-like embouchure is connected to
    t_solver : :py:class:`TemporalSolver\
        <openwind.temporal.temporal_solver.TemporalSolver>`
        The TemporalSolver this object is a part of

    """

    def __init__(self, flute, pipe_ends, t_solver):
        super().__init__(flute, t_solver)
        self.pipe_end, = pipe_ends
        self.flute = flute
        self._opening_factor = 1.0
        radius_pipe = self.pipe_end.get_physical_params()[0]
        self.rad = self.flute.get_rad_model_window(np.pi*radius_pipe**2)

    def set_dt(self, dt):
        self.dt = dt
        self.reset_variables()
        self._precompute_constants()

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
        if opening_factor<1e-8:
            raise ValueError('The flute embouchure hole (window) can not bel fully closed.')
        # Only update the factor if it changed significantly : too expensive (5,92 sec pour set_op contre 4,44)
        if not (abs(opening_factor - self._opening_factor) < 1e-5):
        #    np.isclose(opening_factor, self._opening_factor, atol=1e-5):
            self._opening_factor = opening_factor
            old_Zplus = self.Zc
            self._precompute_constants()
            assert self.Zc == old_Zplus  # Zplus is not allowed to change!

    def _precompute_constants(self):
        # nothing yet
        r_pipe, rho, c = self.pipe_end.get_physical_params()
        self.radius = self.flute.get_equivalent_radius()
        self.Zc = rho*c/(np.pi*self.radius**2)
        self.rho = rho
        self.celerity = c

        self.jet_params = self.flute.get_scaled_model_parameters(self.celerity)
        self.rad_params = self.rad.compute_temporal_coefs(self.radius, self.rho,
                                                          self.celerity, self._opening_factor)[:2]

    def get_next_eta(self, next_tau):
        """
        Compute the jet position at the edge, at the future time step from past ac. velocity

        Parameters
        ----------
        next_tau : float
            Scaled time delay at the next time step.

        Raises
        ------
        ValueError
            Error if the delay is too short with respect to the time step.

        Returns
        -------
        next_eta : float
            The jet position at the last time step.

        """

        n = len(self.mem_lambda)#int(self.t/self.dt) # current step
        N = int(np.floor(next_tau/self.dt))
        epsilon = next_tau/self.dt - N
        if N<2:
            raise ValueError('the step time is too large, the jet delay should'
                             f'correspond to at least 2 step: tau/dt={next_tau/self.dt} not > 2.'
                             ' Otherwise it becomes an implicit problem.')
        # deal with negative time
        nN = max(0, n-N)
        nNp1 = max(0, n-N+1) # if N<2 => Np1>=n, which is impossible: mem_lambda[n] does not exists

        next_eta = (1-epsilon)*self.mem_lambda[nNp1] + epsilon*self.mem_lambda[nN]
        next_eta += self.flute.noise_level.get_value(self.get_current_time())*np.random.randn()

        return next_eta


    def one_step(self):
        """Advance one time step.

        Computes the evolution of internal variables `eta`, and updates the flux
        on the connected :py:class:`TemporalPipeEnd \
        <openwind.temporal.tpipe.TemporalPipeEnd>`

        """
        prev_zeta = self._prev_zeta  # zeta^{n-\half} internal variables related to the radiation
        this_eta = self._this_eta
        last_eta = self._last_eta

        Ascale, Gj, Gl, C, y0 = self.jet_params
        alpha_r, beta_r = self.rad_params

        # get speed of jet and its time derivative
        Uj, dUj, next_Uj = self.flute.get_Uj_dUj(self.get_current_time(), self.dt)
        if Uj<1e-8:
            raise ValueError('The jet speed must be strictly positive! '
                             f'Here Uj({self.get_current_time():.3f})={Uj:.2e}!')
        if next_Uj<1e-8: # to avoid issue at the last time step if the consign is zero outside the simulation duration (ex: ramp)
            next_Uj = Uj

        # scalinf coef for pressure and flow from/to pipe
        p_scaling = (self.rho*self.celerity*Uj)/Ascale /self.flute.scaling.get_scaling_pressure()
        u_scaling =  p_scaling/(self.Zc/self.flute.scaling.get_impedance()) # (Uj*np.pi*self.radius**2)/Ascale  /self.flute.scaling.get_scaling_flow() #

        # get the delayed non-lin source term and its derivative
        next_tau = C/(next_Uj) / self.flute.scaling.get_time() # time delay
        next_eta = self.get_next_eta(next_tau) # jet position at the edge tip
        Gamma = np.tanh(this_eta - y0)
        dGamma = (np.tanh(next_eta - y0) - np.tanh(last_eta - y0))/(2*self.dt)

        # =============================================================================
        # Compute flow
        # =============================================================================
        p_no_flow = self.pipe_end.get_p_no_flow()/p_scaling # the pressure from the pipe needs to be rescaled
        Ap = -(self.pipe_end.get_alpha() * self.flute.scaling.get_impedance())/self.Zc  # alpha = dt /(2* m_end)
        Ar = 1/(beta_r + alpha_r*self.dt/2)

        Br = np.sqrt(alpha_r)*Ar*prev_zeta
        Bp = p_no_flow
        Btot = Br - Bp

        source = Gj*(Gamma*dUj/Uj + dGamma)
        alpha = Gl*Uj
        beta = Ar - Ap
        gamma = Btot - source

        if alpha>1e-8 or alpha*gamma/beta**2>1e-6:
            flow = -1*np.sign(gamma)*(-beta + np.sqrt(beta**2 + 4*alpha*np.abs(gamma)))/(2*alpha)
        else: # for low value of alpha use Taylor dev instead =>
            flow = -gamma/beta  + np.sign(gamma)*alpha*gamma**2/beta**3
        # flow = -gamma/beta # without non-linear losses

        next_zeta = 2*Ar*(beta_r*prev_zeta - self.dt/2*np.sqrt(alpha_r)*flow) - prev_zeta
        # next_zeta = Ar*( (beta_r - self.dt/2*alpha_r)*prev_zeta  - self.dt*np.sqrt(alpha_r)*flow)

        # update the flow in the pipe
        self.pipe_end.update_flow(flow*u_scaling) # in the pipe, the equations are scaled differently

        # =============================================================================
        # Check
        # =============================================================================
        # delta_zeta = (next_zeta - prev_zeta)/self.dt
        # mu_zeta = (next_zeta + prev_zeta)/2
        # mu_p_r = Ar*flow + Br
        # mu_p_p = Ap*flow + Bp

        # err_45a = (delta_zeta + np.sqrt(alpha_r)*mu_p_r)/delta_zeta
        # err_45b = (np.sqrt(alpha_r)*mu_zeta - beta_r*mu_p_r + flow)/flow

        # err_46a = (mu_zeta - prev_zeta + np.sqrt(alpha_r)*self.dt/2*mu_p_r)/mu_zeta
        # err_46b = (flow + np.sqrt(alpha_r)*prev_zeta - (beta_r + alpha_r*self.dt/2)*mu_p_r)/flow

        # err_delta_p = (mu_p_r - mu_p_p) - source
        # p_pipe_end = self.pipe_end.get_q_nph()/p_scaling
        # err_p_pipe  = p_pipe_end/mu_p_p - 1

        # =============================================================================
        # next step
        # =============================================================================
        super().remember_flow_and_pressure(self.pipe_end)  # For recording
        super().remember_y(this_eta)

        self._prev_zeta = next_zeta
        self.mem_lambda.append(flow)
        self._this_eta = next_eta
        self._last_eta = this_eta


    def get_maximal_dt(self):
        """This numerical scheme has no CFL condition???."""
        dt_max = np.inf
       # TODO
       # calculer le dt max avec le max de Uj??
        return dt_max

    def reset_variables(self):
        # Internal variable
        self._prev_zeta=0.0
        # Initialisation of flow/jet-position
        self.mem_lambda = [0.0]
        self._this_eta = 0
        self._last_eta = 0

    def __str__(self):
        return "TFlute({})".format(self.pipe_end)
    def __repr__(self):
        return self.__str__()

    def energy(self):
        """In its current form, we are not able to get an energy associated to this scheme."""
        raise NotImplementedError("The energy of the flute scheme has not been computed.")


    def dissipated_last_step(self):
        """In its current form, we are not able to get an energy associated to this scheme."""
        raise NotImplementedError("The energy of the flute scheme has not been computed.")
