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
Numerical scheme for a Pipe with losses.

The numerical sche
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.csr import csr_matrix
# Low-level solution to avoid overhead every time we do a matrix-vector multiplication
from scipy.sparse._sparsetools import csr_matvec

from openwind.continuous import ParametricRoughness
from openwind.temporal import TemporalPipe


class TemporalRoughPipe(TemporalPipe):
    """
    A temporal pipe with viscothermal losses based on the "parametric roughness" model.

    It is a modified version of :py:class:`TemporalPipe\
    <openwind.temporal.tpipe.TemporalPipe>`.

    The loss model must be 'parametric_roughness'.
    """


    def __init__(self, pipe, t_solver, **params):
        super().__init__(pipe, t_solver, **params)
        assert isinstance(pipe.get_losses(), ParametricRoughness)

    def _precompute_losses(self):
        """Precomputes coefficients used for the update with losses."""
        # losses = self.pipe.get_losses()
        # omega1, omega2 = losses.get_omega12_at(self.pipe, 0)
        # zeta0, zeta1, zeta2, zeta3 = losses.get_zetas_at(self.pipe, 0)
        # nu0, nu1, nu2, nu3 = losses.get_nus_at(self.pipe, 0)
        # r0 = zeta0
        # print("Running optimization...")
        # print("zeta0, zeta1, zeta2, zeta3 =", (zeta0, zeta1, zeta2, zeta3))
        # print("nu0, nu1, nu2, nu3 =", (nu0, nu1, nu2, nu3))
        # ri, li = optimize_RiLi_roughness(omega1, omega2,
        #                                  zeta0, zeta1, zeta2, zeta3,
        #                                  OMEGA_MIN, OMEGA_MAX,
        #                                  N = losses.N_diffrepr_var)
        # g0 = nu0
        # gi, ci = optimize_RiLi_roughness(omega1, omega2,
        #                                  nu0, nu1, nu2, nu3,
        #                                  OMEGA_MIN, OMEGA_MAX,
        #                                  N = losses.N_diffrepr_var)

        # self._diffrepr_coefs = (r0, ri, li), (g0, gi, ci)
        losses = self.pipe.get_losses()
        # DEBUG use this scheme to simulate a lossless pipe
        if losses.N_diffrepr_var == 0:
            empty = np.zeros((0,1))
            ri = empty
            self._diffrepr_coefs = (0, empty, empty), (0, empty, empty)
        else:
            (r0, ri, li), (g0, gi, ci) = losses.calc_diffrepr_coefs(self.pipe)
            # Apply weighting due to the mesh
            weightsL2 = self.mesh.get_weights()
            weightsH1 = self.mesh.assemble_H1_from_L2(self.mesh.get_weights())
            self._diffrepr_coefs = ((r0 * weightsL2, ri * weightsL2, li * weightsL2),
                                    (g0 * weightsH1, gi * weightsH1, ci * weightsH1))
        # print("diffrepr_coefs =", self._diffrepr_coefs)
        print("ri.shape =", (ri * weightsL2).shape)
        print("gi.shape =", (gi * weightsH1).shape)
        self._loss_N = ri.shape[0]

    def reset_variables(self):
        super().reset_variables()
        self._Pi = np.zeros((self._loss_N, self.nH1))
        self._Vi = np.zeros((self._loss_N, self.nL2))
        self._V_prev = np.zeros(self.nL2)
        self._Vi_prev = np.zeros_like(self._Vi)
        self._next_p_no_flow = np.zeros(self.nH1)

    def _precompute_matrices(self):
        self._precompute_losses()
        dt = self._dt
        self.MH1inv = diags(1/self.mH1)
        self.ML2inv = diags(1/self.mL2)
        self.dtinvMBt = dt * self.MH1inv @ self.Bh.T
        self.dtinvML2B = dt * self.ML2inv @ self.Bh
        self.energy_matrix = diags(self.mH1) @ (np.eye(self.nH1) - 1/4 * self.dtinvMBt @ self.dtinvML2B)

        (r0, ri, li), (g0, gi, ci) = self._diffrepr_coefs
        _, ri_, _, gi_ = r0/2, ri/2, g0/2, gi/2
        _, _, li_, ci_ = self.mH1/dt, self.mL2/dt, li/dt, ci/dt
        ritilde = (li * ri) / (li + dt/2 * ri)
        gitilde = (ci * gi) / (ci + dt/2 * gi)
        rtilde = r0 + np.sum(ritilde, axis=0)
        gtilde = g0 + np.sum(gitilde, axis=0)

        denom_P = self.mH1 + dt/2 * gtilde
        self.p_to_p_noflow = self.mH1 / denom_P
        self.pi_to_p_noflow = dt * gitilde / (2 * denom_P)
        self.v_to_p_noflow = - dt/2 * diags(1/denom_P) @ self.Bh.T
        assert isinstance(self.v_to_p_noflow, csr_matrix)

        self.end_minus.set_alpha(dt/(2 * denom_P[0]))
        self.end_plus.set_alpha(dt/(2 * denom_P[-1]))

        self.p_to_pi = gi_ / (ci_ + gi_)
        self.pi_to_pi = (ci_ - gi_) / (ci_ + gi_)

        # Coefficients for computing V
        denom_V = self.mL2 + dt/2 * rtilde
        self.v_to_v = (self.mL2 - dt/2 * rtilde) / denom_V
        self.vi_to_v = dt * ritilde / denom_V
        self.p_to_v = dt * diags(1/denom_V) @ self.Bh
        assert isinstance(self.p_to_v, csr_matrix)
        self.v_to_vi = ri_ / (li_ + ri_)
        self.vi_to_vi = (li_ - ri_) / (li_ + ri_)

    def __str__(self):
        return "TemporalRoughPipe{}".format(self.label)

    def get_p_no_flow(self, pos=...):
        """
        Compute the pressure value if their was "no flow".

        See [Thibault_losses]_ for more details.

        The actual pressure at time n+1/2 will be

        .. math::
            q^{n+1/2} = p_{\\text{no flow}} - \\alpha  w^{n+1/2}.

        Parameters
        ----------
        pos: int
            The index of the dof at which the pressure must be computed

        Returns
        -------
        float
            The "no flow" pressure at the specified index
        """
        return self._next_p_no_flow[pos]

    def one_step(self, check_scheme=True):
        """Advance one time step.

        Assumes the flux of both pipe ends have already been updated.

        See [Thibault_losses]_ for more details.

        Raises
        ------
        AssertionError
            if either of the fluxes of the pipe ends has not been updated since
            the last time step
        """
        P, V = self.PV
        Pi = self._Pi
        Vi = self._Vi
        flow_left = self.end_minus.get_w_nph() # Boundary effects
        flow_right = self.end_plus.get_w_nph()

        # Update of P
        P_nph = self.get_p_no_flow().copy()
        P_nph[0] = self.end_minus.accept_q_nph()
        P_nph[-1] = self.end_plus.accept_q_nph()

        # Limit dynamic memory allocation by using in-place operations

        P_next = 2*P_nph
        P_next -= P

        Pi_next = self.p_to_pi * (P + P_next)
        Pi_next += self.pi_to_pi * Pi

        # Update of V
        V_next = np.add.reduce(self.vi_to_v * Vi, axis=0)
        # V_next += self.p_to_v * P_next # /!\ Sparse matrix - vector multiplication
        csr_matvec(self.nL2, self.nH1, self.p_to_v.indptr,
                   self.p_to_v.indices,
                   self.p_to_v.data, P_next,
                   V_next)
        V_next += self.v_to_v * V

        # Vi_next = self.v_to_vi * (V + V_next) + self.vi_to_vi * Vi
        Vi_next = self.v_to_vi * (V + V_next)
        Vi_next += self.vi_to_vi * Vi

        # if check_scheme:
        #     dt = self._dt
        #     (r0, ri, li), (g0, gi, ci) = self._diffrepr_coefs
        #     # Check that the scheme is properly implemented
        #     eq1 = self.mL2 * (V_next - V)/dt + r0 * (V_next + V)/2 + np.sum(ri * (V_next + V - Vi_next - Vi)/2, axis=0) - self.Bh @ P_next
        #     eq2 = self.mH1 * (P_next - P)/dt + g0 * (P_next + P)/2 + np.sum(gi * (P_next + P - Pi_next - Pi)/2, axis=0) + self.Bh.T @ V
        #     eq2[0] += flow_left
        #     eq2[-1] += flow_right
        #     eq3 = li * (Vi_next - Vi)/dt + ri * (Vi_next + Vi - V_next - V)/2
        #     eq5 = ci * (Pi_next - Pi)/dt + gi * (Pi_next + Pi - P_next - P)/2

        #     print("Rel. error on dtV :",np.sum(np.abs(eq1)) / np.sum(np.abs(diags(self.mL2) * (V_next - V)/dt)))
        #     print("Rel. error on dtP :",np.sum(np.abs(eq2)) / np.sum(np.abs(self.mH1 * (P_next - P)/dt)))
        #     print("Rel. error on dtVi :",np.sum(np.abs(eq3)) / np.sum(np.abs(li * (Vi_next - Vi)/dt)))
        #     print("Rel. error on dtPi :",np.sum(np.abs(eq5)) / np.sum(np.abs(ci * (Pi_next - Pi)/dt)))

        # Put the new variables in the right place
        self.PV = (P_next, V_next)
        # Save a few steps (only useful for energy/power checks)
        self._V_prevprev, self._Vi_prevprev = self._V_prev, self._Vi_prev
        self._P_prev, self._Pi_prev = P, Pi
        self._V_prev, self._Vi_prev = V, Vi
        # and update the scheme
        self._Pi, self._Vi = Pi_next, Vi_next

        # Prepare the next step of P
        self._compute_next_p_no_flow()

    def _compute_next_p_no_flow(self):
        """
        Compute p_{no flow}^{n+1/2} from P^n and V^{n+1/2} assuming no
        boundary flow.
        """
        P, V = self.PV
        # self._next_p_no_flow = (self.p_to_p_noflow * P +
        #             np.add.reduce(self.pi_to_p_noflow * self._Pi, axis=0)
        #             + self.v_to_p_noflow * V) # /!\ Sparse matrix - vector multiplication

        self._next_p_no_flow = (self.p_to_p_noflow * P +
                    np.add.reduce(self.pi_to_p_noflow * self._Pi, axis=0))

        # Low-level call for efficient sparse matrix-vector multiplication
        # csr_matmul(M, N, indptr, indices, data, other, result)
        # computes Y += A*X
        assert self.v_to_p_noflow.shape == (self.nH1, self.nL2)
        csr_matvec(self.nH1, self.nL2, self.v_to_p_noflow.indptr,
                   self.v_to_p_noflow.indices,
                   self.v_to_p_noflow.data, V,
                   self._next_p_no_flow)

    def energy(self):
        """
        Compute the amount of energy stored in the pipe.

        The energy computed for the scheme is guaranteed
        to remain positive and to be nonincreasing once source terms are removed,
        which are the same as for the lossless pipe.

        See [Thibault_losses]_ for more details

        Returns
        -------
        float
            The stored energy
        """
        energy_P = self.energy_P()
        energy_V = self.energy_V()
        energy_Pi = self.energy_Pi()
        energy_Vi = self.energy_Vi()
        energy_Delta = self.energy_Delta()
        energies = [energy_P, energy_V, energy_Pi, energy_Vi, energy_Delta]
        assert all([e >= 0 for e in energies])
        return sum(energies)

    def energy_Pi(self):
        """
        Returns
        -------
        float
            The amount of energy stored in the auxiliary variables Pi
        """
        _, (_, _, ci) = self._diffrepr_coefs
        Pi = self._Pi
        return 1/2 * np.sum(Pi * ci * Pi)

    def energy_Vi(self):
        """
        Returns
        -------
        float
            The amount of energy stored in the auxiliary variables Vi
        """
        (_, _, li), _ = self._diffrepr_coefs
        Vi = self._Vi
        Vi_prev = self._Vi_prev
        return (1/4 * np.sum(Vi * li * Vi) +
                1/4 * np.sum(Vi_prev * li * Vi_prev))

    def energy_Delta(self):
        """
        Returns
        -------
        float
            ???
        """
        (r0, _, li), _ = self._diffrepr_coefs
        P, V = self.PV
        moy_V = (V + self._V_prev) / 2
        dt_Vi = (self._Vi - self._Vi_prev) / self._dt
        Delta = r0 * moy_V + np.sum(li * dt_Vi, axis=0)
        return self._dt**2/8 * np.sum(Delta * (1/self.mL2) * Delta)

    def dissipated_last_step(self):
        """
        Compute the variation of energy due to viscothermal losses.

        It should be equal to :math:`(E^n - E^{n-1})/dt` (within numerical
        errors).

        If the pipe's internal data has been modified within the last two
        steps, the result may be incorrect.
        """
        (r0, ri, li), (g0, gi, ci) = self._diffrepr_coefs
        P, V = self.PV
        V_pp, V_p = self._V_prevprev, self._V_prev
        Vi_pp, Vi_p, Vi = self._Vi_prevprev, self._Vi_prev, self._Vi
        P_p, Pi_p, Pi = self._P_prev, self._Pi_prev, self._Pi
        muV = (V + V_p)/2
        muV_p = (V_p + V_pp)/2
        muVi = (Vi + Vi_p)/2
        muVi_p = (Vi_p + Vi_pp)/2
        muP = (P + P_p)/2
        muPi = (Pi + Pi_p)/2
        dissip_V0 = np.sum(r0 * (muV**2 + muV_p**2))/2
        dissip_Vi = np.sum(ri * ((muVi - muV)**2 + (muVi_p - muV_p)**2))/2
        dissip_P = np.sum(g0 * (muP)**2)
        dissip_Pi = np.sum(gi * (muPi - muP)**2)
        dissip_power = dissip_V0 + dissip_Vi + dissip_P + dissip_Pi
        assert dissip_power >= 0
        return dissip_power * self._dt

    def add_pressure(self, dP):
        super().add_pressure(dP)
        self._compute_next_p_no_flow()
