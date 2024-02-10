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
"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.csr import csr_matrix
# Low-level solution to avoid overhead every time we do a matrix-vector multiplication
from scipy.sparse._sparsetools import csr_matvec

from openwind.continuous import ThermoviscousDiffusiveRepresentation
from openwind.temporal import TemporalPipe


class TemporalLossyPipe(TemporalPipe):
    """
    A temporal pipe with viscothermal losses.

    It is a modified version of :py:class:`TemporalPipe\
    <openwind.temporal.tpipe.TemporalPipe>`.

    The loss model must be 'diffrepr'.

    More details on the scheme can be found in [Thibault_losses]_.

    See Also
    --------
    :py:class:`ThermoviscousDiffusiveRepresentation<openwind.continuous.thermoviscous_models.ThermoviscousDiffusiveRepresentation>`
        The losses model adapted for this scheme (corresponding to the keywrd
        'diffrepr')

    Parameters
    ----------
    pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
        The pipe we are simulating.
    t_solver : :py:class:`TemporalSolver <openwind.temporal.temporal_solver.TemporalSolver>`
        The temporal solver leading this pipe
    **params : keywords
        Discretization parameters. See :py:class:`Mesh<openwind.discretization.mesh.Mesh>`.

    References
    ----------
    .. [Thibault_losses] Thibault, A. and Chabassier, J. "Dissipative time-domain 1D
        model for viscothermal acoustic propagation in wind instruments".
        To be published.

    Attributes
    ----------
    label: str
        the label of the pipe

    """

    __slots__ = {"PV":"(array, array): tuple with the Pressure/Volume along the pipe.",
                 "end_minus":':py:class:`.TemporalPipeEnd`: The "upstream" (or left or minus or start) end of the pipe',
                 "end_plus": ':py:class:`.TemporalPipeEnd`: The "downstream" (or rigth or plus) end of the pipe',
                 "_diffrepr_coefs":None,
                 "_loss_N":None,
                 "nH1":None, "nL2":None,
                 "_P0":None, "_Pi":None,
                 "_P_prev":None, "P0_prev":None, "_Pi_prev":None,
                 "_Vi":None, "_V_prev":None, "_Vi_prev":None,
                 "_V_prevprev":None,
                 "_next_p_no_flow":None,
                 "mH1":None, "mL2":None, "Bh":None,
                 "MH1inv":None, "ML2inv":None,
                 "dtinvMBt":None, "dtinvML2B":None,
                 "energy_matrix":None,
                 "p_to_p_noflow":None,
                 "p0_to_p_noflow":None,
                 "pi_to_p_noflow":None,
                 "v_to_p_noflow":None,
                 "p_to_p0":None,
                 "p0_to_p0":None,
                 "pi_to_p0":None,
                 "p_to_pi":None,
                 "pi_to_pi":None,
                 "v_to_v":None,
                 "vi_to_v":None,
                 "p_to_v":None,
                 "v_to_vi":None,
                 "vi_to_vi":None,
                 }

    def __init__(self, pipe, t_solver, **params):
        super().__init__(pipe, t_solver, **params)
        assert isinstance(pipe.get_losses(), ThermoviscousDiffusiveRepresentation)

    def _precompute_losses(self):
        """Precomputes coefficients used for the update with losses."""
        self._diffrepr_coefs = self.get_diffrepr_coefficients()
        (r0, ri, li), (g0, gi, c0, ci) = self._diffrepr_coefs
        self._loss_N = ri.shape[0]

    def reset_variables(self):
        super().reset_variables()
        self._P0 = np.zeros(self.nH1)
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


        # ---*** NEW SCHUR COMPLEMENT ***---

        (r0, ri, li), (g0, gi, c0, ci) = self._diffrepr_coefs
        r0_, ri_, g0_, gi_ = r0/2, ri/2, g0/2, gi/2
        mP_, mV_, li_, c0_, ci_ = self.mH1/dt, self.mL2/dt, li/dt, c0/dt, ci/dt

        # Coefficients for computing P
        # Gamma_ = g0_ + np.sum(gi_, axis=0)
        # alpha_i = gi_ / (ci_ + gi_)
        gi_2 = (gi_ * ci_) / (ci_ + gi_)
        Gamma_2 = g0_ + np.sum(gi_2, axis=0)
        Gamma_3 = (c0_ * Gamma_2) / (c0_ + Gamma_2)
        denom_P = mP_ + Gamma_3
        self.p_to_p_noflow = mP_ / denom_P
        self.p0_to_p_noflow = Gamma_3 / denom_P
        self.pi_to_p_noflow = gi_2 / Gamma_2 * self.p0_to_p_noflow
        assert np.allclose(self.pi_to_p_noflow, gi_2*c0_/(c0_+Gamma_2)/denom_P)
        self.v_to_p_noflow = - diags(1/(2*denom_P)) @ self.Bh.T  # csr sparse matrix
        assert isinstance(self.v_to_p_noflow, csr_matrix)
        self.end_minus.set_alpha(1/(2 * denom_P[0]))
        self.end_plus.set_alpha(1/(2 * denom_P[-1]))

#        # <<<--- Coefficients for P_next : debug
#        self.p_to_p = (mP_ - Gamma_3) / (mP_ + Gamma_3)
#        assert np.allclose(self.p_to_p_noflow, (1+self.p_to_p)/2)
#        self.p0_to_p = 2 * Gamma_3 / (mP_ + Gamma_3)
#        assert np.allclose(self.p0_to_p_noflow, self.p0_to_p/2)
#        self.pi_to_p = 2 * (gi_2 * c0_) / (c0_ + Gamma_2) / (mP_ + Gamma_3)
#        assert np.allclose(self.pi_to_p_noflow, self.pi_to_p/2)
#        self.v_to_p = -diags(1/(mP_ + Gamma_3)) @ self.Bh.T
#        assert np.allclose(self.v_to_p_noflow.todense(), self.v_to_p.todense()/2)
#        # --->>>


        denom_P0 = c0_ + Gamma_2
        self.p_to_p0 = Gamma_2 / denom_P0
        self.p0_to_p0 = (c0_ - Gamma_2) / denom_P0
        self.pi_to_p0 = -2 * gi_2 / denom_P0

        self.p_to_pi = gi_ / (ci_ + gi_)
        self.pi_to_pi = (ci_ - gi_) / (ci_ + gi_)

        # Coefficients for computing V
        ri_2 = (ri_ * li_) / (ri_ + li_)
        Psi_ = r0_ + np.sum(ri_2, axis=0)
        denom_V = mV_ + Psi_
        self.v_to_v = (mV_ - Psi_) / denom_V
        self.vi_to_v = 2 * ri_2 / denom_V
        self.p_to_v = diags(1/denom_V) @ self.Bh
        assert isinstance(self.p_to_v, csr_matrix)
        self.v_to_vi = ri_ / (li_ + ri_)
        self.vi_to_vi = (li_ - ri_) / (li_ + ri_)

    def __str__(self):
        return "TLossyPipe{}".format(self.label)

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

    def one_step(self, check_scheme=False):
        """Advance one time step.

        Assumes the flux of both pipe ends have already been updated.

        See [Thibault_losses]_ for more details.

        Raises
        ------
        AssertionError
            if either of the fluxes of the pipe ends has not been updated since
            the last time step
        """
#        dt = self.dt
#        hdt = self.dt/2
        P, V = self.PV
        P0 = self._P0
        Pi = self._Pi
        Vi = self._Vi
#        flow_left = self.end_minus.get_w_nph() # Boundary effects
#        flow_right = self.end_plus.get_w_nph()

        # Update of P
        P_nph = self.get_p_no_flow().copy()
        P_nph[0] = self.end_minus.accept_q_nph()
        P_nph[-1] = self.end_plus.accept_q_nph()

        # P_next = 2*P_nph - P
        # P0_next = self.p_to_p0 * (P + P_next) + self.p0_to_p0 * P0 + np.add.reduce(self.pi_to_p0 * Pi, axis=0)
        # Pi_next = self.p_to_pi * (P + P_next - P0 - P0_next) + self.pi_to_pi * Pi
        # # Update of V
        # V_next = self.v_to_v * V + np.add.reduce(self.vi_to_v * Vi, axis=0) + self.p_to_v @ P_next
        # Vi_next = self.v_to_vi * (V + V_next) + self.vi_to_vi * Vi

        # Limit dynamic memory allocation by using in-place operations

        P_next = 2*P_nph
        P_next -= P

        # P0_next = self.p_to_p0 * (P + P_next) + self.p0_to_p0 * P0 + np.sum(self.pi_to_p0 * Pi, axis=0)
        P0_next = P+P_next
        P0_next *= self.p_to_p0
        P0_next += self.p0_to_p0 * P0
        # P0_next += np.sum(self.pi_to_p0 * Pi, axis=0)
        P0_next += np.add.reduce(self.pi_to_p0 * Pi, axis=0) # Slightly more efficient?

        # Pi_next = self.p_to_pi * (P + P_next - P0 - P0_next) + self.pi_to_pi * Pi
        Pi_next = self.p_to_pi * (P + P_next - P0 - P0_next)
        Pi_next += self.pi_to_pi * Pi

        # Update of V
        # V_next = self.v_to_v * V + np.sum(self.vi_to_v * Vi, axis=0) + self.p_to_v @ P_next
        # V_next = np.sum(self.vi_to_v * Vi, axis=0)
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



#        if check_scheme:
#            dt = self._dt
#            (r0, ri, li), (g0, gi, c0, ci) = self._diffrepr_coefs
#            # Check that the scheme is properly implemented
#            eq1 = self.mL2 * (V_next - V)/dt + r0 * (V_next + V)/2 + np.sum(ri * (V_next + V - Vi_next - Vi)/2, axis=0) - self.Bh @ P_next
#            eq2 = self.mH1 * (P_next - P)/dt + g0 * (P_next + P - P0_next - P0)/2 + np.sum(gi * (P_next + P - P0_next - P0 - Pi_next - Pi)/2, axis=0) + self.Bh.T @ V
#            eq2[0] += flow_left
#            eq2[-1] += flow_right
#            eq3 = li * (Vi_next - Vi)/dt + ri * (Vi_next + Vi - V_next - V)/2
#            eq4 = c0 * (P0_next - P0)/dt + g0 * (P0_next + P0 - P_next - P)/2 + np.sum(gi * (Pi_next + Pi + P0_next + P0 - P_next - P)/2, axis=0)
#            eq5 = ci * (Pi_next - Pi)/dt + gi * (Pi_next + Pi + P0_next + P0 - P_next - P)/2

#            print("Rel. error on dtV :",np.sum(np.abs(eq1)) / np.sum(np.abs(diags(self.mL2) * (V_next - V)/dt)))
#            print("Rel. error on dtP :",np.sum(np.abs(eq2)) / np.sum(np.abs(self.mH1 * (P_next - P)/dt)))
#            print("Rel. error on dtVi :",np.sum(np.abs(eq3)) / np.sum(np.abs(li * (Vi_next - Vi)/dt)))
#            print("Rel. error on dtP0 :",np.sum(np.abs(eq4)) / np.sum(np.abs(c0 * (P0_next - P0)/dt)))
#            print("Rel. error on dtPi :",np.sum(np.abs(eq5)) / np.sum(np.abs(ci * (Pi_next - Pi)/dt)))
#            plt.plot(eq1)

        # Put the new variables in the right place
        self.PV = (P_next, V_next)
        # Save a few steps (only useful for energy/power checks)
        self._V_prevprev, self._Vi_prevprev = self._V_prev, self._Vi_prev
        self._P_prev, self._P0_prev, self._Pi_prev = P, P0, Pi
        self._V_prev, self._Vi_prev = V, Vi
        # and update the scheme
        self._P0, self._Pi, self._Vi = P0_next, Pi_next, Vi_next

        # Prepare the next step of P
        self._compute_next_p_no_flow()

    def _compute_next_p_no_flow(self):
        """
        Compute p_{no flow}^{n+1/2} from P^n and V^{n+1/2} assuming no
        boundary flow.
        """
        P, V = self.PV
        # self._next_p_no_flow = (self.p_to_p_noflow * P + self.p0_to_p_noflow * self._P0 +
        #             np.add.reduce(self.pi_to_p_noflow * self._Pi, axis=0)
        #             + self.v_to_p_noflow * V) # /!\ Sparse matrix - vector multiplication

        self._next_p_no_flow = (self.p_to_p_noflow * P + self.p0_to_p_noflow * self._P0 +
                    np.add.reduce(self.pi_to_p_noflow * self._Pi, axis=0))

        # Low-level call for efficient sparse matrix-vector multiplication
        # csr_matmul(M, N, indptr, indices, data, other, result)
        # computes Y += A*X
        assert self.v_to_p_noflow.shape == (self.nH1, self.nL2)
        csr_matvec(self.nH1, self.nL2, self.v_to_p_noflow.indptr,
                   self.v_to_p_noflow.indices,
                   self.v_to_p_noflow.data, V,
                   self._next_p_no_flow)

#        assert np.allclose(self._next_p_no_flow, (P + self._compute_P_next())/2)

#    def _compute_P_next(self):
#        P, V = self.PV
#        P_next = (self.p_to_p * P + self.p0_to_p * self._P0 +
#                    np.sum(self.pi_to_p * self._Pi, axis=0)
#                    + self.v_to_p @ V)
#        return P_next


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
        energy_P0 = self.energy_P0()
        energy_Pi = self.energy_Pi()
        energy_Vi = self.energy_Vi()
        energy_Delta = self.energy_Delta()
        energies = [energy_P, energy_V, energy_P0, energy_Pi, energy_Vi, energy_Delta]
        assert all([e >= 0 for e in energies])
        return sum(energies)

    def energy_P0(self):
        """
        Returns
        -------
        float
            The amount of energy stored in the auxiliary variables P0
        """
        _, (_, _, c0, _) = self._diffrepr_coefs
        P0 = self._P0
        return 1/2 * np.sum(P0 * c0 * P0)

    def energy_Pi(self):
        """
        Returns
        -------
        float
            The amount of energy stored in the auxiliary variables Pi
        """
        _, (_, _, _, ci) = self._diffrepr_coefs
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
        (r0, ri, li), (g0, gi, c0, ci) = self._diffrepr_coefs
        P, V = self.PV
        V_pp, V_p = self._V_prevprev, self._V_prev
        Vi_pp, Vi_p, Vi = self._Vi_prevprev, self._Vi_prev, self._Vi
        P_p, P0_p, P0, Pi_p, Pi = self._P_prev, self._P0_prev, self._P0, self._Pi_prev, self._Pi
        muV = (V + V_p)/2
        muV_p = (V_p + V_pp)/2
        muVi = (Vi + Vi_p)/2
        muVi_p = (Vi_p + Vi_pp)/2
        muP = (P + P_p)/2
        muPi = (Pi + Pi_p)/2
        muP0 = (P0 + P0_p)/2
        dissip_V0 = np.sum(r0 * (muV**2 + muV_p**2))/2
        dissip_Vi = np.sum(ri * ((muVi - muV)**2 + (muVi_p - muV_p)**2))/2
        dissip_P0 = np.sum(g0 * (muP0 - muP)**2)
        dissip_Pi = np.sum(gi * (muPi + muP0 - muP)**2)
        dissip_power = dissip_V0 + dissip_Vi + dissip_P0 + dissip_Pi
        assert dissip_power >= 0
#        print("Dissipated energy =",dissip_power*self._dt)
        return dissip_power * self._dt

    def add_pressure(self, dP):
        super().add_pressure(dP)
        self._compute_next_p_no_flow()
