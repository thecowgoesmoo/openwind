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
FrequentialPipeFEM with additional variables of diffusive representation.
"""

import numpy as np
from scipy.sparse import csr_matrix, bmat, diags, coo_matrix
from openwind.frequential import FrequentialPipeFEM
from openwind.continuous import ThermoviscousDiffusiveRepresentation

class FrequentialPipeDiffusiveRepresentation(FrequentialPipeFEM):
    """
    Computes the same results as a FrequentialPipeFEM with `diffrepr` losses,
    but slower as it uses additional variables instead of static elimination.

    To be used, the option ``losses="diffrepr+"`` must be given to
    :py:class:`FrequentialSolver<openwind.frequential.frequential_solver.FrequentialSolver>`
    or :py:class:`ImpedanceComputation<openwind.impedance_computation.ImpedanceComputation>`

    Example
    -------
    >>> result = ImpedanceComputation(frequencies, tested_geom,
    ...                                losses='diffrepr+')

    .. note::
        Now, the only interest of this class is to compare two implementation
        (with and without additional variables). it could be usefull for futur
        developments.

    Parameters
    ----------
    pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
        the pipe that will be converted

    **kwarg : keyword arguments
        Keyword with optional precisions for the generation of the \
        :py:class:`Mesh<openwind.discretization.mesh.Mesh>`

    Attributes
    ----------
    label: str
        the label of the pipe
    end_minus, end_plus: :py:class:`FpipeEnd<FPipeEnd>`
        The start and end of the pipe
    nL2, nH1: int
        The number of dof corresponding to the L2 and H1 variables in this pipe
    ntot_dof: int
        The total number of dof in the entire instrument (graph)

    """

    def __init__(self, pipe, **kwargs):
        assert isinstance(pipe.get_losses(), ThermoviscousDiffusiveRepresentation)
        super().__init__(pipe, **kwargs)
        self._arrange_variables()
        self._compute_coefficients()

    def is_compatible_for_modal(self):
        return True

    def _arrange_variables(self):
        """We arrange the system in the following order:
            V, P, V_i, P_0, P_i,
        independently of the convention.

        (This is the order in which system (1.75) is written in Alexis' rapport
        de stage.)
        """
        if self.convention == 'PH1':
            self.n_v, self.n_p = self.nL2, self.nH1
        elif self.convention == 'VH1':
            self.n_v, self.n_p = self.nH1, self.nL2
        else:
            raise ValueError

        self.n_oscillators = self.pipe.get_losses().get_number_of_dampers()

        self._offset_V = 0
        self._offset_P = self._offset_V + self.n_v
        self._offset_Vi = self._offset_P + self.n_p
        self._offset_P0 = self._offset_Vi + self.n_oscillators * self.n_v
        self._offset_Pi = self._offset_P0 + self.n_p

    def _compute_coefficients(self):
        self._viscothermal_coefficients = self.get_diffrepr_coefficients()

    def _compute_Mh(self):
        mass_L2_H1 = self.get_mass_matrices()
        if self.convention == 'PH1':
            mass_V, mass_P = mass_L2_H1
        else:
            mass_P, mass_V = mass_L2_H1

        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients
        mass_Vi = np.ravel(li, order='C') # flatten
#        assert all(np.ravel(li, order='C') == np.concatenate([l for l in li]))
        mass_P0 = c0
        mass_Pi = np.ravel(ci, order='C')

        Mh = np.concatenate([mass_V, mass_P, mass_Vi, mass_P0, mass_Pi])
        # Mh = mass_diag[:, np.newaxis]# + damping_diag[:, np.newaxis]
        return Mh


    def _compute_Kh(self):
        Bh = self.get_Bh()
        if self.convention == 'VH1':
            Bh = -Bh.T

        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients
        # Assemble horizontally a series of blocks with diagonal ri (or gi)
        try:
            ri_blocks = bmat([[diags(r) for r in ri]])
            assert ri_blocks.shape == (self.n_v, self.n_oscillators * self.n_v)
            gi_blocks = bmat([[diags(g) for g in gi]])
            assert gi_blocks.shape == (self.n_p, self.n_oscillators * self.n_p)
        except ValueError:
            ri_blocks = csr_matrix((self.n_v, 0))
            gi_blocks = csr_matrix((self.n_p, 0))

        sum_g = g0 + np.sum(gi, axis=0)

        # assemble the matrix
        local_Kh    = bmat([[None,         -Bh,           -ri_blocks, None,          None      ],
                         [Bh.T,         None,          None,       diags(-sum_g), -gi_blocks],
                         [-ri_blocks.T, None,          None,       None,          None      ],
                         [None,         diags(-sum_g), None,       None,          gi_blocks ],
                         [None,         -gi_blocks.T,  None,       gi_blocks.T,   None      ]
                         ], dtype='float64')

        # Damping coefficients go on the main diagonal of local_Kh
        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients
        damping_V = r0 + np.sum(ri, axis=0)
        damping_P = g0 + np.sum(gi, axis=0)
        damping_Vi = np.ravel(ri, order='C')
        damping_P0 = damping_P
        damping_Pi = np.ravel(gi, order='C')
        damping_diag = np.concatenate([damping_V, damping_P, damping_Vi, damping_P0, damping_Pi])

        mat_ret = local_Kh + np.diag(damping_diag)
        return coo_matrix(mat_ret)


    def _compute_indep_freq(self):
        """Overrides method in FrequentialPipeFEM.

        Compute local component of matrix Ah_nodiag.
        """
        Bh = self.get_Bh()
        if self.convention == 'VH1':
            Bh = -Bh.T

        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients
        # Assemble horizontally a series of blocks with diagonal ri (or gi)
        try:
            ri_blocks = bmat([[diags(r) for r in ri]])
            assert ri_blocks.shape == (self.n_v, self.n_oscillators * self.n_v)
            gi_blocks = bmat([[diags(g) for g in gi]])
            assert gi_blocks.shape == (self.n_p, self.n_oscillators * self.n_p)
        except ValueError:
            ri_blocks = csr_matrix((self.n_v, 0))
            gi_blocks = csr_matrix((self.n_p, 0))

        sum_g = g0 + np.sum(gi, axis=0)

        # assemble the matrix
        local_Ah = bmat([[None,         -Bh,           -ri_blocks, None,          None      ],
                         [Bh.T,         None,          None,       diags(-sum_g), -gi_blocks],
                         [-ri_blocks.T, None,          None,       None,          None      ],
                         [None,         diags(-sum_g), None,       None,          gi_blocks ],
                         [None,         -gi_blocks.T,  None,       gi_blocks.T,   None      ]
                         ], dtype='complex128')
        return local_Ah

    def _compute_diags(self, omegas_scaled):
        """Overrides method in FrequentialPipeFEM.

        Compute local component of matrix Ah_diags.
        """
        mass_L2_H1 = self.get_mass_matrices()
        if self.convention == 'PH1':
            mass_V, mass_P = mass_L2_H1
        else:
            mass_P, mass_V = mass_L2_H1

        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients
        mass_Vi = np.ravel(li, order='C') # flatten
#        assert all(np.ravel(li, order='C') == np.concatenate([l for l in li]))
        mass_P0 = c0
        mass_Pi = np.ravel(ci, order='C')

        mass_diag = np.concatenate([mass_V, mass_P, mass_Vi, mass_P0, mass_Pi])

        # Damping coefficients go on the main diagonal
        damping_V = r0 + np.sum(ri, axis=0)
        damping_P = g0 + np.sum(gi, axis=0)
        damping_Vi = np.ravel(ri, order='C')
        damping_P0 = damping_P
        damping_Pi = np.ravel(gi, order='C')
        damping_diag = np.concatenate([damping_V, damping_P, damping_Vi, damping_P0, damping_Pi])





        local_Ah_diags = mass_diag[:, np.newaxis] * 1j * omegas_scaled + damping_diag[:, np.newaxis]

#        print("Rescaled mass_V {:.3e}".format(mass_V[0] * self.pipe.get_scaling().get_scaling_Zv()))
#        print("Rescaled mass_P {:.3e}".format(mass_P[0] * self.pipe.get_scaling().get_scaling_Yt()))
#        print("Rescaled mass_Vi {:.3e}".format(mass_Vi[0] * self.pipe.get_scaling().get_scaling_Zv()))
#        print("Rescaled mass_P0 {:.3e}".format(mass_P0[0] * self.pipe.get_scaling().get_scaling_Yt()))
#        print("Rescaled mass_Pi {:.3e}".format(mass_Pi[0] * self.pipe.get_scaling().get_scaling_Yt()))

        return local_Ah_diags

    def get_number_dof(self):
        return (1 + self.n_oscillators) * self.n_v + (2 + self.n_oscillators) * self.n_p

    def get_first_L2_index(self):
        if self.convention == 'PH1':
            return self.get_first_index() + self._offset_V
        else:
            return self.get_first_index() + self._offset_P

    def get_first_H1_index(self):
        if self.convention == 'PH1':
            #print("get_first_H1_index() ->", self.get_first_index(),'+', self._offset_P,'=', self.get_first_index() + self._offset_P)
            return self.get_first_index() + self._offset_P
        else:
            return self.get_first_index() + self._offset_V

    def check_solution(self, Uh, omega_scaled):
        """DEBUG : Double-check that the solution seems correct."""

        # Check assuming PH1
        assert self.convention == 'PH1'

        mass_V, mass_P = self.get_mass_matrices()

        (r0, ri, li), (g0, gi, c0, ci) = self._viscothermal_coefficients

        # Extract data from Uh
        Uh_ = Uh[self.get_indices()]
        V = Uh_[self._offset_V:self._offset_V + self.n_v]
        P = Uh_[self._offset_P:self._offset_P + self.n_p]
        Vi = Uh_[self._offset_Vi:self._offset_Vi + self.n_v * self.n_oscillators]
        Vi = np.reshape(Vi, (self.n_oscillators, self.n_v), order='C')
        P0 = Uh_[self._offset_P0:self._offset_P0 + self.n_p]
        Pi = Uh_[self._offset_Pi:self._offset_Pi + self.n_p * self.n_oscillators]
        Pi = np.reshape(Pi, (self.n_oscillators, self.n_p), order='C')

        # Equations on V, Vi, P0 and Pi should be exactly verified
        eq_V = -self.get_Bh() @ P + 1j * omega_scaled * mass_V * V + r0 * V + np.sum(ri * (V - Vi), axis=0)
        print("eq_V :", eq_V)
#        print("-Bh", -self.get_Bh().todense())
#        print("j omega mass_V",  1j * omega_scaled * mass_V)
#        print("r0 + sum(ri)", r0 + np.sum(ri, axis=0))
#        print("-ri", -ri)
#        lines_V = bmat([[1j * omega_scaled * diags(mass_V) + diags(r0), -self.get_Bh()]])
#        print("lines_V\n", lines_V.todense())
#        print("lines_V @ Uh[self.n_v + self.n_p]", lines_V @ Uh_[:self.n_v + self.n_p])

        eq_Vi = li * 1j * omega_scaled * Vi + ri * (Vi - V)
        print("eq_Vi :", eq_Vi)
        eq_Vi_2 = Vi - ri / (li * 1j * omega_scaled + ri) * V
        print("eq_Vi_2 :", eq_Vi_2)

        z_v = r0 + np.sum(1/(1/ri + 1/(li*1j*omega_scaled)), axis=0)
        eq_Vi_3 = r0 * V + np.sum(ri * (V - Vi), axis=0) - z_v * V
        print("eq_Vi_3 :", eq_Vi_3)
        print("z_v :", z_v / self.mesh.get_weights())
        eq_P0 = c0 * 1j * omega_scaled * P0 + g0 * (P0 - P) + np.sum(gi * (Pi + P0 - P), axis=0)
        print("eq_P0 :", eq_P0)
        eq_Pi = ci * 1j * omega_scaled * Pi + gi * (Pi + P0 - P)
        print("eq_Pi :", eq_Pi)

        assert np.allclose(eq_V, 0)
        assert np.allclose(eq_Vi, 0)
        assert np.allclose(eq_Vi_2, 0)
        assert np.allclose(eq_Vi_3, 0)
        assert np.allclose(eq_P0, 0)
        assert np.allclose(eq_Pi, 0)

        # Equation on P should be verified except at the boundary
        eq_P = self.get_Bh().T @ V + 1j * omega_scaled * mass_P * P + g0 * (P - P0) + np.sum(gi * (P - P0 - Pi), axis=0)
        print("eq_P :", eq_P)

#        local_Ah = self._compute_indep_freq() + diags(self._compute_diags(omega_scaled)[:,0])
#        print("local_Ah")
#        print(local_Ah.todense())
#        print("local_Ah @ Uh_", local_Ah @ Uh_)

        assert np.allclose(eq_P[1:-1], 0)
