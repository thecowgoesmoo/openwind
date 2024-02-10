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

import numpy as np
from scipy.sparse import csr_matrix, bmat, hstack, coo_matrix

from openwind.continuous import EndPos
from openwind.frequential import FrequentialComponent
from openwind.discretization import DiscretizedPipe

import pdb

class FPipeEnd:
    """
    Access to one end of a FrequentialPipeFEM.

    Parameters
    ----------
    f_pipe : :py:class:`FrequentialPipeFEM`
        the pipe to which this End corresponds
    pos : {:py:class:`EndPos.MINUS, EndPos.PLUS<openwind.continuous.netlist.EndPos>`}
        Position in the pipe's indices. MINUS if start of the pipe, PLUS if end.

    Attributes
    ----------
    convention: {'PH1', 'VH1'}
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
    """

    def __init__(self, f_pipe, pos):
        self.f_pipe = f_pipe
        assert isinstance(pos, EndPos)
        self.pos = pos
        self.convention = f_pipe.convention

    def get_index(self):
        """
        Index of the H1 variable at this pipe-end, in the assembled vector
        of all unknowns Uh.

        Returns
        -------
        int
        """
        indices = self.f_pipe.get_H1_indices()
        return indices[self.pos.array_pos]

    def get_index_L2(self):
        """Index of the L2 variable at this pipe-end, in the assembled vector
        of all unknowns Uh.

        Returns
        -------
        int
        """
        indices = self.f_pipe.get_L2_indices()
        return indices[self.pos.array_pos]

    def get_physical_params(self):
        """
        The value of some physical quantities at this pipe-end

        Returns
        -------
        radius : float
            The radius of the pipe
        rho : float
            The air density
        c : float
            The sound celerity
        """
        radius = self.f_pipe.pipe.get_radius_at(self.pos.x)
        rho = self.f_pipe.pipe.get_physics().rho(self.pos.x)
        c = self.f_pipe.pipe.get_physics().c(self.pos.x)
        return radius, rho, c

    def get_diff_radius(self, diff_index):
        """
        Differentiate the radius wr to one design paramter.

        Parameters
        ----------
        diff_index : int
            The index of the design parameter to which differentiate, such as
            stocked in :py:class:`OptimizationParameters\
            <openwind.design.design_parameter.OptimizationParameters>`

        Returns
        -------
        float

        """

        return self.f_pipe.pipe.get_diff_radius_at(self.pos.x, diff_index)


class FrequentialPipeFEM(FrequentialComponent, DiscretizedPipe):
    """
    A frequential version of a pipe assuming FEM discretization.


    Computes for every frequency, every data for the linear system to solve

    .. math::
        A_h U_h = L_h

    The pipe only contributes to the matrix :math:`A_h` with a block placed
    in the global matrix at the indices corresponding to the dof of this pipe.
    The expression of this block is specified in the docstring of
    :py:class:`DiscretizedPipe<openwind.discretization.discretized_pipe.DiscretizedPipe>`.

    Parameters
    ----------
    pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
        the pipe that will be converted

    **kwarg : keyword arguments
        Keyword with optional precisions for the generation of the \
        :py:class:`Mesh<openwind.discretization.mesh.Mesh>`

    Attributes
    ----------
    end_minus, end_plus: :py:class:`FpipeEnd<FPipeEnd>`
        The start and end of the pipe
    nL2, nH1: int
        The number of dof corresponding to the L2 and H1 variables in this pipe
    ntot_dof: int
        The total number of dof in the entire instrument (graph)


    """

    def __init__(self, pipe, **kwargs):
        DiscretizedPipe.__init__(self, pipe, **kwargs)
        self.end_minus = FPipeEnd(self, EndPos.MINUS)
        self.end_plus = FPipeEnd(self, EndPos.PLUS)
        
    def is_compatible_for_modal(self):
        return DiscretizedPipe.is_compatible_for_modal(self)
    
    def _compute_Mh(self):
        """ Compute all the diagonals of the matrix Ah.
        All the diagonals are keep in a dense matrix used to actualize
        the diagonal of the sparse matrix at each iteration of the
        frequency loop (best methodology).

        .. warning ::

            Different for frequential_pipe_diffusive_representation

        """
        local_Mh_1, local_Mh_2 = self.get_mass_matrices()
        return np.concatenate((local_Mh_1.T, local_Mh_2.T)) 

    def _compute_Kh(self):
        Bh = self.get_Bh()
        # everything outside the diagonal
        local_Kh = bmat([[None, -Bh], [Bh.T, None]], dtype='float64')
        return coo_matrix(local_Kh)
    
    
    def _compute_indep_freq(self):
        Bh = self.get_Bh()
        # everything outside the diagonal
        local_Ah = bmat([[None, -Bh], [Bh.T, None]], dtype='complex128')
        return local_Ah

    def _compute_diags(self, omegas_scaled):
        """ Compute all the diagonals of the matrix Ah.
        All the diagonals are keep in a dense matrix used to actualize
        the diagonal of the sparse matrix at each iteration of the
        frequency loop (best methodology).

        .. warning ::

            Different for frequential_pipe_diffusive_representation

        """
        local_Ah_diags = self.get_mass_matrices_with_losses(omegas_scaled)
        return local_Ah_diags

    def get_ends(self):
        """
        The two ends of the pipe (minus and plus)

        Returns
        -------
        tuple of :py:class:`FpipeEnd<FPipeEnd>`
        """
        return self.end_minus, self.end_plus

    def get_number_dof(self):
        return self.nL2 + self.nH1

    def get_first_L2_index(self):
        """
        The index of the first L2 dof in the entire instrument

        Returns
        -------
        int
        """
        return self.get_first_index()

    def get_first_H1_index(self):
        """
        The index of the first H1 dof in the entire instrument

        Returns
        -------
        int
        """
        return self.get_first_index() + self.nL2

    def get_L2_indices(self):
        """
        Return the indices of the L2 variable as located in the entire instrument.

        *Usage:*

        .. code-block:: python

            V = Uh[f_pipe.get_L2_indices()]         # if convention is 'PH1'

        Returns
        -------
        range

        """
        i0 = self.get_first_L2_index()
        return range(i0, i0 + self.nL2)

    def get_H1_indices(self):
        """
        Return the indices of the H1 variable as located in the entire instrument.


        *Usage:*

        .. code-block:: python

            P = Uh[f_pipe.get_H1_indices()]         # if convention is 'PH1'

        Returns
        -------
        range

        """
        i0 = self.get_first_H1_index()
        return range(i0, i0 + self.nH1)

    def get_contrib_Mh(self):
        local_Mh = self._compute_Mh()
        indices = self.get_indices()
        return indices, local_Mh
    
    def get_contrib_Kh(self):
        local_Kh = self._compute_Kh()
        # return self.place_in_big_matrix_float(local_Kh)
        return self.place_in_big_matrix(local_Kh)
        
    def get_contrib_indep_freq(self):
        local_Ah = self._compute_indep_freq()
        return self.place_in_big_matrix(local_Ah)

    def get_contrib_freq(self, omegas_scaled):
        """
        Compute all the diagonals of the matrix Ah.
        All the diagonals are keep in a dense matrix used to actualize
        the diagonal of the sparse matrix at each iteration of the
        frequency loop (best methodology).
        """
        local_Ah_diags = self._compute_diags(omegas_scaled)
        return self.get_indices(), local_Ah_diags

    def place_interp_matrix(self, x_interp_local, variable='L2'):
        """
        Matrix to interpolate a variable at some x values.

        Parameters
        ----------
        x_interp_local : array of float
            The x-value at which interpolate the variables.
        variable : {'L2', 'H1'}, optional
            The variable to interpolate. The default is 'L2'.

        Returns
        -------
        interp_mat : sparse matrix

        """
        n_interp = len(x_interp_local)
        n_left = self.get_first_index()
        mat_left = csr_matrix((n_interp, n_left))
        n_right = self.ntot_dof - self.get_indices().stop
        mat_right = csr_matrix((n_interp, n_right))
        mat_H1 = csr_matrix((n_interp, self.nH1))
        mat_L2 = csr_matrix((n_interp, self.nL2))
        if variable=='L2':
            local_mat = self.get_interp_mat_L2(x_interp_local)
            interp_mat = hstack([mat_left, local_mat, mat_H1, mat_right])
        else:
            local_mat = self.get_interp_mat_H1(x_interp_local)
            interp_mat = hstack([mat_left, mat_L2, local_mat, mat_right])
        return  interp_mat

    def place_interp_matrix_grad(self, x_interp_local, variable='L2'):
        """
        Matrix to interpolate a the gradient of a variable at some x values.

        Parameters
        ----------
        x_interp_local : array of float
            The x-value at which interpolate the variables.
        variable : {'L2', 'H1'}, optional
            The variable to interpolate. The default is 'L2'.

        Returns
        -------
        interp_mat : sparse matrix

        """
        n_interp = len(x_interp_local)
        n_left = self.get_first_index()
        mat_left = csr_matrix((n_interp, n_left))
        n_right = self.ntot_dof - self.get_indices().stop
        mat_right = csr_matrix((n_interp, n_right))
        mat_H1 = csr_matrix((n_interp, self.nH1))
        mat_L2 = csr_matrix((n_interp, self.nL2))
        if variable=='L2':
            local_mat = self.get_interp_mat_L2_grad(x_interp_local)
            interp_mat = hstack([mat_left, local_mat, mat_H1, mat_right])
        else:
            local_mat = self.get_interp_mat_H1_grad(x_interp_local)
            interp_mat = hstack([mat_left, mat_L2, local_mat, mat_right])
        return  interp_mat

    # %% differential
    def _compute_diags_dAh(self, omegas_scaled, diff_index):
        local_dAh_diags = self.get_mass_matrices_with_losses_dAh(omegas_scaled,
                                                                 diff_index)
        return local_dAh_diags

    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        local_dAh_diags = self._compute_diags_dAh(omegas_scaled, diff_index)
        # dAh_diags = np.zeros((self.ntot_dof, len(omegas_scaled)),
        #                     dtype='complex128')
        # dAh_diags[self.get_indices(), :] = local_dAh_diags
        return self.get_indices(), local_dAh_diags
