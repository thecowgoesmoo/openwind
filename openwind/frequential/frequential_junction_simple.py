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
import scipy.sparse as ssp

from openwind.frequential import FrequentialComponent


class FrequentialJunctionSimple(FrequentialComponent):
    """
    Frequential representation of a junction between two pipes without masses.

    Assumes convention PH1.

    This component contributes only to the matrix :math:`A_h` :

    .. code-block:: text

                         ┌                        ┐
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  1  .  . │ ← line of the 1st pipe end's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
            Ah_contrib = │ .  .  .  .  . -1  .  . │ ← line of the 2nd pipe end's d.o.f.
                         │ . -1  .  .  1  .  .  . │ ← line of this component's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
                         └                        ┘


    Parameters
    ----------
    junc : :py:class:`JunctionSimple<openwind.continuous.junction.JunctionSimple>`
        The continuous version of the junction which is converted

    ends : list of 2 :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>`\
        or :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        The pipe ends this junction connects
    """

    def __init__(self, junc, ends):       
        self.junc = junc
        assert len(ends) == 2
        self.ends = ends
        self.__set_physical_params()
        if any(end.convention != 'PH1' for end in ends):
            msg = ("FrequentialJunction does not yet support VH1 convention")
            raise ValueError(msg)
    
    def is_compatible_for_modal(self):
        return self.junc.is_compatible_for_modal()
    
    def __set_physical_params(self):
        radii = []
        rhos = []
        for end in self.ends:
            radius, rho, _ = end.get_physical_params()
            radii.append(radius)
            rhos.append(rho)
        assert all(np.isclose(rhos, rho))
        self.rho = sum(rhos)/2.0
        self.r_minus, self.r_plus = radii

    def get_number_dof(self):
        return 1

    
    def get_contrib_Kh(self):
        # assembled_interaction_matrix = ssp.lil_matrix((self.ntot_dof,
        #                                                self.ntot_dof))
        # interaction = [-1, 1]
        # for i, f_pipe_end in enumerate(self.ends):
        #     assembled_interaction_matrix[self.get_indices(),
        #                                  f_pipe_end.get_index()] = interaction[i]
        # return assembled_interaction_matrix - assembled_interaction_matrix.T
        row = list()
        col = list()
        data = np.array([-1, 1])
        for f_pipe_end in self.ends:
            row.append(list(self.get_indices()))
            col.append(f_pipe_end.get_index())
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_tot = np.append(data, -1*data)
        return row_tot, col_tot, data_tot

    def get_contrib_indep_freq(self):
        row = list()
        col = list()
        data = np.array([-1, 1])
        for f_pipe_end in self.ends:
            row.append(list(self.get_indices()))
            col.append(f_pipe_end.get_index())
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_tot = np.append(data, -1*data)
        return row_tot, col_tot, data_tot
