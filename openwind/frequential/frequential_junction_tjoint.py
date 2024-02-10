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

from openwind.frequential import FrequentialComponent


class FrequentialJunctionTjoint(FrequentialComponent):
    """
    Frequential representation of a T-junction of pipes.

    Assumes similar main tube radius at both side of the branched tube.
    Assumes convention PH1. This component contributes only to the matrix
    :math:`A_h` :

    .. code-block:: text

                                 ┌                              ┐
                                 │ .  .  .  .  .  .  .  .  .  . │
                                 │ .  .  .  .  .  .  . -1  1  . │ ← line of the 1st pipe end's d.o.f.
                                 │ .  .  .  .  .  .  .  .  .  . │
                                 │ .  .  .  .  .  .  .  1  1  . │ ← line of the 2nd pipe end's d.o.f.
         Ah_contrib =  1/sqrt(2) │ .  .  .  .  .  .  .  .  .  . │
                                 │ .  .  .  .  .  .  .  . -2  . │ ← line of the 3rd pipe end's d.o.f.
                                 │ .  .  .  .  .  .  .  .  .  . │
                                 │ .  1  . -1  .  .  . jwS .  . │ ← line of the 1st junction's d.o.f.
                                 │ . -1  . -1  .  2  .  . jwA . │ ← line of the 2nd junction's d.o.f.
                                 │ .  .  .  .  .  .  .  .  .  . │
                                 └                              ┘


    where `jwS`:math:`=j\\omega m_s 2 \\sqrt{2}` and
    `jwA`:math:`=j\\omega m_a / \\sqrt{2}` with :math:`m_s, m_a` the acoustic
    masses specified in :py:class:`JunctionTjoint<openwind.continuous.junction.JunctionTjoint>`.

    Parameters
    ----------
    junc : :py:class:`JunctionTjoint<openwind.continuous.junction.JunctionTjoint>`
        The continuous version of the junction which is converted

    ends : list of 3 :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>`\
        or :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        The pipe ends this junction connects
    """

    def __init__(self, junc, ends):
        self.junc = junc
        assert len(ends) == 3, f"{junc.label}: A T-junction needs 3 pipe ends."
        self.ends = ends
        if any(end.convention != 'PH1' for end in ends):
            msg = ("FrequentialJunction needs PH1 convention")
            raise ValueError(msg)

    def is_compatible_for_modal(self):
        return self.junc.is_compatible_for_modal()
    
    def __get_physical_params(self):
        radii = []
        rhos = []
        for end in self.ends:
            radius, rho_i, _ = end.get_physical_params()
            radii.append(radius)
            rhos.append(rho_i)
        assert all(np.isclose(rhos, rho_i)), f'"{self.ends[2].f_pipe.pipe.label}": The air density is discontinuous and should not!'
        assert np.isclose(radii[0], radii[1]), f'"{self.ends[2].f_pipe.pipe.label}": The radius of the main bore is discontinuous and should not!'
        rho = sum(rhos)/3.0
        r_main = sum(radii[0:2])/2.0
        r_side = radii[2]
        return r_main, r_side, rho

    def __get_masses(self):
        r_main, r_side, rho = self.__get_physical_params()
        M, T = self.junc.compute_diagonal_masses(r_main, r_side, rho)
        return M, T

    def get_number_dof(self):
        return 2  # len(self.mass_junction)
    
    def get_contrib_Mh(self):
        # mass_junction = self.__get_masses()[0]
        # my_contrib = mass_junction
        # # Place on our indices
        # Mh = np.zeros(self.ntot_dof,
        #                     dtype='float64')
        # Mh[self.get_indices()] = my_contrib
        # return Mh
        my_contrib = self.__get_masses()[0]
        return self.get_indices(), my_contrib
    
    def get_contrib_Kh(self):
        # interaction_matrix = self.__get_masses()[1]
        # assembled_interaction_matrix = ssp.lil_matrix((self.ntot_dof,
        #                                                self.ntot_dof))
        # for i in range(len(self.ends)):
        #     f_pipe_end = self.ends[i]
        #     interaction = interaction_matrix[:, i]
        #     for k, val in zip(self.get_indices(), np.ravel(interaction)):
        #         assembled_interaction_matrix[k, f_pipe_end.get_index()] = val
        # return assembled_interaction_matrix - assembled_interaction_matrix.T
        interaction_matrix = self.__get_masses()[1]
        row = list()
        col = list()
        data = list()
        for i, f_pipe_end in enumerate(self.ends):
            interaction = interaction_matrix[:, i]
            for k, val in zip(self.get_indices(), np.ravel(interaction)):
                row.append(k)
                col.append(f_pipe_end.get_index())
                data.append(val)
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_a = np.array(data)
        data_tot = np.append(data_a, -1*data_a)
        return row_tot, col_tot, data_tot
        
    def get_contrib_freq(self, omegas_scaled):
        mass_junction = self.__get_masses()[0]
        my_contrib = 1j * omegas_scaled * mass_junction[:, np.newaxis]
        return self.get_indices(), my_contrib

    def get_contrib_indep_freq(self):
        interaction_matrix = self.__get_masses()[1]
        row = list()
        col = list()
        data = list()
        for i, f_pipe_end in enumerate(self.ends):
            interaction = interaction_matrix[:, i]
            for k, val in zip(self.get_indices(), np.ravel(interaction)):
                row.append(k)
                col.append(f_pipe_end.get_index())
                data.append(val)
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_a = np.array(data)
        data_tot = np.append(data_a, -1*data_a)
        return row_tot, col_tot, data_tot


    # ----- differential -----
    def _get_diff_masses(self, diff_index):
        r_main, r_side, rho = self.__get_physical_params()

        d_radii = []
        for end in self.ends:
            d_radius = end.get_diff_radius(diff_index)
            d_radii.append(d_radius)
        assert np.isclose(d_radii[0], d_radii[1])
        d_r_main = sum(d_radii[0:2])/2.0
        d_r_side = d_radii[2]
        dM, dT = self.junc.diff_diagonal_masses(r_main, r_side, rho,
                                                d_r_main, d_r_side)
        return dM, dT

    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        dM, _ = self._get_diff_masses(diff_index)
        local_dAh_diags = 1j * omegas_scaled * dM[:, np.newaxis]
        return self.get_indices(), local_dAh_diags
        # # Place on our indices
        # dAh_diags = np.zeros((self.ntot_dof, len(omegas_scaled)),
        #                      dtype='complex128')
        # dAh_diags[self.get_indices(), :] = local_dAh_diags
        # return dAh_diags
