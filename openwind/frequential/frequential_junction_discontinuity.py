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

class FrequentialJunctionDiscontinuity(FrequentialComponent):
    """
    Frequential representation of a junction between two pipes with mass in case
    of discontinuity of section.

    Assumes convention PH1. This component contributes only to the matrix
    :math:`A_h` :

    .. code-block:: text

                         ┌                        ┐
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  1  .  . │ ← line of the 1st pipe end's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
           Ah_contrib =  │ .  .  .  .  . -1  .  . │ ← line of the 2nd pipe end's d.o.f.
                         │ . -1  .  .  1 jwm .  . │ ← line of this component's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
                         └                        ┘

    where `jwm`:math:`=j\\omega m`, with :math:`m` the acoustic mass specified
    in :py:class:`JunctionDiscontinuity<openwind.continuous.junction.JunctionDiscontinuity>`.
    :math:`Ah_contrib = jw Mh + Kh`

    Parameters
    ----------
    junc : :py:class:`JunctionDiscontinuity<openwind.continuous.junction.JunctionDiscontinuity>`
        The continuous version of the junction which is converted

    ends : list of 2 :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>`\
        or :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        The pipe ends this junction connects
    """

    def __init__(self, junc, ends):
        self.junc = junc
        assert len(ends) == 2
        self.ends = ends
        if any(end.convention != 'PH1' for end in ends):
            msg = ("FrequentialJunction does not yet support VH1 convention")
            raise ValueError(msg)
            
    def is_compatible_for_modal(self):
        return self.junc.is_compatible_for_modal()
    
    def __get_physical_params(self):
        radii = []
        rhos = []
        for end in self.ends:
            radius, rho, _ = end.get_physical_params()
            radii.append(radius)
            rhos.append(rho)
        assert all(np.isclose(rhos, rho))
        rho = sum(rhos)/2.0
        r1, r2 = radii
        return r1, r2, rho

    def __get_masses(self):
        r1, r2, rho = self.__get_physical_params()
        mass = self.junc.compute_mass(r1, r2, rho)
        return mass

    def get_number_dof(self):
        return 1

    def get_contrib_Mh(self):
        mass_junction = self.__get_masses()
        my_contrib = mass_junction
        return self.get_indices(), my_contrib
    
        

    def get_contrib_Kh(self):
        
        data = np.array([-1, 1])
        row = list()
        col = list()
        for f_pipe_end in self.ends:
            row.append(list(self.get_indices()))
            col.append(f_pipe_end.get_index())
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_tot = np.append(data, -1*data)
        return row_tot, col_tot, data_tot
    
    
    def get_contrib_freq(self, omegas_scaled):
        # a moyen terme appel à Mh et multiplication par jw 
        mass_junction = self.__get_masses()
        my_contrib = 1j * omegas_scaled * mass_junction
        return self.get_indices(), my_contrib

    def get_contrib_indep_freq(self):
        data = np.array([-1, 1])
        row = list()
        col = list()
        for f_pipe_end in self.ends:
            row.append(list(self.get_indices()))
            col.append(f_pipe_end.get_index())
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_tot = np.append(data, -1*data)
        return row_tot, col_tot, data_tot

    # ----- differential -----
    def _get_diff_masses(self, diff_index):
        r1, r2, rho = self.__get_physical_params()

        d_radii = []
        for end in self.ends:
            d_radius = end.get_diff_radius(diff_index)
            d_radii.append(d_radius)
        dmass = self.junc.get_diff_mass(r1, r2, rho, d_radii[0], d_radii[1])
        return dmass

    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        dmass = self._get_diff_masses(diff_index)
        local_dAh_diags = 1j * omegas_scaled * dmass
        return self.get_indices(), local_dAh_diags
        # Place on our indices
        # dAh_diags = np.zeros((self.ntot_dof, len(omegas_scaled)),
        #                      dtype='complex128')
        # dAh_diags[self.get_indices(), :] = local_dAh_diags
        # return dAh_diags
