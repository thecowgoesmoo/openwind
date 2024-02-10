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


class FrequentialRadiation(FrequentialComponent):
    """
    Compute for every frequency the radiation data for the linear system to
    solve

    .. math::

        A_h U_h = L_h

    This component contributes only to the matrix :math:`A_h` :

    .. code-block:: text

                         ┌               ┐
                         │ .  .  .  .  . │
                         │ .  .  .  .  . │
            Ah_contrib = │ .  .  .  Yr . │ ← line of the pipe end's d.o.f.
                         │ .  .  .  .  . │
                         │ .  .  .  .  . │
                         └               ┘

    with `Yr`:math:`=Y_r` the radiation admittance if 'PH1' convention,
    or :math:`Z_r` the radiation impedance if 'VH1' convention. Their
    expressions depend on the :py:class:`PhysicalRadiation\
    <openwind.continuous.physical_radiation.PhysicalRadiation>` given.


    Parameters
    ----------

    rad : :py:class:`PhysicalRadiation <openwind.continuous.physical_radiation.PhysicalRadiation>`
        the PhysicalRadiation model used

    freq_ends : tuple of one :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>` or \
        :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        Frequential Pipe end associated to this radiation condition.
    opening_factor: float, optional
        The ratio of opening of the radiating opening (1: open, 0: closed).
        Defaults to 1.0

    """

    def __init__(self, rad, freq_ends, opening_factor=1.0):
        self.freq_end = freq_ends[0]  # Unpack one
        self.rad = rad
        self._opening_factor = opening_factor

    def is_compatible_for_modal(self):
        return self.rad.is_compatible_for_modal()
    
    def set_opening_factor(self, opening_factor):
        """
        Update the opening factor of the opening

        Parameters
        ----------
        opening_factor: float
            The new ratio of opening (1: open, 0: closed).

        """
        self._opening_factor = opening_factor

    def __get_physical_params(self):
        return self.freq_end.get_physical_params()

    def get_number_dof(self):
        return 0

    def get_contrib_freq(self, omegas_scaled):
        radius, rho, c = self.__get_physical_params()
        coef_rad = self.rad.get_radiation_at(omegas_scaled, radius, rho, c,
                                             self._opening_factor)
        return self.freq_end.get_index(), coef_rad


    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        radius, rho, c = self.__get_physical_params()
        dr = self.freq_end.get_diff_radius(diff_index)
        local_dAh_diags = self.rad.get_diff_radiation_at(dr, omegas_scaled,
                                                         radius, rho, c,
                                                         self._opening_factor)
        return self.freq_end.get_index(), local_dAh_diags
        # dAh_diags = np.zeros((self.ntot_dof, len(omegas_scaled)),
        #                      dtype='complex128')
        # dAh_diags[self.freq_end.get_index(), :] = local_dAh_diags
        # return dAh_diags
