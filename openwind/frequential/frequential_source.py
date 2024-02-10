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
from openwind.continuous import RadiationPade


class FrequentialSource(FrequentialComponent):
    """
    Computes the source terme Lh for the linear system to solve

    .. math::

        Ah.Uh = Lh

    This component contributes only to the :math:`L_h` matrix with:

    .. code::

                         ┌   ┐
                         │ . │
                         │ . │
            Lh_contrib = │ . │
                         │ 1 │ ← line of the pipe end's d.o.f H1 variable
                         │ . │
                         └   ┘


    Parameters
    ----------
    source : :py:class:`Excitator <openwind.continuous.excitator.Excitator>`
        Excitator. Must be :py:class:`Flow <openwind.continuous.excitator.Flow>`.
    ends : list of :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>`
        Frequential Pipe end associated to this source condition.

    """

    def __init__(self, source, ends):
        self.end, = ends  # Unpack one
        self.source = source

    def is_compatible_for_modal(self):
        return True

    def is_flute_like(self):
        """Is this source a flute-like source?

        useful for impedance computation (Z=p ou Z=1/u)
        """
        return False

    def get_scaling(self):
        """
        Return the scaling associated to the source

        Returns
        -------
        :py:class:`Scaling<openwind.continuous.scaling.Scaling>`

        """
        return self.source.scaling

    def get_convention(self):
        """
        The convention at the source end

        Returns
        -------
        {'PH1', 'VH1'}

        """
        return self.end.convention

    def get_number_dof(self):
        return 0

    def get_contrib_source(self):
        return [self.get_source_index()], [1]

    def get_source_index(self):
        """
        Get index where this source brings a nonzero term.

        Returns
        -------
        int
        """
        return self.end.get_index()

    def get_Zc0(self):
        """
        Return the real characteristic impedance at the source end

        .. math::
            Z_c = \\frac{\\rho c}{S}

        Returns
        -------
        float

        """
        radius, rho, c = self.end.get_physical_params()
        return rho*c/(np.pi*radius**2)


class FrequentialFluteSource(FrequentialComponent):
    """
    Compute for every frequency the radiation data for the linear system to
    solve and the source term

    .. math::

        A_h U_h = L_h

    This component contributes only to the matrix :math:`A_h` :

    .. code-block:: text

                         ┌                        ┐
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  1  .  . │ ← line of the pipe end's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
           Ah_contrib =  │ .  .  .  . Yr -1  .  . │ ← line of the radiation d.o.f.
                         │ . -1  .  .  1  .  .  . │ ← line of this component's d.o.f.
                         │ .  .  .  .  .  .  .  . │
                         │ .  .  .  .  .  .  .  . │
                         └                        ┘

                         ┌   ┐
                         │ . │
            Lh_contrib = │ . │
                         │ 1 │ ← line of this component's d.o.f.
                         │ . │
                         └   ┘


    with `Yr`:math:`=Y_r` the radiation admittance if 'PH1' convention,
    or :math:`Z_r` the radiation impedance if 'VH1' convention. Their
    expressions depend on the :py:class:`PhysicalRadiation\
    <openwind.continuous.physical_radiation.PhysicalRadiation>` given.


    Parameters
    ----------
    source: :py:class:`Flute <openwind.continuous.excitator.Flute>`
        The flute model in continous time domain.

    rad : :py:class:`PhysicalRadiation <openwind.continuous.physical_radiation.PhysicalRadiation>`
        the PhysicalRadiation model used

    freq_ends : tuple of one :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>` or \
        :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        Frequential Pipe end associated to this radiation condition.
    opening_factor: float, optional
        The ratio of opening of the radiating opening (1: open, 0: closed).
        Defaults to 1.0

    """

    def __init__(self, source, freq_ends):
        self.freq_end = freq_ends[0]  # Unpack one
        self.source = source
        Sp = np.pi * self.__get_physical_params()[0]**2
        self.rad = source.get_rad_model_window(Sp)
        self._opening_factor = 1

    def is_flute_like(self):
        """Is this source a flute-like source?

        useful for impedance computation (Z=p ou Z=1/u)
        """
        return True

    def get_scaling(self):
        """
        Return the scaling associated to the source

        Returns
        -------
        :py:class:`Scaling<openwind.continuous.scaling.Scaling>`

        """
        return self.source.scaling

    def get_convention(self):
        """
        The convention at the source end

        Returns
        -------
        {'PH1', 'VH1'}

        """
        return self.freq_end.convention

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
        if opening_factor<1e-8:
            raise ValueError('The flute embouchure hole (window) can not bel fully closed.')
        self._opening_factor = opening_factor

    def __get_physical_params(self):
        return self.freq_end.get_physical_params()

    def get_number_dof(self):
        return 2

    def get_contrib_Kh(self):
        # assembled_interaction_matrix = ssp.lil_matrix((self.ntot_dof,
        #                                                self.ntot_dof))
        # interaction = [-1, 1]
        # for i, f_pipe_end in enumerate(self.ends):
        #     assembled_interaction_matrix[self.get_indices(),
        #                                  f_pipe_end.get_index()] = interaction[i]
        # return assembled_interaction_matrix - assembled_interaction_matrix.T
        return self.get_contrib_indep_freq()

    def get_contrib_indep_freq(self):
        row = list()
        col = list()
        data = np.array([-1, 1])
        ind_p_rad = self.get_indices()[0]
        row = [self.freq_end.get_index(), ind_p_rad]
        col = [self.get_indices()[1], self.get_indices()[1]]
        # return Matrice - Matrice.T
        row_tot = np.append(row, col)
        col_tot = np.append(col, row)
        data_tot = np.append(data, -1*data)
        return row_tot, col_tot, data_tot

    def get_contrib_freq(self, omegas_scaled):
        rho, c = self.__get_physical_params()[1:]
        radius = self.source.get_equivalent_radius()
        coef_rad = self.rad.get_radiation_at(omegas_scaled, radius, rho, c,
                                             self._opening_factor)
        return self.get_indices()[0], coef_rad


    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        raise NotImplementedError()
        # radius, rho, c = self.__get_physical_params()
        # dr = self.freq_end.get_diff_radius(diff_index)
        # local_dAh_diags = self.rad.get_diff_radiation_at(dr, omegas_scaled,
        #                                                  radius, rho, c,
        #                                                  self._opening_factor)
        # return self.get_indices()[0], local_dAh_diags

    def get_contrib_source(self):
        return [self.get_source_index()], [1]

    def get_source_index(self):
        """
        Get index where this source brings a nonzero term.

        Returns
        -------
        int
        """
        return self.get_indices()[1]

    def get_Zc0(self):
        """
        Return the real characteristic impedance at the source end

        .. math::
            Z_c = \\frac{\\rho c}{S}

        Returns
        -------
        float

        """
        rho, c = self.__get_physical_params()[1:]
        radius = self.source.get_equivalent_radius()
        return rho*c/(np.pi*radius**2)
