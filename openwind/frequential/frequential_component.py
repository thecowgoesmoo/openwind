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
Base class for the matrices of linear components and their interactions.
"""

from abc import ABC, abstractmethod
from scipy.sparse import csr_matrix, bmat

import numpy as np

# Abstract Base Class
class FrequentialComponent(ABC):
    """
    A component of an instrument with linear behavior.

    This is the frequential version of the netlist component (
    :py:class:`Pipe<openwind.continuous.pipe.Pipe>` and
    :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`)

    Computing the impedance of a complex instrument with
    the Finite Elements Method or with the Transfer Matrix Method
    amounts to solving a rather large system of equations,
    given by the fundamental equations of each component
    of the instrument, and the way they are connected:

    .. math::

        Ah.Uh = Lh

    where Ah is the sum of the matrix contributions of all components,
    and Lh is the sum of the source contributions of all components.

    Each :py:class:`FrequentialComponent` may have internal variables, also
    called "degrees of freedom" or `dof`,
    corresponding to coefficients in the unknown vector `Uh`.

    For performance reasons, its contribution to Ah is split into
    a frequency-independent part and a frequency-dependent part.
    The latter is assumed to lie on the diagonal of Ah.
    """

    def is_compatible_for_modal(self):
        """
        Is this component compatible for modal computation?

        Returns
        -------
        boolean
        """
        return False

    # --------------------------------------------------------
    # These methods must be overridden by implementing classes
    # --------------------------------------------------------

    @abstractmethod
    def get_number_dof(self):
        """Number of degrees of freedom added by this component.

        The number of equations is the total number of degrees of freedom.

        Returns
        -------
        int
        """

    def get_contrib_indep_freq(self):
        """
        Contribution of this component to the frequency-independent terms of Ah.

        Returns
        -------
        row_index : list of int
            The row indices

        col_index: list of int
            The column indices

        values : list of complex float
            The values
        """
        return [], [], []

    def get_contrib_Mh(self):
        """
        Contribution of this component to the diagonal of the matrix Mh.

        Parameters
        ----------
        none

        Returns
        -------
        row_index : list of int
            The diagonal indices

        values : list of complex float
            The values
        """
        return [], []

    def get_contrib_Kh(self):
        """
        Contribution of this component to the matrix Kh.

        Parameters
        ----------
        none

        Returns
        -------
        row_index : list of int
            The row indices

        col_index: list of int
            The column indices

        values : list of complex float
            The values
        """
        return [], [], []

    def get_contrib_freq(self, omegas_scaled):
        """
        Contribution of this component to the frequency-dependent diagonal of Ah.

        Parameters
        ----------
        omegas_scaled : array of float
            angular frequncies scaled thanks to :py:class:`Scaling\
            <openwind.continuous.scaling.Scaling>`

        Returns
        -------
        row_index : indices
            The row indices

        values : array (n_indices, len(omegas))
            The values for each freq
        """
        return [],  np.zeros((0,len(omegas_scaled)))

    def get_contrib_source(self):
        """
        Contribution of this component to the right-hand side Lh.

        Returns
        -------
        row_index : list of integer
            The row indices

        values : list of complex float
            The values
        """
        return [], []

    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        """
        Differentiation of the contribution of this component to the
        frequency-dependent diagonal of Ah with respect to one design parameter

        Parameters
        ----------
        omegas_scaled : array of float
            angular frequncies scaled thanks to :py:class:`Scaling\
            <openwind.continuous.scaling.Scaling>`
        diff_index : int
            The index of the design parameter to which differentiate, such as
            stocked in :py:class:`OptimizationParameters\
            <openwind.design.design_parameter.OptimizationParameters>`

        Returns
        -------
        row_index : indices
            The row indices

        values : array (n_indices, len(omegas))
            The values for each freq
        """
        return [],  np.zeros((0,len(omegas_scaled)))

    def get_contrib_dAh_indep_freq(self, diff_index):
        """
        Differentiation of the contribution of this component to the
        frequency-independent terms of Ah.

        Parameters
        ----------
        diff_index : int
            The index of the design parameter to which differentiate, such as
            stocked in :py:class:`OptimizationParameters\
            <openwind.design.design_parameter.OptimizationParameters>`

        Returns
        -------
        row_index : list of int
            The row indices

        col_index: list of int
            The column indices

        values : list of complex float
            The values
        """
        return [], [], []

    # ---------------------------------------------------
    # The following functions do not need to be overriden
    # ---------------------------------------------------

    def set_total_degrees_of_freedom(self, ntot_dof):
        """
        Update the number of dof in the total problem (entire graph)

        Parameters
        ----------
        ntot_dof : int
            The new number of dof.

        """
        self.ntot_dof = ntot_dof

    def set_first_index(self, ind_1st):
        """
        Set the value of the first index corresponding to this component along
        the dof of the total problem (entire graph)

        Parameters
        ----------
        ind_1st : int
            the value of the index.

        """
        self.ind_1st = ind_1st

    def get_first_index(self):
        """
        The value of the first index corresponding to this component along
        the dof of the total problem (entire graph)

        Returns
        -------
        int

        """
        return self.ind_1st

    def get_indices(self):
        """
        All the indices corresponding to this component along the dof of the
        total problem (entire graph)


        Returns
        -------
        range

        """
        i0 = self.get_first_index()
        return range(i0, i0 + self.get_number_dof())

    def place_in_big_matrix(self, local_Ah):
        """
        Build the contribution to the overall matrix Ah, assuming it is
        locally given by local_Ah.

        It is used by components for which the contribution is a localized
        block, e.g. pipes.

        Returns
        -------
        row_index : list of integer
            The row indices

        col_index: list of integer
            The column indices

        values : list of complex float
            The values

        """
        assert local_Ah.shape == (self.get_number_dof(),)*2
        row = local_Ah.row + self.get_first_index()
        col = local_Ah.col + self.get_first_index()
        data = local_Ah.data
        return row, col, data


    def place_in_big_matrix_float(self, local_Ah):
        """
        Build the contribution to the overall matrix Ah, assuming it is
        locally given by local_Ah.

        It is used by components for which the contribution is a localized
        block, e.g. pipes.

        Returns
        -------
        sparse matrix of size (ntot_dof x ntot_dof)

        """
        assert local_Ah.shape == (self.get_number_dof(),)*2
        n_left = self.get_first_index()
        Ah_left = csr_matrix((n_left, n_left))
        n_right = self.ntot_dof - self.get_indices().stop
        Ah_right = csr_matrix((n_right, n_right))
        Ah = bmat([[Ah_left, None, None], [None, local_Ah, None],
                   [None, None, Ah_right]], dtype='float64')
        return Ah

    def __repr__(self):
        classname = type(self).__name__
        classname = classname.replace('Frequential','F')
        msg = "<{}>".format(classname, id(self))
        return msg
