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

from openwind.discretization import Element
import numpy as np
import scipy.sparse as sp


def _swapaxes(data, axis):
    """Put axis `axis` in last position."""
    if axis in [-1, len(data.shape) - 1]: # No need to swap
        return data

    if sp.issparse(data):
        if axis == 0:
            return data.T
        else:
            raise ValueError("Invalid axis number for sparse matrix")
    else:
        return np.swapaxes(data, axis, -1)



class Mesh:
    """Calculate the mesh and the nodes for the quadrature of a given Pipe.

    Parameters
    ----------
    pipe : :py:class:`Pipe<openwind.continuous.pipe.Pipe>`
        The section of tubing to mesh.
    l_ele : float, optional
        The length of a mesh element. If None, it is chosen automatically.
    order : int, optional
        The order of quadrature on each element. If None, it is chosen
        automatically (adaptative meshing).
    shortestLbd : float, optional
        The smallest wavelength used. Default is 0.17, which corresponds
        approximately to 2kHz.
    dofPerLbd : int, optional
        The number of degrees of freedom per wavelength. Default is 10.

    Attributes
    ----------
    elements : list of :py:class:`Element<openwind.discretization.element.Element>`
        The element constituing the mesh.
    """
    ORDER_MIN = 4   # Can be changed by setting Mesh.ORDER_MIN = ...
    """ The minimal order authorized for automatic mesh."""
    ORDER_MAX = 10
    """ The maximal order authorized for automatic mesh."""

    def __init__(self, pipe, l_ele=None, order=None, shortestLbd=0.17,
                 dofPerLbd=10):
        if pipe.get_length() <= 0:
            raise ValueError('The pipe length of the {} is not positive:'
                             '{:.2e} m'.format(pipe.label, pipe.get_length()))
        if l_ele is not None:
            l_ele_norm = np.array(l_ele)/pipe.get_length()
        else:
            l_ele_norm = l_ele
        nb_nodes = pipe.get_length() / shortestLbd * dofPerLbd
        if isinstance(l_ele, list):
            assert isinstance(order, (list, int))
            self.__mesh_from_lists(l_ele_norm, order)
        else:
            self.__mesh_auto(l_ele_norm, order, nb_nodes)

        self.update_dof()

#    def has_fixed_order(self):
#        return self._fixed_order is not None
#
#    def get_fixed_order(self):
#        assert self.has_fixed_order()
#        return self._fixed_order

# %% Mesh imposed by lists
    def __mesh_from_lists(self, l_ele_norm, order):
        if isinstance(order, int):  # Make sure order is a list
            order = [order] * len(l_ele_norm)
        x_eles = np.cumsum(np.append(0, l_ele_norm))
        assert np.isclose(x_eles[-1], 1)
        orders = np.array(order, dtype=int)
        self.__create_elements(x_eles, orders)

# %% Mesh build with fixed length or automaticly and fixed order or automatic
    def __mesh_auto(self, l_ele_norm, order, nb_nodes):
        self._fixed_l_ele = l_ele_norm
        self._fixed_order = order
        self.__set_order_bounds()
        orders = self.__mesh_pipe(nb_nodes)
        assert order is None or np.all(orders == order)
        x_eles = np.linspace(0, 1, (len(orders)+1))

        self.__create_elements(x_eles, orders)

    def __set_order_bounds(self):
        if self._fixed_order is None:
            self._order_min = Mesh.ORDER_MIN
            self._order_max = Mesh.ORDER_MAX
        else:
            self._order_min = self._fixed_order
            self._order_max = self._fixed_order

    def __mesh_pipe(self, nb_nodes):
        nb_elements = self.__nb_elements_pipe(nb_nodes)
        order = self.__order_of_eles(nb_nodes, nb_elements)
        return order*np.ones(nb_elements, dtype=int)

    def __nb_elements_pipe(self, nb_nodes):
        if self._fixed_l_ele is not None:
            # Allow elements to be slightly longer (for float error)
            tol = 1e-10
            longest_allowed = self._fixed_l_ele + tol
            return int(max(1, np.ceil(1 / longest_allowed)))
        else:
            return int(np.ceil(nb_nodes / self._order_max))

    def __order_of_eles(self, nb_nodes, nb_elements):
        if self._fixed_order is not None:
            return self._fixed_order*np.ones(nb_elements, dtype=int)
        else:
            return int(max(self._order_min,
                           min(self._order_max,
                               np.ceil(nb_nodes/nb_elements))))

# %% create elements

    def __create_elements(self, x_eles, orders):
        self.elements = []
        for k in range(len(orders)):
            self.elements.append(Element(x_eles[k], x_eles[k+1], orders[k]))

    def update_dof(self):
        """
        Set or update the number of dof on the mesh after a modification.
        """
        self.__set_nH1()
        self.__set_nL2()
        self.__organize_dof()

    def __set_nH1(self):
        self._nH1 = int(self.get_orders().sum() + 1)

    def __set_nL2(self):
        self._nL2 = int((self.get_orders() + 1).sum())

    def __organize_dof(self):
        self._ddlH1 = []
        self._ddlL2 = []
        self._assembling_matrix = sp.lil_matrix((self._nL2, self._nH1))
        for k in range(len(self.elements)):
            curR = self.elements[k].order
            if k == 0:
                ddlH1 = np.arange(0, curR + 1)
                ddlL2 = np.arange(0, curR + 1)
            else:
                ddlH1 = np.arange(ddlH1[-1], ddlH1[-1] + curR + 1)
                ddlL2 = np.arange(ddlL2[-1] + 1, ddlL2[-1] + 1 + curR + 1)
            self._ddlH1.append(ddlH1)
            self._ddlL2.append(ddlL2)
            self._assembling_matrix[ddlL2, ddlH1] = 1
        self._assembling_matrix = self._assembling_matrix.tocsc()

        # How many L2 dof contribute to each H1 dof
        nb_of_repeats = np.sum(self._assembling_matrix, axis=0).A.ravel()
        self._assembling_matrix_mean = (self._assembling_matrix @
                                        sp.diags(1/nb_of_repeats))

    def get_orders(self):
        """
        The order of each elements of the mesh

        Returns
        -------
        list of int
        """
        return np.array([element.order for element in self.elements])

    def get_lengths(self):
        """
        The length of each elements of the mesh

        Returns
        -------
        list of float
        """
        return np.array([element.get_length() for element in self.elements])

    def get_xL2(self):
        """
        The location of the nodes associated to the L2 variable.

        Returns
        -------
        list of float
        """
        return np.concatenate([elem.get_nodes() for elem in self.elements])

    def get_xH1(self):
        """
        The location of the nodes associated to the H1 variable.

        Returns
        -------
        list of float
        """
        return self.trim_H1_from_L2(self.get_xL2())

    def get_weights(self):
        """
        The weight associated to each node.

        Returns
        -------
        list of float
        """
        return np.concatenate([elem.get_weights() for elem in self.elements])

    def get_nL2(self):
        """
        The dof number associated to the L2 variable

        Returns
        -------
        int
        """
        return self._nL2

    def get_nH1(self):
        """
        The dof number associated to the H1 variable

        Returns
        -------
        int
        """
        return self._nH1

    def get_Bh(self):
        """
        Assemble sparse matrix Bh.

        Bh contains the dot product of L2 basis functions against the gradient
        of H1 basis functions.

        Return
        ------
         sparse matrix in csr format
        """
        blocks = []
        for elem in self.elements:
            blocks.append(elem.get_Bh_coeff())
        Bh_L2 = sp.block_diag(blocks, format='lil')
        return self.assemble_H1_from_L2(Bh_L2, axis=-1).tocsr()

    def __repr__(self):
        s = "<openwind.discretization.Mesh, {}>"
        return s.format(self.get_info())

    def get_info(self):
        """
        Representation of the range of orders of this mesh.

        Returns
        -------
        str
        """
        orders = self.get_orders()
        if min(orders) < max(orders):
            order_info = "{}<=order<={}".format(min(orders), max(orders))
        else:
            order_info = "order={}".format(orders[0])
        return "nH1={}, nL2={}, {}".format(self._nH1, self._nL2, order_info)

# %% play with H1 and L2
    def assemble_H1_from_L2(self, data, axis=-1, operation='sum'):
        """
        Collapse L2 data to H1 by summing on duplicate degrees of freedom.

        To be used when assembling finite-element matrices.

        If `data` is a sparse matrix, the result is also a sparse matrix.

        Parameters
        ----------
        data : array-like
            The data to collapse; the size of data along`axis` should
            initially be `nL2`.
            On degrees of freedom shared by two elements, the data are summed,
            so that `axis` is shrinked to size `nH1`.
        axis : int, optional
            The axis along which to collapse. Default is -1 (last axis).
            Other values are supported only for numpy arrays (not sparse matrices).
        operation : {'sum', 'trim', 'mean'}
            What operation to perform on duplicate degrees of freedom.
            'sum' takes the sum of the data.
            'trim' discards duplicated data. If the data had two different
            values at a duplicated degree of freedom, it may take either.
            'mean' takes the mean (average) of the data.

        Returns
        -------
        data_H1 : array-like
            Assembled version of `data`.
            Only the shape along `axis` is changed from `nL2` to `nH1`.
        """
        # Check that the old shape has size nL2 on the axis to collapse
        if data.shape[axis] != self._nL2:
            msg = ("Data with shape %s cannot be "
                  "assembled along axis %d (size = %d != nL2 = %d).")
            msg = msg % (str(data.shape), axis, data.shape[axis], self._nL2)
            raise ValueError(msg)

        # Put the axis to collapse in last position
        data = _swapaxes(data, axis)

        new_shape = data.shape[:-1] + (self._nH1,)

        if operation == 'sum':
            data_H1 = data @ self._assembling_matrix
        elif operation == 'mean':
            data_H1 = data @ self._assembling_matrix_mean
        elif operation == 'trim':
            # Could be optimized ?
            data_H1 = np.zeros(new_shape, dtype=data.dtype)
            for k in range(len(self.elements)):
                data_H1[..., self._ddlH1[k]] = data[..., self._ddlL2[k]]
        assert data_H1.shape == new_shape
        # Swap axis back to its position
        data_H1 = _swapaxes(data_H1, axis)
        return data_H1

    def get_L2_to_H1_matrix(self):
        """Matrix used to convert L2 data to H1 by averaging.

        Returns
        -------
        (nH1, nL2) sparse matrix
            It which averages the repeated degrees of freedom.
        """
        return self._assembling_matrix_mean.T

    def trim_H1_from_L2(self, data, axis=-1):
        """Collapse L2 data to H1 by deleting data on duplicate degrees of freedom.

        See Also
        --------
        assemble_H1_from_L2
        """
        return self.assemble_H1_from_L2(data, axis=axis, operation='trim')


# %% Interpolation

    def get_interp_mat_L2(self, x_interp, compute_grad=False):
        """Compute the interpolation matrix associated to one pipe, for the L2
        variable.

        The final matrix shape is (len(x_interp), nL2). Only the
        interpolation points on the considered shape must be given.
        It allows also to compute the matrix to interpolate the gradient of a
        variable.

        Parameters
        ----------
        x_interp : nparray
            Normalized location of the observation point along the pipe. Should
            be in the range [0,1]
        compute_grad : bool
            if True the interpolation matrix of the gradient of the variable
            is computed.

        """

        if compute_grad:
            get_block = lambda elem, x:elem.get_diff_lagrange(x)
        else:
            get_block = lambda elem, x:elem.get_lagrange(x)

        eps = 1e-10
        if not (np.all(x_interp>=-eps) and np.all(x_interp<=1+eps)):
            raise ValueError("Some interpolation points are outside the "
                             "considered shape: {}.".format(x_interp))

        blocks = []
        ind_elements = []
        for elem in self.elements:
            x_norm = (x_interp - elem._x_start)/elem._length
            ind_in  = (x_norm >= 0) & (x_norm <= 1)
            blocks.append(get_block(elem, x_norm[ind_in]))
            ind_elements.append(np.nonzero(ind_in)[0])
        interp_mat =  sp.block_diag(blocks)
        ind_duplicate = np.concatenate(ind_elements)

        n_keep = len(x_interp)
        n_interp_duplicate = len(ind_duplicate)
        if n_interp_duplicate > n_keep:
            duplicate_to_interp = sp.dok_matrix((n_keep, n_interp_duplicate))
            for j in range(n_keep):
                i = (ind_duplicate==j)
                duplicate_to_interp[j, i] = 1/sum(i)
            duplicate_to_interp.tocoo()
        else:
            duplicate_to_interp = sp.dia_matrix((np.ones(n_keep), 0),
                                        shape=(n_keep, n_keep))
        return duplicate_to_interp.dot(interp_mat)


    def get_interp_mat_H1(self, x_interp, compute_grad=False):
        """Compute the interpolation matrix associated to one pipe, for the H1
        variable.

        The final matrix shape is (len(x_interp), nH1). Only the
        interpolation points on the considered shape must be given.
        It allows also to compute the matrix to interpolate the gradient of a
        variable.

        Parameters
        ----------
        x_interp : nparray
            Normalized location of the observation point along the pipe. Should
            be in the range [0,1]
        compute_grad : bool
            if True the interpolation matrix of the gradient of the variable
            is computed.

        """
        interp_mat_L2 = self.get_interp_mat_L2(x_interp, compute_grad=compute_grad)
        return interp_mat_L2.dot(self._assembling_matrix)
