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
from openwind.discretization import Mesh

class DiscretizedPipe:
    """
    A Pipe, with one possible FEM discretization.

    The wave propagation equations in the pipe are spatially discretize by
    following the Finit Element Method (FEM) the wave propagation equations.
    A mesh is created composed of element(s), on which nodes are placed.

    The acoustic pressure and flow are computed on these nodes, resulting
    on a given number of degree of freedom (dof). One of these quatities is
    :math:`H^1` (by default the pressure) and the nodes are fusioned at the
    boundary of the elements. The other one is :math:`L^2` and the nodes are
    splitted at the boundary  of the elements.

    After discretization, the wave propagation equations can be written as
    a linear system:

    .. math::
        \\begin{pmatrix}
        M_h^{L^2} & -B_h \\\\
        B_h^{\\star} & M_h^{H^1}
        \\end{pmatrix} U_h = L_h

    where :math:`\\star` designates the adjoint and with
    :math:`M_h^{L^2}, M_h^{H^1}` the mass matrices associated to the
    :math:`L^2` and :math:`H^1` variables; :math:`B_h` a matrix associated to
    the gradient computation and :math:`U_h` the vector grouping
    the dof corresponding to the :math:`L^2` then :math:`H^1` variables.
    More details on the discretization process are given in [Tournemenne_Disc]_.

    It is possible to interpolate the value of the acoustics fields between the
    nodes from the base function using for the FEM.


    Parameters
    ----------
    pipe : :py:class:`Pipe<openwind.continuous.pipe.Pipe>`
        The Pipe to be discretized
    kwargs : keyword arguments
        give to :py:class:`Mesh<openwind.discretization.mesh.Mesh>`\
        (for ex: the length of the elements).

    References
    ----------
    .. [Tournemenne_Disc] Tournemenne, Robin, and Juliette Chabassier. 2019. \
        “A Comparison of a One-Dimensional Finite Element Method and the \
        Transfer Matrix Method for the Computation of Wind Music Instrument \
        Impedance.” Acta Acustica United with Acustica 5: 838.

    Attributes
    ----------
    nH1, nL2 : int
        number of dof for the H1 and the L2 variables
    mesh: :py:class:`Mesh<openwind.discretization.mesh.Mesh>`
        The mesh associated to the pipe
    convention : {'PH1', 'VH1'}
        Determine if the H1 variable is the pressure (PH1) pr the flow (VH1)

    """

    def __init__(self, pipe, **kwargs):
        self.pipe = pipe
        self.convention = pipe.get_convention()
        self.mesh = Mesh(pipe, **kwargs)

        self.nL2 = self.mesh.get_nL2()
        self.nH1 = self.mesh.get_nH1()
    
    def is_compatible_for_modal(self):
        return self.pipe.is_compatible_for_modal()
    
    def __repr__(self):
        return "<{}>".format(self.mesh.get_info())

    def __assemble_coefficients(self, coef_pressure, coef_flow):
        """Assemble L2 and H1 coefficients, according to this pipe's convention
        and adimensionalization.
        Select which coefficient corresponds to the H1 variable and to the L2
        variable. Then apply the weighting of each node.
        The coefficients may be multi-dimensional, but their last dimension
        must be of size nL2.

        Parameters
        ----------
        coef_pressure, coef_flow : array-like

        Returns
        -------
        coef_L2, coef_H1 : array-like
            `coef_H1` corresponds to `coef_pressure` if convention is 'PH1'
        """
        assert coef_pressure.shape[-1] in [1, self.mesh.get_nL2()]
        assert coef_flow.shape[-1] in [1, self.mesh.get_nL2()]

        if self.convention == "VH1":
            coef_H1 = coef_flow
            coef_L2 = coef_pressure
        elif self.convention == "PH1":
            coef_H1 = coef_pressure
            coef_L2 = coef_flow
        else:
            raise ValueError("Convention must be 'VH1' or 'PH1'.")
        weights = self.mesh.get_weights()
        # contains GL weights * element_length
        weight_L2 = coef_L2 * weights
        weight_H1 = self.mesh.assemble_H1_from_L2(coef_H1 * weights)
        return weight_L2, weight_H1

    def get_Bh(self):
        """
        Get the Bh matrix in a sparse format

        Returns
        -------
        coo_matrix
        """

        return self.mesh.get_Bh()

    def get_mass_matrices(self):
        """
        Compute the mass matrices without visco-thermal losses

        Returns
        -------
        mass_L2, mass_H1 : np.array
            two vectors containing the diagonal coefficients of the mass matrix
        """
        x_nodes = self.mesh.get_xL2()
        mass_pressure = self.pipe.get_coef_pressure_at(x_nodes)
        mass_flow = self.pipe.get_coef_flow_at(x_nodes)
        return self.__assemble_coefficients(mass_pressure, mass_flow)

    def get_mass_matrices_with_losses(self, omegas_scaled):
        """
        Compute the losses part of the mass matrices

        Returns
        -------
        mass_L2, mass_H1 : np.array
            two vectors containing the diagonal coefficients of the mass matrix
        """
        x_nodes = self.mesh.get_xL2()
        mass_pressure = self.pipe.get_Yt_at(x_nodes, omegas_scaled)
        mass_flow = self.pipe.get_Zv_at(x_nodes, omegas_scaled)
        mass_L2, mass_H1 = self.__assemble_coefficients(mass_pressure,
                                                        mass_flow)
        return np.concatenate((mass_L2.T, mass_H1.T))
    
        
 # --------- Differentiation ---------------
    def get_mass_matrices_dAh(self, diff_index):
        """
        Differentiate the lossless part of the mass matrices wr to one design
        parameters

        Parameters
        ----------
        diff_index : int
            The index of the design parameters in \
            :py:class:`OptimizationParameters<openwind.technical.design_parameter>`.

        Returns
        -------
        np.array
            two vector with the diagonal coefficients of the derivative of the
            mass matrices.

        """
        x_nodes = self.mesh.get_xL2()
        mass_pressure = self.pipe.get_diff_coef_pressure_at(x_nodes,
                                                            diff_index)
        mass_flow = self.pipe.get_diff_coef_flow_at(x_nodes, diff_index)
        return self.__assemble_coefficients(mass_pressure, mass_flow)

    def get_mass_matrices_with_losses_dAh(self, omegas_scaled, diff_index):
        """
        Differentiate the losses part of the mass matrices wr to one design
        parameters

        Parameters
        ----------
        diff_index : int
            The index of the design parameters in \
            :py:class:`OptimizationParameters<openwind.technical.design_parameter>`.

        Returns
        -------
        np.array
            two vector with the diagonal coefficients of the derivative of the
            mass matrices.

        """
        x_nodes = self.mesh.get_xL2()
        mass_pressure = self.pipe.get_diff_Yt_at(x_nodes, omegas_scaled,
                                                 diff_index)
        mass_flow = self.pipe.get_diff_Zv_at(x_nodes, omegas_scaled,
                                             diff_index)
        mass_L2, mass_H1 = self.__assemble_coefficients(mass_pressure,
                                                        mass_flow)
        return np.concatenate((mass_L2.T, mass_H1.T))

# --------------Diffusive reprensation of losses---------------------------
    def get_diffrepr_coefficients(self):
        """Discretize viscothermal coefficients of the pipe,
        and assemble them for use in finite elements simulations.

        Assumes the use of ThermoviscousDiffusiveRepresentation (or analog).

        Returns
        -------
        (r0, ri, li), (g0, gi, c0, ci) : tuple of tuple of arrays
            the diagonal coefficients of the matrices from the model
        """
        x_nodes = self.mesh.get_xL2()
        losses = self.pipe.get_losses()
        viscous_coefs = losses.get_viscous_coefficients_at(self.pipe, x_nodes)
        thermal_coefs = losses.get_thermal_coefficients_at(self.pipe, x_nodes)
        weights = self.mesh.get_weights()
        if self.convention == "VH1":
            assembled_viscous_coefs = [self.mesh.assemble_H1_from_L2(r * weights) for r in viscous_coefs]
            assembled_thermal_coefs = [g * weights for g in thermal_coefs]
        else:
            assembled_viscous_coefs = [r * weights for r in viscous_coefs]
            assembled_thermal_coefs = [self.mesh.assemble_H1_from_L2(g * weights) for g in thermal_coefs]
        return assembled_viscous_coefs, assembled_thermal_coefs

 # --------- Interpolation ---------------
    def get_interp_mat_H1(self, x_interp):
        """
        The matrix to interpolate the H1 variable on a given x vector

        This matrix allows the interpolation of the H1 variable by multiplying
        it with the complet Uh vector.

        Parameters
        ----------
        x_interp : np.array
            The x-vector on which interpolate the H1 variable.

        Returns
        -------
        np.array
            The inteprolation matrix

        """
        return self.mesh.get_interp_mat_H1(x_interp)

    def get_interp_mat_L2(self, x_interp):
        """
        The matrix to interpolate the L2 variable on a given x vector

        This matrix allows the interpolation of the L2 variable by multiplying
        it with the complet Uh vector.

        .. warning::
            The dof being doubled at the element boundaries, the interpolation
            can be discontinuous.

        Parameters
        ----------
        x_interp : np.array
            The x-vector on which interpolate the H1 variable.

        Returns
        -------
        np.array
            The inteprolation matrix

        """
        return self.mesh.get_interp_mat_L2(x_interp)

    def get_interp_mat_L2_continuous(self, x_interp):
        """
        The matrix to interpolate the L2 variable on a given x vector

        This matrix allows the interpolation of the L2 variable by multiplying
        it with the complet Uh vector.

        .. warning::
            The continuity of the L2 variable at the element boundaries is
            guarantee by averaging the values of the two corresponding dof.

        Parameters
        ----------
        x_interp : np.array
            The x-vector on which interpolate the H1 variable.

        Returns
        -------
        np.array
            The inteprolation matrix

        """
        mat_H1 = self.mesh.get_interp_mat_H1(x_interp)
        mat_L2 = mat_H1 @ self.mesh.get_L2_to_H1_matrix()
        return mat_L2

    def get_interp_mat_H1_grad(self, x_interp):
        """
        The matrix to interpolate the gradient of the  H1 variable on a given x
        vector

        This matrix allows the interpolation of the spatial gradient of the H1
        variable by multiplying  it with the complet Uh vector.

        Parameters
        ----------
        x_interp : np.array
            The x-vector on which interpolate the H1 variable.

        Returns
        -------
        np.array
            The inteprolation matrix

        """
        return self.mesh.get_interp_mat_H1(x_interp, compute_grad=True)

    def get_interp_mat_L2_grad(self, x_interp):
        """
        The matrix to interpolate the gradient of the  L2 variable on a given x
        vector

        This matrix allows the interpolation of the spatial gradient of the L2
        variable by multiplying  it with the complet Uh vector.

        Parameters
        ----------
        x_interp : np.array
            The x-vector on which interpolate the L2 variable.

        Returns
        -------
        np.array
            The inteprolation matrix

        """
        return self.mesh.get_interp_mat_L2(x_interp, compute_grad=True)
