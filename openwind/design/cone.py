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
Pipe defined as a conical frustum.
"""
import numpy as np

from openwind.design import DesignShape, eval_, diff_

def linear(x, x1, x2, r1, r2):
    """
    Calculate linear function from 2 points coordinates.

    .. math::
        r(x) = r_1 + \\frac{x - x_1}{x_2 - x_1}  (r_2 - r_1) \


    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point coordinates
    x2, r2 : float
        the second point coordinates
    """
    r = r1 + (x - x1) * (r2 - r1)/(x2 - x1)
    return r

def linear_diff_x(x, x1, x2, r1, r2):
    """
    Differentiate the linear function w.r. to the abscissa.

    .. math::
        \\frac{\\partial r(x)}{\\partial x} = \\frac{r_2 - r_1}{x_2 - x_1} \


    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point coordinates
    x2, r2 : float
        the second point coordinates
    """
    y = (r2 - r1) / (x2 - x1) * np.ones_like(x)
    return y


def linear_diff_x1(x, x1, x2, r1, r2):
    """
    Differentiate the linear function w.r. to the first enpoint abscissa.

    .. math::
        \\frac{\\partial r(x)}{\\partial x_1} \


    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point coordinates
    x2, r2 : float
        the second point coordinates
    """
    y = (r2 - r1) * (x - x2)/((x2 - x1)**2)
    return y


def linear_diff_x2(x, x1, x2, r1, r2):
    """
    Differentiate the linear function w.r. to the second enpoint abscissa.

    .. math::
        \\frac{\\partial r(x)}{\\partial x_2} \


    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point coordinates
    x2, r2 : float
        the second point coordinates
    """
    y = (r2 - r1) * (x1 - x)/((x2 - x1)**2)
    return y


class Cone(DesignShape):
    """
    Pipe defined as a conical frustum.

    The radius of this pipe follows a linear evolution between the two
    endpoints.

    .. math::
        r(x) = r_1 + \\frac{x - x_1}{x_2 - x_1}  (r_2 - r_1)

    with

    - :math:`x_1, x_2`  the endpoints positions
    - :math:`r_1, r_2`  the endpoints radii

    Parameters
    ----------
    *params : 4 openwind.design.design_parameter.DesignParameter
        The four parameter necessary to describe the shape in this order:
        :math:`x_1, x_2, r_1, r_2`
    """

    def __init__(self, *params, label=None):
        if len(params) != 4:
            raise ValueError("A conical furstum needs 4 parameters : x1, x2, r1, r2.")
        self.params = params
        self.label = label

    def is_TMM_compatible(self):
        return True

    def is_cylinder(self):
        """Is this a straight tube?

        Only evaluated for the current value of the parameters. May change
        during an optimization.

        The analytical solution of the propagation equation with losses in a
        cylindrical pipe being known, this method can be used to treat
        differently cylinders and conical frustums in hybrid (FEM, TMM)
        frequential computation.

        Returns
        -------
        bool

        """
        x1, x2, r1, r2 = eval_(self.params)
        return r1 == r2

    def get_radius_at(self, x_norm):
        x1, x2, r1, r2 = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        radius = linear(x, x1, x2, r1, r2)
        self.check_bounds(x, [x1, x2])
        return radius

    def get_diff_radius_at(self, x_norm, diff_index):
        x1, x2, r1, r2 = eval_(self.params)
        dx1, dx2, dr1, dr2 = diff_(self.params, diff_index)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)
        x = self.get_position_from_xnorm(x_norm)
        diff_radius = linear_diff_x(x, x1, x2, r1, r2)*dx_norm
        if dx1 != 0:
            diff_radius += dx1*linear_diff_x1(x, x1, x2, r1, r2)
        if dx2 != 0:
            diff_radius += dx2*linear_diff_x2(x, x1, x2, r1, r2)
        if dr1 != 0:
            diff_radius += dr1 * linear(x, x1, x2, 1, 0)
        if dr2 != 0:
            diff_radius += dr2 * linear(x, x1, x2, 0, 1)
        self.check_bounds(x, [x1, x2])
        return diff_radius

    def get_conicity_at(self, x_norm):
        x1, x2, r1, r2 = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        return linear_diff_x(x, x1, x2, r1, r2)

    def diff_conicity_wr_xnorm(self, x_norm):
        return 0.

    def get_diff_conicity_at(self, x_norm, diff_index):
        x1, x2, r1, r2 = eval_(self.params)
        dx1, dx2, dr1, dr2 = diff_(self.params, diff_index)
        return np.ones_like(x_norm)*((dr2- dr1)/(x2 - x1)
                                     - (r2 - r1)*(dx2 - dx1)/(x2 - x1)**2)


    def get_endpoints_position(self):
        return self.params[0], self.params[1]

    def get_endpoints_radius(self):
        return self.params[2], self.params[3]

    def get_conicity_jacobian(self):
        var_param = [p for p in self.params if p.is_variable()]
        if len(var_param)>0:
            Nactive = sum(var_param[0]._optim_params.active)
            J = np.zeros(Nactive)
            for ind in range(Nactive):
                J[ind] = self.get_diff_conicity_at(0, ind)
            return J
        else:
            return 0

    def create_conicity_constraint(self, Cmin=-np.inf, Cmax=np.inf, keep_constant=False):
        if keep_constant:
            C = self.get_conicity_at(0)
            Cmin = C
            Cmax = C
        var_param = [p for p in self.params if p.is_variable()]
        if len(var_param)>0:
            optim_params = var_param[0]._optim_params
            optim_params.add_nonlinear_constraint(lambda : self.get_conicity_at(0),
                                                  self.get_conicity_jacobian,
                                                  Cmin,
                                                  Cmax,
                                                  label='Conicity ' + self.label)
