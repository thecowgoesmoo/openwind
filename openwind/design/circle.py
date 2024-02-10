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
Pipe the radius of which follows a circle equation: constant curvature.
"""
import numpy as np

from openwind.design import DesignShape, eval_, diff_


def compute_center_coor(x1, x2, y1, y2, R):
    """
    Determine the center coordinates of the circle.

    The circle is defined from two points and its radius.

    Parameters
    ----------
    x1, x2, y1, y2 : floats
        Coordinates of the two points.
    R : float
        Radius of the circle.

    Returns
    -------
    Xc, Yc : floats
        Coordinates of the center.

    """
    error_msg = ('The diameter of the circle must be larger than the length of'
                 ' the pipe. Here: {}<{}'.format(2*np.abs(R), np.abs(x2 - x1)))
    assert (2*np.abs(R) > np.abs(x2 - x1)), error_msg

    Deltax = x2 - x1
    Deltay = y2 - y1
    if Deltay != 0:
        u = x1**2 - x2**2 + y1**2 - y2**2

        a = 1 + (Deltax**2) / (Deltay**2)
        b = 2 * y1 * Deltax / Deltay - 2 * x1 + u * Deltax / (Deltay**2)
        c = u**2 / (4 * Deltay**2) + y1 * u / Deltay + y1**2 + x1**2 - R**2

        Xc1 = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        Xc2 = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        Yc1 = -(u + 2 * Deltax * Xc1) / (2 * Deltay)
        Yc2 = -(u + 2 * Deltax * Xc2) / (2 * Deltay)
        if np.sign(R)*Yc1 >= np.sign(R)*max(y1, y2):
            Xc = Xc1
            Yc = Yc1
        else:
            Xc = Xc2
            Yc = Yc2
    else:  # If Deltay = 0 it is a particular case not to divide by 0
        Xc = 0.5 * (x1 + x2)
        Yc = y1 + np.sign(R)*np.sqrt(R**2 - 0.25 * (x2 - x1)**2)
    return Xc, Yc


def circle(x, x1, x2, y1, y2, R):
    """Calculate images with a circular function between 2 points.

    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    R : float or list
        the curvature radius
    """
    Xc, Yc = compute_center_coor(x1, x2, y1, y2, R)
    return Yc - np.sign(R)*np.sqrt(R**2 - (x - Xc)**2)


def dcircle_dx(x, x1, x2, y1, y2, R):
    Xc = compute_center_coor(x1, x2, y1, y2, R)[0]
    return (x - Xc)*np.sign(R)/np.sqrt(R**2 - (x - Xc)**2)


def diff_center_coor(x1, x2, y1, y2, R, dx1, dx2, dy1, dy2, dR):
    """
    Differentiate the center coordinates.

    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    R : float or list
        the curvature radius
    dx1, dr1 : float
        the derivative of the the first point
    dx2, dr2 : float
        the derivative of the the second point
    dR : float
        the derivative of the the curvature radius

    Returns
    -------
    Xc, Yc : float
        The center coordinates.
    dXc, dYc : float
        The derivative of the center coordinates.
    """

    Deltax = x2 - x1
    Deltay = y2 - y1
    d_Deltax = dx2 - dx1
    d_Deltay = dy2 - dy1

    if Deltay != 0:
        u = x1**2 - x2**2 + y1**2 - y2**2
        du = 2*dx1*x1 - 2*dx2*x2 + 2*dy1*y1 - 2*dy2*y2

        a = 1 + (Deltax**2) / (Deltay**2)
        da = 2*d_Deltax*Deltax/(Deltay**2) - 2*d_Deltay*Deltax**2/(Deltay**3)

        b = 2*y1*Deltax/Deltay - 2*x1 + u*Deltax/(Deltay**2)
        db = (2*dy1*Deltax/Deltay - 2*dx1 + du*Deltax/(Deltay**2)
              + d_Deltax*(2*y1/Deltay + u/Deltay**2)
              - d_Deltay*(2*y1*Deltax/(Deltay**2) + 2*u*Deltax/(Deltay**3)))

        c = u**2/(4*Deltay**2) + y1*u/Deltay + y1**2 + x1**2 - R**2
        dc = (du*(u/(2*Deltay**2) + y1/Deltay) + dy1*(u/Deltay + 2*y1)
              + 2*dx1*x1 - 2*dR*R - d_Deltay*(u**2/(2*Deltay**3)
                                              + y1*u/Deltay**2))

        discr = np.sqrt(b**2 - 4 * a * c)
        d_discr = 0.5*(2*db*b - 4*da*c - 4*a*dc)/discr

        Xc1 = (-b - discr) / (2 * a)
        Xc2 = (-b + discr) / (2 * a)
        dXc1 = (-db - d_discr)/(2*a) - da/a*Xc1
        dXc2 = (-db + d_discr)/(2*a) - da/a*Xc2

        Yc1 = -(u + 2 * Deltax * Xc1) / (2 * Deltay)
        Yc2 = -(u + 2 * Deltax * Xc2) / (2 * Deltay)
        dYc1 = (-(du + 2*d_Deltax*Xc1 + 2*Deltax*dXc1)/(2*Deltay)
                - d_Deltay/Deltay*Yc1)
        dYc2 = (-(du + 2*d_Deltax*Xc2 + 2*Deltax*dXc2)/(2*Deltay)
                - d_Deltay/Deltay*Yc2)

        if np.sign(R)*Yc1 >= np.sign(R)*max(y1, y2):
            Xc = Xc1
            Yc = Yc1
            dXc = dXc1
            dYc = dYc1
        else:
            Xc = Xc2
            Yc = Yc2
            dXc = dXc2
            dYc = dYc2
    else:  # If Deltay = 0 it is a particular case not to divide by 0
        Xc = 0.5 * (x1 + x2)
        Yc = y1 + np.sign(R)*np.sqrt(R**2 - 0.25 * Deltax**2)
        dXc = 0.5*(dx1 + dx2)
        dYc = (dy1 + np.sign(R)*0.5*(2*dR*R - 0.5*d_Deltax*Deltax)
               / np.sqrt(R**2 - 0.25*Deltax**2))
    return Xc, Yc, dXc, dYc


def diff_circle(x, x1, x2, y1, y2, R, dx1, dx2, dy1, dy2, dR):
    """Differentitate the circular equation wrt the different parameters.

    Parameters
    ----------
    x : float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    R : float or list
        the curvature radius
    dx1, dr1 : float
        the derivative of the the first point
    dx2, dr2 : float
        the derivative of the the second point
    dR : float
        the derivative of the the curvature radius
    """
    if np.abs(R) < 0.5*np.abs(x2 - x1):
        raise ValueError('The diameter of the circle must be larger than the'
                         ' length of the pipe. Here: '
                         '{}<{}'.format(2*np.abs(R), np.abs(x2 - x1)))

    Xc, Yc, dXc, dYc = diff_center_coor(x1, x2, y1, y2, R, dx1, dx2, dy1, dy2, dR)
    dy = (dYc - 0.5*np.sign(R)*(2*dR*R + 2*dXc*(x - Xc))
          / np.sqrt(R**2 - (x - Xc)**2))
    return dy

class Circle(DesignShape):
    """
    Pipe the radius of which follows a circle equation: constant curvature.

    The circle equation is obtained from the coordinates of two points
    :math:`x_1, r_1` and :math:`x_2, r_2` and the curvature radius :math:`R`.

    Parameters
    ----------
    *params: 5 openwind.design.design_parameter.DesignParameter
        The five parameters in this order: :math:`x_1, x_2, r_1, r_2, R`

    """

    def __init__(self, *params, label=None):
        if len(params) != 5:
            raise ValueError("A circular shape need 5 parameters.")
        self.params = params
        self.label = label

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        kwarg = {'digit':digit, 'unit':unit, 'disp_optim':disp_optim}
        positions = self.get_endpoints_position()
        radii = self.get_endpoints_radius()
        curv = self.params[-1].__str__(**kwarg)
        return '{}\t{}\t{}\t{}\t{:>11s}\t{}'.format(positions[0].__str__(**kwarg),
                                                    positions[1].__str__(**kwarg),
                                                    radii[0].__str__(diameter=diameter, **kwarg),
                                                    radii[1].__str__(diameter=diameter, **kwarg),
                                                    type(self).__name__, curv)

    def get_position_from_xnorm(self, x_norm):
        Xmin = self.params[0].get_value()
        Xmax = self.params[1].get_value()
        return x_norm*(Xmax - Xmin) + Xmin

    def get_radius_at(self, x_norm):
        x1, x2, r1, r2, R = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        radius = circle(x, x1, x2, r1, r2, R)
        self.check_bounds(x, [x1, x2])
        return radius

    def get_diff_radius_at(self, x_norm, diff_index):
        x1, x2, r1, r2, R = eval_(self.params)
        dx1, dx2, dr1, dr2, dR = diff_(self.params, diff_index)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [x1, x2])
        diff_radius = dcircle_dx(x, x1, x2, r1, r2, R)*dx_norm
        diff_radius += diff_circle(x, x1, x2, r1, r2, R, dx1, dx2, dr1, dr2, dR)
        return diff_radius

    def get_endpoints_position(self):
        return self.params[0], self.params[1]

    def get_endpoints_radius(self):
        return self.params[2], self.params[3]

    def get_conicity_at(self, x_norm):
        x1, x2, r1, r2, R = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [x1, x2])
        dradius = dcircle_dx(x, x1, x2, r1, r2, R)
        return dradius

    def diff_conicity_wr_xnorm(self, x_norm):
        y = self.get_radius_at(x_norm)

        x1, x2, r1, r2, R = eval_(self.params)
        Xc, Yc = compute_center_coor(x1, x2, r1, r2, R)
        x = self.get_position_from_xnorm(x_norm)
        return self.get_length()*(1/(Yc - y)
                                  + dcircle_dx(x, x1, x2, r1, r2, R)
                                  * (x - Xc)/(Yc - y)**2)

    def get_diff_conicity_at(self, x_norm, diff_index):
        x1, x2, r1, r2, R = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [x1, x2])

        dx1, dx2, dr1, dr2, dR = diff_(self.params, diff_index)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)
        Xc, Yc, dXc, dYc = diff_center_coor(x1, x2, r1, r2, R, dx1, dx2, dr1,
                                            dr2, dR)
        y = self.get_radius_at(x_norm)
        dy = self.get_diff_radius_at(x_norm, diff_index)

        return ((dx_norm - dXc)*(Yc - y) - (x - Xc)*(dYc - dy))/(Yc - y)**2
