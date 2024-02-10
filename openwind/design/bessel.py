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
Pipe the radius of which follows a Bessel horn equation.
"""


from openwind.design import DesignShape, eval_, diff_
import numpy as np

def bessel(x, x1, x2, r1, r2, alpha):
    """Calculate images with a "Bessel horn" function between 2 points.

    The radius of the pipe follows the equation

    .. math::
        \\begin{eqnarray}
        r(x) & = & r_1  \left( \\frac{x_1 - x_p}{x - x_p} \\right)^{\\alpha} \\\\
        x_p & = & \\frac{x_1 - R x_2}{1 - R} \\\\
        R & = & \\left( \\frac{r_2}{r_1} \\right)^{1/\\alpha}
        \\end{eqnarray}

    with:

    - :math:`x_1, x_2`: the endpoints positions of the pipe
    - :math:`r_1, r_2`: the endpoints radii of the pipe
    - :math:`\\alpha`: the expansion rate of the horn (inverse of)

    Parameters
    ----------
    x : float, array of float
        the point at which the value of r is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    if r1==r2:
        raise ValueError("A Bessel horn can not have to 2 equal radii (r1!=r2)")
    rr = (r2/r1)**(1/alpha)  # radius ratio
    xp = (x1 - rr*x2)/(1 - rr)  # abscissa of the pole
    return r1 * ((x1 - xp) / (x - xp))**alpha


def dbessel_dx(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to x.

    .. math::
        \\frac{\partial r(x))}{\partial x} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    xp = (x1 - rr*x2)/(1 - rr)
    return -alpha/(x-xp) * r1 * ((x1 - xp) / (x - xp))**alpha


def dbessel_dxp(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to xp.

    .. math::
        \\frac{\partial r(x))}{\partial x_p} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    xp = (x1 - rr*x2)/(1 - rr)
    dy_dxp = r1*alpha*(x1 - x)/(x - xp)**2 * ((x1 - xp)/(x - xp))**(alpha - 1)
    return dy_dxp


def dbessel_drr(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to rr.

    .. math::
        \\frac{\partial r(x))}{\partial R} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    dxp_drr = (x1 - x2) / (1 - rr)**2
    dy_dxp = dbessel_dxp(x, x1, x2, r1, r2, alpha)
    return dy_dxp * dxp_drr


def dbessel_dr1(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to r1.

    .. math::
        \\frac{\partial r(x))}{\partial r_1} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    xp = (x1 - rr*x2)/(1 - rr)

    drr_dr1 = -1/(alpha*r1) * rr
    dy_drr = dbessel_drr(x, x1, x2, r1, r2, alpha)
    dy_dr1 = ((x1 - xp) / (x - xp))**alpha
    return dy_drr * drr_dr1 + dy_dr1


def dbessel_dr2(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to r2.

    .. math::
        \\frac{\partial r(x))}{\partial r_2} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    drr_dr2 = 1/(alpha*r2) * rr
    dy_drr = dbessel_drr(x, x1, x2, r1, r2, alpha)
    return dy_drr * drr_dr2


def dbessel_dx1(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to x1.

    .. math::
        \\frac{\partial r(x))}{\partial x_1} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    xp = (x1 - rr*x2)/(1 - rr)
    dxp_dx1 = 1/(1 - rr)
    dy_dxp = dbessel_dxp(x, x1, x2, r1, r2, alpha)
    dy_dx1 = alpha*r1/(x1 - xp) * ((x1 - xp)/(x - xp))**alpha
    return dy_dxp * dxp_dx1 + dy_dx1


def dbessel_dx2(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to x2.

    .. math::
        \\frac{\partial r(x))}{\partial x_2} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    dxp_dx2 = -rr/(1 - rr)
    dy_dxp = dbessel_dxp(x, x1, x2, r1, r2, alpha)
    return dy_dxp * dxp_dx2


def dbessel_dalpha(x, x1, x2, r1, r2, alpha):
    """Differentiate with respect to alpha.

    .. math::
        \\frac{\partial r(x))}{\partial \\alpha} \


    Parameters
    ----------
    x : float, array of float
        the point at which the value of y is calculated
    x1, r1 : float
        the first point
    x2, r2 : float
        the second point
    alpha : float or list
        coefficient of the bessel: power
    """
    rr = (r2/r1)**(1/alpha)
    xp = (x1 - rr*x2)/(1 - rr)
    A = (x1 - xp)/(x - xp)

    drr_dalpha = -1/alpha**2 * np.log(r2/r1) * rr
    dy_dalpha = r1 * np.log(A) * A**alpha
    dy_drr = dbessel_drr(x, x1, x2, r1, r2, alpha)

    return dy_drr*drr_dalpha + dy_dalpha


class Bessel(DesignShape):
    """
    Pipe the radius of which follows a "Bessel horn" equation.

    The Bessel name comes from the fact that, for this radius evolution, the
    lossless Webster equation in plane wave, can be re-written as a Bessel equation.
    The solution makes appears Bessel function. More details are available in [CK_horn]_.

    The radius of the pipe follows the equation

    .. math::
        \\begin{eqnarray}
        r(x) & = & r_1  \left( \\frac{x_1 - x_p}{x - x_p} \\right)^{\\alpha} \\\\
        x_p & = & \\frac{x_1 - R x_2}{1 - R} \\\\
        R & = & \\left( \\frac{r_2}{r_1} \\right)^{1/\\alpha}
        \\end{eqnarray}

    with:

    - :math:`x_1, x_2`: the endpoints positions of the pipe
    - :math:`r_1, r_2`: the endpoints radii of the pipe
    - :math:`\\alpha`: the expansion rate of the horn (inverse of).

        - :math:`\\alpha<-1` or :math:`\\alpha>0`: convex horn (near 0, the expansion is quicker)
        - :math:`-1<\\alpha<0`: concave horn (near 0, the expansion is quicker)
        - :math:`\\alpha=-1`: cone



    Parameters
    ----------
    *params : 5 openwind.design.design_parameter.DesignParameter
        The five parameters in this order: :math:`x_1, x_2, r_1, r_2, \\alpha`

    References
    ----------
    .. [CK_horn] Chaigne A., Kergomard J, "Acoustics of Musical Instruments",\
    Chap.7.5.1, Springer, 2016.

    """

    def __init__(self, *params, label=None):
        if len(params) != 5:
            raise ValueError("A bessel horn shape need 5 parameters.")
        if params[2] == params[3]:
            raise ValueError("A bessel horn can not have to 2 equal radii (r1!=r2)")
        if params[-1] == 0:
            raise ValueError("The expansion rate of a Bessel Horn must be different to 0 (=cylinder). Please use Conical shape instead.")
        self.params = params
        self.label = label

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        kwarg = {'digit':digit, 'unit':unit, 'disp_optim':disp_optim}
        positions = self.get_endpoints_position()
        radii = self.get_endpoints_radius()
        param = self.params[-1]
        return '{}\t{}\t{}\t{}\t{:>11s}\t{}'.format(positions[0].__str__(**kwarg),
                                                    positions[1].__str__(**kwarg),
                                                    radii[0].__str__(diameter=diameter, **kwarg),
                                                    radii[1].__str__(diameter=diameter, **kwarg),
                                                    type(self).__name__,
                                                    param.__str__(digit=digit, disp_optim=disp_optim))

    def get_radius_at(self, x_norm):
        x1, x2, r1, r2, alpha = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        radius = bessel(x, x1, x2, r1, r2, alpha)
        self.check_bounds(x, [x1, x2])
        return radius

    def get_diff_radius_at(self, x_norm, diff_index):
        x1, x2, r1, r2, alpha = eval_(self.params)
        dx1, dx2, dr1, dr2, dalpha = diff_(self.params, diff_index)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)
        x = self.get_position_from_xnorm(x_norm)
        diff_radius = dbessel_dx(x, x1, x2, r1, r2, alpha)*dx_norm
        if dx1 != 0:
            diff_radius += dx1*dbessel_dx1(x, x1, x2, r1, r2, alpha)
        if dx2 != 0:
            diff_radius += dx2*dbessel_dx2(x, x1, x2, r1, r2, alpha)
        if dr1 != 0:
            diff_radius += dr1*dbessel_dr1(x, x1, x2, r1, r2, alpha)
        if dr2 != 0:
            diff_radius += dr2*dbessel_dr2(x, x1, x2, r1, r2, alpha)
        if dalpha != 0:
            diff_radius += dalpha*dbessel_dalpha(x, x1, x2, r1, r2, alpha)
        self.check_bounds(x, [x1, x2])
        return diff_radius

    def get_endpoints_position(self):
        return self.params[0], self.params[1]

    def get_endpoints_radius(self):
        return self.params[2], self.params[3]

    def get_conicity_at(self, x_norm):
        x1, x2, r1, r2, alpha = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [x1, x2])
        dradius = dbessel_dx(x, x1, x2, r1, r2, alpha)
        return dradius

    def diff_conicity_wr_xnorm(self, x_norm):
        x1, x2, r1, r2, alpha = eval_(self.params)
        x = self.get_position_from_xnorm(x_norm)
        y = bessel(x, x1, x2, r1, r2, alpha)

        rr = (r2/r1)**(1/alpha)
        xp = (x1 - rr*x2)/(1 - rr)
        A = -alpha/(x-xp)

        dA = alpha/(x - xp)**2
        dy = dbessel_dx(x, x1, x2, r1, r2, alpha)
        return self.get_length()*(A*dy + dA*y)

    def get_diff_conicity_at(self, x_norm, diff_index):
        x1, x2, r1, r2, alpha = eval_(self.params)
        dx1, dx2, dr1, dr2, dalpha = diff_(self.params, diff_index)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [x1, x2])
        dx = self.get_diff_position_from_xnorm(x_norm, diff_index)

        rr = (r2/r1)**(1/alpha)
        drr = rr* ( -dalpha/alpha**2 * np.log(r2/r1) - dr1/(alpha*r1)
                   + dr2/(alpha*r2))
        xp = (x1 - rr*x2)/(1 - rr)
        dxp = drr*(x1 - x2)/(1 - rr)**2 + (dx1 - rr*dx2)/(1 - rr)

        A = -alpha/(x-xp)
        y = bessel(x, x1, x2, r1, r2, alpha)

        dA_dalpha = -dalpha/(x - xp)
        dA_dx = dx*alpha/(x - xp)**2
        dA_dxp = -dxp * alpha/(x - xp)**2
        dA = dA_dalpha + dA_dx + dA_dxp

        dy = self.get_diff_radius_at(x_norm, diff_index)

        return dA*y + A*dy
