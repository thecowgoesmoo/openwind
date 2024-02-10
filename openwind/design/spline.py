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
Pipe the radius of which is defined by a spline (piecewise polynomial curve
twice differentiable).
"""

import numpy as np
import scipy.interpolate as SI

from openwind.design import DesignShape, eval_, diff_, VariableParameter


def organize_spline_param(X, R):
    """
    Sorte the couple of coordinate and convert into np.array

    Parameters
    ----------
    X : list of float
        The x coordinates of the control points.
    R : list of float
        The radii of the control points.

    Returns
    -------
    X : array of float
        The sorted x coordinates.
    R : array of float
        The radii sorted wr to the x coordinates.

    """

    X, R = (np.array(x) for x in zip(*sorted(zip(X, R))))
    return X, R


def construct_spline_matrix(H):
    """
    Compute the matrix used to compute the second derivative at the control
    points

    Parameters
    ----------
    H : list
        The distances between the control points.

    Returns
    -------
    M : array NxN
        The matrix.

    """
    N = len(H) - 1
    ind = np.arange(N, dtype=int)
    M = np.zeros((N, N))
    M[ind, ind] = (1/3)*(H[ind] + H[ind+1])  # diagonal
    M[ind[:-1], ind[1:]] = (1/6)*H[ind[1:]]
    M[ind[1:], ind[:-1]] = (1/6)*H[ind[1:]]
    return M


def spline_coeff(x, X, R):
    """
    Compute the coefficients of the piecewise polynomial

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.

    Returns
    -------
    A : array
        The polynomial coefficients :math:`A(x)`.
    B : array
        The polynomial coefficients :math:`B(x)`.
    H : list of float
        The distance between the x-coordinate of the knots.
    Rsec : list of float
        The second derivative of the radius at the knots.
    k : list of int
        The index of the part of the piecewise polynomial is placed each x.

    """
    H = np.diff(X)
    N = len(X) - 2  # matrix size

    M = construct_spline_matrix(H)
    dR_H = np.diff(R)/H
    F = np.diff(dR_H)
    Rsec = np.zeros(N+2)
    Rsec[1:-1] = np.linalg.solve(M, F)

    k = np.searchsorted(X, x, side='right')-1
    if k.__class__ == np.int64:  # k is different depending of x type
        k = np.array([k])
    k[k > N] = N
    A = (x - X[k])/H[k]
    B = 1 - A

    return A, B, H, Rsec, k

def spline_deriv_coeff(x, X, R, A, B, H, Rsec, k, pos_deriv):
    """
    Differentiate the coefficients with respect to a knot position

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.
    A : array
        The polynomial coefficients :math:`A(x)`.
    B : array
        The polynomial coefficients :math:`B(x)`.
    H : list of float
        The distance between the x-coordinate of the knots.
    Rsec : list of float
        The second derivative of the radius at the knots.
    k : list of int
        The index of the part of the piecewise polynomial is placed each x.
    pos_deriv : int
        The index of the knot position with respect wich it is differentiate.

    Returns
    -------
    dA : array
        The derivative of A.
    dH : list
        The derivative of each H.
    dRsec : list
        The derivative of each Rsec.

    """
    dR_H = np.diff(R)/H
    dH = np.zeros(H.shape)
    if pos_deriv < len(dH):
        dH[pos_deriv] = -1.0
    if pos_deriv > 0:
        dH[pos_deriv - 1] = 1.0

    M = construct_spline_matrix(H)
    dM = construct_spline_matrix(dH)
    dF = - dH[1:]/H[1:]*dR_H[1:] + dH[:-1]/H[:-1]*dR_H[:-1]
    dRsec = np.zeros(X.shape)
    dRsec[1:-1] = np.linalg.solve(M, dF - dM.dot(Rsec[1:-1]))
    # differentiation of the coefficient of the polynomial expression
    Hx = H[k]
    kj = (k == pos_deriv)
    if np.isscalar(x):
        dA = np.zeros(1)
    else:
        dA = np.zeros(x.shape)
    dA[kj] = - B[kj]/Hx[kj]
    if pos_deriv > 0:
        kj_minus = (k == (pos_deriv - 1))
        dA[kj_minus] = - A[kj_minus]/Hx[kj_minus]
    return dA, dH, dRsec


def cubicspline(x, X, R):
    """ Calculate cubicspline between x1 and x2 and passing by the knots
    defined in param by using the scipy function.

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.


    """
    X, R = organize_spline_param(X, R)
    spl = SI.CubicSpline(X, R)
    return spl(x)


def spline(x, X, R):
    """
    Calculate cubicspline between x1 and x2.

    This is an homemade which assure that the curve passes by the knots.

    .. math ::
        r(x) = B(x) R_k + A(x) R_{k+1} + \\frac{H_k^2}{6} \\left((B(x)^3 - B(x))R_k'' + \
           (A(x)^3 - A(x)) R_{k+1}'' \\right)

    with :math:`R_k`, the radii of the knots, :math:`H_k = X_{k+1} - X_{k}`
    the distance between the x-coordinate of the knots, :math:`R_k''`
    the second derivative of the radius at the control points, computed to assure
    the C2 property, and :math:`A(x), B(x)` polynomial coefficient such as

    .. math::
        \\begin{align}
        A(x) & = \\frac{x - X_k}{H_k} \\\\
        B(x) & = 1 - A(x)
        \\end{align}

    It is assumed that :math:`R_0''=R_N''=0``.

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.

    Returns
    -------
    array of float

    """
    X, R = organize_spline_param(X, R)
    A, B, H, Rsec, k = spline_coeff(x, X, R)
    spl = (B*R[k] + A*R[k+1] + (H[k]**2)/6*((B**3 - B)*Rsec[k] +
           (A**3 - A)*Rsec[k+1]))
    if np.isscalar(x):
        spl = float(spl)
    return spl

def diff_spline_wr_x(x, X, R):
    """
    Calculate the differential of the cubicspline wr to x

    .. math::
        \\frac{\\partial r(x)}{\\partial x}

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.

    """
    X, R = organize_spline_param(X, R)
    A, B, H, Rsec, k = spline_coeff(x, X, R)
    diff_spl = (R[k+1] - R[k])/H[k] + H[k]/6*((-3*B**2 + 1)*Rsec[k]
                                              + (3*A**2 - 1)*Rsec[k+1])
    if np.isscalar(x):
        diff_spl = float(diff_spl)
    return diff_spl

def deriv_spline_radius(x, X, R, rad_deriv):
    """
    Compute the derivative of the cubic spline with respect to one knot radius

    .. math::
        \\frac{\\partial r(x)}{\\partial R_k}

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.
    rad_deriv: index of the radius to which is computed the derivative.
    """
    # all the radii are fixed to 0, and the position are kept similar
    new_radii = np.zeros_like(R)
    # the radius of interest is fixed to 1
    new_radii[rad_deriv] = 1
    return spline(x, X, new_radii)

def deriv_spline_position(x, X, R, pos_deriv):
    """
    derivation with respect to the x-coordinate of a knot

     .. math::
        \\frac{\\partial r(x)}{\\partial X_k}

    Parameters
    ----------
    x : array of float
        The locations at which compute the spline.
    X : array of float
        The x coordinate of the control points.
    R : array of float
        The radii of the control points.
    pos_deriv: index of the knot location to which is computed the derivative.

    """
    X, R = organize_spline_param(X, R)
    # linear system to compute the second derivative
    A, B, H, Rsec, k = spline_coeff(x, X, R)
    # linear sys. to compute the variation of the second derivative
    dA, dH, dRsec = spline_deriv_coeff(x, X, R, A, B, H, Rsec, k, pos_deriv)
    Hx = H[k]
    dHx = dH[k]

    dP = (dA * (R[k + 1] - R[k]) +
          dA*Hx**2/6*(-1*(3*B**2 - 1)*Rsec[k] + (3*A**2 - 1)*Rsec[k + 1]) +
          dHx*Hx/3*((B**3 - B) * Rsec[k] + (A**3 - A) * Rsec[k + 1]) +
          (Hx**2)/6 * ((B**3 - B) * dRsec[k] + (A**3 - A) * dRsec[k + 1]))
    if np.isscalar(x):
        dP = float(dP)
    return dP


class Spline(DesignShape):
    """
    Pipe the radius of which is defined by a spline.

    A spline is a piecewise polynomial curve twice differentiable. Here the
    spline is defined by, at least, two points :math:`(X_1, R_1)` and
    :math:`(X_2, R_2)`.

     .. math ::
        r(x) = B(x) R_k + A(x) R_{k+1} + \\frac{H_k^2}{6} \\left((B(x)^3 - B(x))R_k'' + \
           (A(x)^3 - A(x)) R_{k+1}'' \\right)

    with :math:`R_k`, the radii of the knots, :math:`H_k = X_{k+1} - X_{k}`
    the distance between the x-coordinate of the knots, :math:`R_k''`
    the second derivative of the radius at the control points, computed to assure
    the C2 property, and :math:`A(x), B(x)` polynomial coefficient such as

    .. math::
        \\begin{align}
        A(x) & = \\frac{x - X_k}{H_k} \\\\
        B(x) & = 1 - A(x)
        \\end{align}

    The piecwise polynomial respect these conditions:

    - The curve passes by all the construction points of coordinates \
        :math:`(X_k, R_k)`
    - The second derivative of the curve w.r. to the abscissa is continuous
    - At the two endpoints, the second derivative equals zero :math:`R_0''=R_N''=0``

    Parameters
    ----------
    *params : even number (>4) of openwind.design.design_parameter.DesignParameter
        The parameters in this order:
        :math:`x_1, x_2, \\ldots, x_n,  r_1, r_2, \\ldots, r_n`

    """
    def __init__(self, *params, label=None):
        if len(params) < 4:
            raise ValueError("A spline needs at least 4 parameters.")
        if (len(params) % 2) != 0:
            raise ValueError("A spline needs an even number of parameters.")
        N = len(params)//2
        self.X = params[:N]
        self.R = params[N:]
        self.__check_x_sorted()
        self.label=label

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        kwarg = {'digit':digit, 'unit':unit, 'disp_optim':disp_optim}
        bound = '{}\t{}\t{}\t{}'.format(self.X[0].__str__(**kwarg),
                                        self.X[-1].__str__(**kwarg),
                                        self.R[0].__str__(diameter=diameter, **kwarg),
                                        self.R[-1].__str__(diameter=diameter, **kwarg))
        X_int = ''
        for param in self.X[1:-1]:
            X_int += '{}\t'.format(param.__str__(**kwarg))
        R_int = ''
        for param in self.R[1:-1]:
            R_int += '{}\t'.format(param.__str__(diameter=diameter, **kwarg))
        return '{bound}\t{class_:>11s}\t{X}{R}'.format(bound=bound, X=X_int, R=R_int,
                                                       class_=type(self).__name__)

    def __check_x_sorted(self):
        Xval = eval_(self.X)
        if any(np.diff(Xval) <= 0):
            raise ValueError("The position of the points used to describe " +
                             "the spline must be sorted")

    def get_radius_at(self, x_norm):
        self.__check_x_sorted()
        Xval = eval_(self.X)
        Rval = eval_(self.R)
        x = self.get_position_from_xnorm(x_norm)
        self.check_bounds(x, [Xval[0], Xval[-1]])
        return spline(x, Xval, Rval)

    def get_diff_radius_at(self, x_norm, diff_index):
        self.__check_x_sorted()
        Xval = eval_(self.X)
        Rval = eval_(self.R)
        dXs = diff_(self.X, diff_index)
        dRs = diff_(self.R, diff_index)

        x = self.get_position_from_xnorm(x_norm)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)
        diff_radius = diff_spline_wr_x(x, Xval, Rval)*dx_norm
        for k, dX in enumerate(dXs):
            if dX != 0:
                diff_radius += dX * deriv_spline_position(x, Xval, Rval, k)
        for k, dR in enumerate(dRs):
            if dR != 0:
                diff_radius += dR * deriv_spline_radius(x, Xval, Rval, k)
        self.check_bounds(x, [Xval[0], Xval[-1]])
        return diff_radius

    def get_endpoints_position(self):
        return self.X[0], self.X[-1]

    def get_endpoints_radius(self):
        return self.R[0], self.R[-1]

    def get_conicity_at(self, x_norm):
        Xval = eval_(self.X)
        Rval = eval_(self.R)
        x = self.get_position_from_xnorm(x_norm)
        return diff_spline_wr_x(x, Xval, Rval)

    def diff_conicity_wr_xnorm(self, x_norm):
        self.__check_x_sorted()
        Xval = eval_(self.X)
        Rval = eval_(self.R)
        X, R = organize_spline_param(Xval, Rval)
        x = self.get_position_from_xnorm(x_norm)
        A, B, H, Rsec, k = spline_coeff(x, X, R)
        dcon_dx = B*Rsec[k] + A*Rsec[k+1]
        if np.isscalar(x):
            dcon_dx = float(dcon_dx)
        return dcon_dx * self.get_length()

    def get_diff_conicity_at(self, x_norm, diff_index):
        self.__check_x_sorted()
        Xval = eval_(self.X)
        Rval = eval_(self.R)
        dXs = diff_(self.X, diff_index)
        dRs = diff_(self.R, diff_index)
        x = self.get_position_from_xnorm(x_norm)
        dx_norm = self.get_diff_position_from_xnorm(x_norm, diff_index)

        X, R = organize_spline_param(Xval, Rval)
        X, dR = organize_spline_param(Xval, dRs)
        X, dX = organize_spline_param(Xval, dXs)

        A, B, H, Rsec, k = spline_coeff(x, X, R)


        dRsec = spline_coeff(x, X, dR)[3]
        dA = np.zeros_like(A)
        dH = np.zeros_like(H)

        for ind_pos, dpos in enumerate(dX):
            dA_dX, dH_dX, dRsec_dX = spline_deriv_coeff(x, X, R, A, B, H, Rsec,
                                                        k, ind_pos)
            dA += dA_dX*dpos
            dH += dH_dX*dpos
            dRsec += dRsec_dX*dpos

        # diff_spl = (R[k+1] - R[k])/H[k] + H[k]/6*((-3*B**2 + 1)*Rsec[k]
        #                                           + (3*A**2 - 1)*Rsec[k+1])

        d_con = ((dR[k+1] - dR[k])/H[k]
                 - dH[k]*(R[k+1] - R[k])/H[k]**2
                 + dH[k]/6*((-3*B**2 + 1)*Rsec[k] + (3*A**2 - 1)*Rsec[k+1])
                 + H[k]/6*(6*B*dA*Rsec[k] + (-3*B**2 + 1)*dRsec[k]
                           + 6*A*dA*Rsec[k+1] + (3*A**2 - 1)*dRsec[k+1])
                 + dx_norm*(B*Rsec[k] + A*Rsec[k+1])
                 )
        if np.isscalar(x):
            d_con = float(d_con)
        return d_con

    def cut_shape(self, start, stop):
        raise ValueError('It is not yet possible to slice a spline!')


    def create_nodes_distance_constraints(self, Dmin=0, Dmax=np.inf):
        r"""
        Add the info to optim_params to constraint the distances between the nodes of the spline

        For each nodes we have: ``Dmin <= x(k+1) - x(k) <= Dmax``

        - if neither x(k) nor x(k+1) are variable -> no constraint
        - if only x(k+1) is variable it becomes: ``Lmin + x(k) <= x(k+1) <= Lmax + x(k)``
        - if only x(k) is variable it becomes: ``Lmin - x(k+1) <= - x(k) <= Lmax - x(k+1)``

        The constraint is express as a linear problem such as ``lb <= A.dot(x) <= ub``
        it is therefore necessary to know the indices corresponding to x(k) and x(k+1) in the matrix A
        and the corresponding values of A (-1 or +1)

        Parameters
        ----------
        Lmin : float, optional
            The minimal length desired. The default is 0.
        Lmax : float, optional
            The maximal length desired. The default is np.inf.

        """
        check_type = [x.is_variable() and not isinstance(x, VariableParameter) for x in self.X]
        if any(check_type):
            raise ValueError('The spline nodes distance constraint can be use only with "VariableParameter"')
        for k in range(len(self.X) - 1):
            X0 = self.X[k]
            X1 = self.X[k+1]

            if X0.is_variable() or X1.is_variable(): # if no nodes positions are variable => no constraint
                if not X0.is_variable(): # if only X1 is variable X0 is included in the bounds
                    indices = [X1.index]
                    coef = [1.]
                    lb = Dmin + X0.get_value()
                    ub = Dmax + X0.get_value()
                    X1._optim_params.add_linear_constraint(indices, coef, lb, ub)
                elif not X1.is_variable(): # if only X0 is variable X1 is included in the bounds
                    indices = [X0.index]
                    coef = [-1.]
                    lb = Dmin - X1.get_value()
                    ub = Dmax - X1.get_value()
                    X0._optim_params.add_linear_constraint(indices, coef, lb, ub)
                else:
                    indices = [X0.index, X1.index]
                    coef = [-1., 1.]
                    lb = Dmin
                    ub = Dmax
                    X0._optim_params.add_linear_constraint(indices, coef, lb, ub)
