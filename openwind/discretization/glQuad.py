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
Coefficients of spectral finite elements using the Gauss-Lobato quadrature.

Compute the quadrature on each 1D element.
"""
from functools import lru_cache

import numpy as np
import scipy.special as sp
import scipy.linalg as la


class GLQuad:
    """
    Calculate the quantities for the Gauss-Lobatto quadrature for 
    Lagrange polynomial basis functions interpolated on the quadrature
    points (spectral method).

    see 
    [1] Quarteroni, A., Sacco, R., & Saleri, F. (2008). 
    Méthodes Numériques: Algorithmes, analyse et applications. 
    Springer Science & Business Media.
    
    Parameters
    ----------
    r : int
        The order of the quadrature

    Attributes
    ----------
    pts : array of r+1 float
        the point position on the reference element
    weigth : array of r+1 float
        the weight of each point 
    omega : array of r + 1 float
        quantity to calculate integrals
    BK : square matrix (r+1, r+1)
        elementary matrix for gradient term
    """

    def __init__(self, r):
        """Define the 4 attributes."""
        self.pts, self.weight = GLQuad.lobatto(r)
        self.omega = np.zeros((r + 1,))
        for i in range(r + 1):
            ind = np.array([*np.arange(0, i), *np.arange(i + 1, r + 1)])
            self.omega[i] = 1 / np.prod(self.pts[i] - self.pts[ind])

        Grad_Vect = np.zeros((r + 1, r + 1))
        self.BK = np.zeros((r + 1, r + 1))
        for i in range(r + 1):
            ind = np.array([*np.arange(0, i), *np.arange(i + 1, r + 1)])
            Grad_Vect[ind, i] = self.omega[i] / (self.pts[ind] -
                                                 self.pts[i]) / self.omega[ind]
            Grad_Vect[i, i] = self.__dphi1D(i)
            self.BK[:, i] = - Grad_Vect[:, i] * self.weight

    def __dphi1D(self, i):
        r = len(self.omega) - 1
        x = self.pts[i]
        res = 0
        ind = np.array([*np.arange(0, i), *np.arange(i + 1, r + 1)])
        for j in ind:
            ind2 = np.array([*np.arange(0, j), *np.arange(j + 1, r + 1)])
            ind2 = ind2[ind2 != i]
            t = 1
            for k in ind2:
                t = t * (x - self.pts[k])
            res = res + t
        res = res * self.omega[i]
        return res

    # Use a cache to compute polynomials only once per order r
    @lru_cache(maxsize=20)
    def lagranPolys(self):
        """Compute the functions used for the numerical scheme.

        In our case, we use the Lagrangian polynomials taken on the points
        of the quadrature. The reference element is considered which spans
        from 0 to 1
        """
        positions = self.pts

        M = len(positions)
        p = []
        for j in range(M):
            pt = np.poly1d(1.0)
            for k in range(M):
                if k == j:
                    continue
                fac = positions[j]-positions[k]
                pt *= np.poly1d([1.0, -positions[k]])/fac
            p.append(pt)
        return p

    def lagran_polys_derivate(self):
        """Compute de derivate of the Lagrange interpolation polynomials
        """
        positions = self.pts

        M = len(positions)
        polys = []
        for j in range(M):
            pt = np.poly1d(1.0)
            for k in range(M):
                if k == j:
                    continue
                fac = positions[j]-positions[k]
                pt *= np.poly1d([1.0, -positions[k]])/fac
            polys.append(pt)
        dpolys = []
        for p in polys:
            dpolys.append(p.deriv())
        return dpolys

    @staticmethod
    def lobatto(r):
        xw = GLQuad.lobatto_jacobi(r - 1, 0, 0)
        pts_lob = 0.5 + 0.5 * xw[:, 0]
        pts_lob[0] = 0
        pts_lob[r] = 1
        poids_lob = 0.5 * xw[:, 1]
        return pts_lob, poids_lob

    @staticmethod
    def lobatto_jacobi(N, *param):
        if len(param) < 1:
            a = 0
            b = 0
        else:
            a = param[0]
        if len(param) < 2:
            b = 0
        else:
            b = param[1]
            ab = GLQuad.r_jacobi(N + 2, a, b)
            ab[N + 1, 0] = (a - b) / (2 * N + a + b + 2)
            ab[N + 1, 1] = 4 * (N + a + 1) * (N + b + 1) * (N + a + b + 1) \
                / ((2 * N + a + b + 1) * (2 * N + a + b + 2) ** 2)
            xw = GLQuad.gauss(N + 2, ab)
        return xw

    @staticmethod
    def r_jacobi(N, *param):
        if len(param) < 1:
            a = 0
            b = 0
        else:
            a = param[0]
        if len(param) < 2:
            b = 0
        else:
            b = param[1]
        if N <= 0 or a <= -1 or b <= -1:
            raise ValueError("parameter(s) out of range")
        nu = (b - a) / (a + b + 2)
        mu = (2 ** (a + b + 1) * sp.gamma(a + 1) * sp.gamma(b + 1) /
              sp.gamma(a + b + 2))
        if N == 1:
            return np.array([nu, mu])
        N = N - 1
        n = np.arange(1, N + 1)
        nab = 2 * n + a + b
        A = np.array([nu, *np.ones((N,)) * (b ** 2 - a ** 2) /
                      (nab * (nab + 2))])
        n = np.arange(2, N + 1)
        nab = nab[n - 1]
        B1 = 4 * (a + 1) * (b + 1) / ((a + b + 2) ** 2 * (a + b + 3))
        B = 4 * (n + a) * (n + b) * n * (n + a + b) / \
            ((nab ** 2) * (nab + 1) * (nab - 1))
        ab = np.array([A, [mu, B1, *B]]).transpose()
        return ab

    @staticmethod
    def gauss(N, ab):
        N0 = ab.shape[0]
        if N0 < N:
            raise ValueError("input array ab too short")

        J = np.zeros((N, N))
        for n in range(N):
            J[n, n] = ab[n, 0]
        for n in range(1, N):
            J[n, n - 1] = np.sqrt(ab[n, 1])
            J[n - 1, n] = J[n, n - 1]
            VD = la.eig(J)
            idx = np.argsort(VD[0])
            D = np.real(VD[0][idx])
            V = VD[1][:, idx]
            xw = np.array([D, ab[0, 1] * V[0, :].transpose() ** 2]).transpose()
        return xw
