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

"""FIXME

This module contains old, undocumented code.
"""

import numpy as np


def peak_finder(f, Z, nbPeaks=10, interpMethod="parabolic"):
    peaksfr = []
    peaksvl = []
    idxshigh = []
    idxhigh = 0
    idxlow = 0
    lowsfr = [f[0]]
    lowsvl = [np.abs(Z[0])]
    idxslow = [0]
    notmoving = 0
    df = f[1] - f[0]
    limdf = 5
    nStop = np.ceil(limdf / df)
    # bypass for zooming cases
    # nStop = 15
    for i in range(len(Z)):
        # recherche du maximum
        if len(idxshigh) < len(idxslow):
            if np.abs(Z[i]) > np.abs(Z[idxhigh]):
                idxhigh = i
                notmoving = 0
            else:
                notmoving = notmoving + 1
            if notmoving == nStop:
                idxshigh.append(idxhigh)
                Q = q_finder(f, Z, idxhigh, idxlow)
                if interpMethod == "circlefit":
                    Fr, A = circle_fit(f, Z, np.pi, f[idxhigh], Q, idxhigh)
                    peaksfr.append(Fr)
                    peaksvl.append(A)
                elif interpMethod == "parabolic":
                    fb = f[idxhigh - 1]
                    fp = f[idxhigh]
                    fn = f[idxhigh + 1]
                    Zb = np.abs(np.log10(Z[idxhigh - 1]))
                    Zp = np.abs(np.log10(Z[idxhigh]))
                    Zn = np.abs(np.log10(Z[idxhigh + 1]))
                    A = np.array([[fb**2, fb, 1], [fp**2, fp, 1],
                                  [fn**2, fn, 1]])
                    B = np.array([Zb, Zp, Zn])
                    coefs = np.linalg.solve(A, B)
                    Fr = - coefs[1] / 2 / coefs[0]
                    A = coefs[0] * Fr**2 + coefs[1] * Fr + coefs[2]
                    peaksfr.append(Fr)
                    peaksvl.append(10**A)
                elif interpMethod == "none":
                    peaksfr.append(f[idxhigh])
                    peaksvl.append(np.abs(Z[idxhigh]))
                idxlow = idxhigh + 1
                notmoving = 0
        # puis recherche du minimum suivant
        if len(idxshigh) == len(idxslow):
            if np.abs(Z[i]) < np.abs(Z[idxlow]):
                idxlow = i
                notmoving = 0
            else:
                notmoving = notmoving + 1
            if notmoving == nStop:
                lowsfr.append(f[idxlow])
                lowsvl.append(np.abs(Z[idxlow]))
                idxslow.append(idxlow)
                idxhigh = idxlow + 1
                notmoving = 0
                if len(peaksfr) == nbPeaks:
                    break
    return np.array(peaksfr), np.array(peaksvl), np.array(lowsfr),\
        np.array(lowsvl)


def q_finder(f, Z, idxPeak, idxPreviousHole):
    Zzoom = Z[idxPreviousHole: idxPeak + 1]
    idxSqrt2 = np.argmin(np.abs(np.abs(Zzoom) - np.abs(Z[idxPeak]) /
                                np.sqrt(2)))
    fSqrt2 = f[idxPreviousHole + idxSqrt2]
    # We assume peak symmetry
    return f[idxPeak] / (2 * (f[idxPeak] - fSqrt2))


def circle_fit(f, Z, alpha, frEst, Q, iFr):
    # To understand the method used hereafter, please read the annex A1 of the
    # Jean Christophe Le Roux Ph.D. thesis (1994)

    nbp = Z.size

    # estimating the frequency window to take into account for the peak
    # detection
    D = np.tan(alpha / 4)**2 + 4 * Q**2
    Fa = -frEst * (np.tan(alpha / 4) - np.sqrt(D)) / (2 * Q)
    ia = np.argmin(np.abs(f - Fa))
    ib = iFr + (iFr - ia)
    if ib > nbp:
        ib = nbp

    # total number of point for the 2 least square minimisations
    n = ib - ia + 1

    # windowing to the circle arc of interest
    rep_win = Z[ia: ib + 1]
    R = np.real(rep_win)
    X = np.imag(rep_win)
    f_win = f[ia: ib + 1]
    # finding the position of the circle center thanks to the least square
    # minimisation (The idea is to minimize the circle equation for the n
    # points of interest)
    PR1 = np.sum(R)
    PR2 = np.sum(R**2)
    PR3 = np.sum(R**3)

    PX1 = np.sum(X)
    PX2 = np.sum(X**2)
    PX3 = np.sum(X**3)

    PK11 = np.sum(R * X)
    PK12 = np.sum(R * X**2)
    PK21 = np.sum(R**2 * X)

    Pr = PR1**2 / n - PR2
    Px = PX1**2 / n - PX2
    P21 = PK21 + PX3 - PR2 * PX1 / n - PX2 * PX1 / n
    P12 = PK12 + PR3 - PR1 * PX2 / n - PR1 * PR2 / n
    P11 = PR1 * PX1 / n - PK11
    D = 2 * (Pr * Px - P11**2)

    Rce = (P21 * P11 - P12 * Px) / D
    Xce = (P12 * P11 - P21 * Pr) / D
    Rad = np.sqrt(Rce**2 + Xce**2 - 2 * Rce * PR1 / n -
                  2 * Xce * PX1 / n + PR2 / n + PX2 / n)
    # finding the angle giving the circle position where the Frequency
    # resonance is (the farthest point of the circle from the origin)
    dephase = np.arctan2(Xce, Rce)  # ????? atan2

    # This second least square minimization aims to find Fr and Q. The
    # resonance frequency correspond to alphai == 0.
    alphai = np.arctan2(X - Xce, R - Rce) - dephase
    F1 = np.sum(f_win**2)
    F2 = np.sum(f_win ** - 2)
    T1 = np.sum(np.tan(alphai / 2) * f_win)
    T2 = np.sum(np.tan(alphai / 2) / f_win)

    Fr = np.sqrt((n * T1 - T2 * F1) / (T1 * F2 - n * T2))
    Q = -np.sqrt((n * T1 - T2 * F1) * (T1 * F2 - n * T2)) / (n**2 - F1 * F2)

    # Calculation of the amplitude at the resonance (geometrically: amplitude
    # center + radus of the circle)
    A = np.sqrt(Rce**2 + Xce**2) + Rad
    return Fr, A
