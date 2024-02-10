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
This script is part of the numerical examples accompanying the article:
Thibault, A., Chabassier, J., Boutin, H., & Hélie, T. (2023).
Transmission line coefficients for viscothermal acoustics in conical tubes.
Journal of Sound and Vibration, 543, 117355.

Display a diagram comparing the range of validity of the low reduced frequency
model with the range of frequencies and tube radii used in musical acoustics.

This script generates Figure 2 of the article.
"""

from numpy import geomspace, pi, sqrt, meshgrid
import matplotlib.pyplot as plt

plt.figure(figsize=(4.0, 3.0), dpi=200)

freq = geomspace(1e0, 1e6, 100)
radius = geomspace(1e-4, 1e1, 100)
f, r = meshgrid(freq, radius)
omega = 2*pi*f

# Constantes physiques
rho0 = 1.205  # masse volumique
mu = 1.81e-5  # viscosité
c0 = 343  # vitesse du son

# Nombres caractérisant la propagation
# - Longueur d'onde
lambda_ = 2*pi*c0/omega
# - Epaisseur des couches visqueuses
delta_v = sqrt(mu/(rho0*omega))

# - Fréquence réduite
k = r * omega / c0
# - Nombre de Stokes
s = r * sqrt(rho0*omega/mu)
# - Rapport k/s
k_s = k/s


# Gamme de rayons et fréquences de l'acoustique
range_acoustics = (20 < f) * (f < 20_000) * (0.5e-3 < r) * (r < 2e-1)
plt.contour(f, r, range_acoustics, [0.5, 1.5], colors="#000000ff")
plt.contourf(f, r, range_acoustics, [0.5, 1.5], colors="#00000000")
plt.loglog()


# Courbes de niveau de k, s, et k/s
isok = plt.contour(
    f, r, r/lambda_, [1e-1, 1, 10], colors="#2117da", linewidths=[1, 1.5, 2])
isos = plt.contour(
    f, r, r/delta_v, [1, 1e1, 1e2], colors="#d2460a", linewidths=[1, 1.5, 2])
isoks = plt.contour(f, r, delta_v/lambda_,
                    [1e-3, 2e-3], colors="#00b300", linewidths=[1, 1.5])


# Domaine de validité des différents modèles
# range_ZK = (k < 1e-1) * (k/s < 1e-1)
range_ZK = (r/lambda_ < 1e-1) * (k/s < 1e-1)
cs = plt.contourf(f, r, range_ZK, [0.5, 1.5], colors="#00000033")
range_WL = (k < 1e-1) * (s > 1e1)
# plt.contourf(f, r, range_WL, [0.5, 1.5], colors="#00000000", hatches=['--'])
range_Cremer = (s > 1e1)
# plt.contourf(f, r, range_Cremer, [0.5, 1.5], colors="#00000000", hatches=["/"])


t3 = plt.clabel(isoks, inline=True, fmt="$\delta_v/\lambda = %g$",
                manual=[(2e5, 1), (5e4, 1)])
t1 = plt.clabel(isos, inline=True, fmt="$R/\delta_v = %g$",
                manual=[(10, 0.0003), (10, 0.003), (10, 0.03)])
t2 = plt.clabel(isok, inline=True, fmt="$R/\lambda = %g$",
                manual=[(20, 2), (200, 2), (2000, 2)])
labels = t1 + t2 + t3
for t in labels:
    t.set_bbox(dict(color='white'))


plt.xlabel("Frequency (Hz)")
plt.ylabel("Radius (m)")
# plt.legend()
plt.tight_layout()
plt.savefig("Fig3.png", dpi=300)
