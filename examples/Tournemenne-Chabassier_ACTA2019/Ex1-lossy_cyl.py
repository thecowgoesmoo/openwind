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
import matplotlib.pyplot as plt

from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                      FrequentialSolver, Physics, InstrumentPhysics)


def l2error(z1, z2):
    return np.linalg.norm(z1.imped - z2.imped) / np.linalg.norm(z2.imped)


damping = True
F1 = 20
F2 = 2000
ordres = np.arange(2, 11, 1)
fs = np.arange(F1, F2, 1)
temp = 25
physics = Physics(temp)

x, r = [0, 2e-1], [1e-2 / 2, 1e-2 / 2]
player = Player()
mk_model = InstrumentGeometry(list(zip(x, r)))
phy_model = InstrumentPhysics(mk_model,  temp, player, damping, nondim=True,)


lbd = physics.c(0) / F2
N = np.ceil(2 * mk_model.get_main_bore_length() / lbd)
print("nombre d'elements requis: " + str(N))

lenEle = mk_model.get_main_bore_length() / N

zTMM = FrequentialSolver(phy_model, fs, compute_method='TMM', nb_sub=1)
zTMM.solve()

zFEM = []
ErrorTMM = np.empty(len(ordres))
ErrorFEM = np.empty(len(ordres) - 1)

for r in ordres:
    #discret_model = DiscretizedBoreModel(phy_model, lenEle, r,
    #                                     adim=True)
    zFEM_tmp = FrequentialSolver(phy_model, fs,
                                compute_method='FEM', l_ele = lenEle, order = r)
    zFEM_tmp.solve()
    # print("nombre d'elements produits: " + str(discret_model.meshes[0].nb_eles))
    zFEM.append(zFEM_tmp)
    ErrorTMM[r - ordres[0]] =  l2error(zFEM[-1], zTMM)
    if r > ordres[0]:
        ErrorFEM[r - ordres[0]-1] = l2error(zFEM[-1], zFEM[-2])

#%%
plt.figure(2)
plt.clf()
plt.semilogy(ordres, ErrorTMM,color='red',marker='o')
plt.semilogy(ordres[:-1],ErrorFEM, color='k',marker='+')
plt.xlabel("Finite elements order")
plt.ylabel(r'$\frac{\left \|  Z_{FEM}- Z_{TMM} \right \|}{\left \|  Z_{TMM} \right \|} $')
plt.legend(["TMM","FEM"])
plt.title("Cyl - Lossy, N = 3")
plt.grid()

plt.show()
