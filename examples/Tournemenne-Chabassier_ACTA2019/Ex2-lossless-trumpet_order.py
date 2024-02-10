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

from openwind import InstrumentGeometry, Player, ImpedanceComputation, Physics


def l2error(z1, z2):
    return np.linalg.norm(z1.impedance - z2.impedance) / np.linalg.norm(z2.impedance)

damping = False
F1 = 20
F2 = 2000
ordres = np.arange(1, 13, 1)
fs = np.arange(F1, F2, 1)
temp = 25
print('param loaded')
##
player = Player()
mk_model = InstrumentGeometry('Tr_co_MP')
physics = Physics(temp)


zTMM = ImpedanceComputation(fs, 'Tr_co_MP', player=player, temperature=temp, losses=damping, compute_method='TMM', nb_sub=1)

lbd = physics.c(0) / F2
N = np.ceil(6 * mk_model.get_main_bore_length() / lbd)
lenEle = 2.8e-2#mk_model.ltot / N
#print("nombre d'elements requis: " + str(N))

zFEM = []
Error = np.empty(len(ordres))
for r in ordres:

    zFEM_tmp = ImpedanceComputation(fs, 'Tr_co_MP', player=player, temperature=temp, losses=damping, compute_method='FEM', l_ele = lenEle, order=r)
    zFEM_tmp.discretization_infos()
    zFEM.append(zFEM_tmp)

    Error[r - ordres[0]] = l2error(zFEM[-1], zTMM)


#%%
plt.figure(3)
plt.clf()
plt.semilogy(ordres, Error,color='k',marker='o')
plt.xlabel("Finite elements order")
plt.ylabel(r'$\frac{\left \|  Z_{FEM}- Z_{TMM} \right \|}{\left \|  Z_{TMM} \right \|} $')
plt.title("Lossless Trumpet")
plt.grid()

plt.show()
