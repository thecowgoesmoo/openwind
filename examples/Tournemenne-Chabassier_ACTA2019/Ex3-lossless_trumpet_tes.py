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

from openwind import (InstrumentGeometry, Player, InstrumentPhysics,
                      FrequentialSolver)



def l2error(z1, z2):
    return np.linalg.norm(z1.imped - z2.imped) / np.linalg.norm(z2.imped)


damping = False
F1 = 20
F2 = 2000
ordres = np.arange(1, 7, 1)
TESs = [1e-1, 5e-2, 3e-2, 2e-2, 1e-2, 5e-3, 3e-3, 2e-3, 1e-3]
fs = np.arange(F1, F2, 1)
temp = 25
print('param loaded')
##
player= Player()
mk_model = InstrumentGeometry('Tr_co_MP')
phy_model = InstrumentPhysics(mk_model, temp, player, damping)


zTMM = FrequentialSolver(phy_model, fs, compute_method='TMM', nb_sub=1)
zTMM.solve()
zFEM = []
Error = np.empty((len(TESs), len(ordres)))
for i, TES in enumerate(TESs):
    print("TES: " + str(TES))

    for j, r in enumerate(ordres):


        zFEM_tmp = FrequentialSolver(phy_model, fs, compute_method='FEM', l_ele=TES, order=r)
        zFEM_tmp.solve()
        zFEM.append(zFEM_tmp)
        print('----------ordre: ' + str(r))
        Error[i, j] = l2error(zFEM[-1], zTMM)


#%%
plt.figure(4)
plt.clf()
for i in ordres:
    plt.loglog(TESs,Error[:, i - ordres[0]])
plt.xlabel("Target element size")
plt.ylabel(r'$\frac{\left \|  Z_{FEM}- Z_{TMM} \right \|}{\left \|  Z_{TMM} \right \|} $')
plt.title("Lossless Trumpet")
plt.grid()

plt.show()
