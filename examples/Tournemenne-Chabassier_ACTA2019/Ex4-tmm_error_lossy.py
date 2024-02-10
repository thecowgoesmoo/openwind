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
                      Physics, FrequentialSolver)



def l2error(z1, z2):
    return np.linalg.norm(z1.imped - z2.imped) / np.linalg.norm(z2.imped)

damping = True
ordres = [10, 11]
temp = 25
params = [[1e-2, 1e-3, 45],  # cuvette d'embouchure
          [1e-1, 1e-3, 10],  # queue d'embouchure
          [1e-1, 1e-2, 1],  # leadpipe pieces
          [1, 1e-3, 1],  # conical instrument 1
          [1, 1e-2, 1],  # conical instrument 2
          5000,  # double cône discontinus
          1000,  # trompette
]

names = ["Cup", "Backbore", "Leadpipe", "ConicalIns1", "ConicalIns2",
         "discont", "Trumpet"]
# names = ["Cup", "Backbore"]
# subdivisions = [6, 30, 50, 70, 100]
subdivisions = np.hstack((np.arange(1, 20, 1), np.array([22, 26, 32, 40, 56, 82, 144])))
#subdivisions = [9, 20]
Ns = [24, 24, 24, 60, 60, 24, 72]
F1 = 20
F2 = 2000
fs = np.arange(F1, F2, 10)
print('param loaded')
physics = Physics(temp)
traceErr = []
player = Player()
plt.figure(5),
plt.clf()

##
for idx, param in enumerate(params):

    print(names[idx])
    if param == 1000:
        xr = np.loadtxt('Tr_co_MP')
        x = xr[:, 0].tolist()
        r = xr[:, 1].tolist()
        mk_model = InstrumentGeometry('Tr_co_MP')
    elif param == 5000:
        x = [0, 5e-2, 5e-2, 1e-1]
        r = [1e-2, 2e-2, 2.5e-2, 2.2e-2]
        mk_model = InstrumentGeometry(list(zip(x,r)))
    else:
        lgt = param[0]
        rd = param[1]
        alpha = param[2]
        x = [0, lgt]
        r = [rd, rd + lgt * np.tan(alpha * np.pi / 180)]
        if alpha == 45 and lgt == 1e-2 and rd == 1e-3:
            rinit = r[1]
            r[1] = r[0]
            r[0] = rinit
        mk_model = InstrumentGeometry(list(zip(x,r)))

    phy_model = InstrumentPhysics(mk_model, temp, player=player, losses=damping, nondim=True)

    N = Ns[idx]
    print("nombre d'elements requis: " + str(N))
    lEle = mk_model.get_main_bore_length() / N

    zFEM = []
    # pdb.set_trace()
    for r in ordres:
        #discret_model = DiscretizedBoreModel(phy_model, lEle, r,
        #                                 adim=True)
        #print("nombre d'elements produits: " + str(discret_model.meshes[0].nb_eles))
        zFEM_tmp = FrequentialSolver(phy_model, fs, compute_method='FEM', l_ele= lEle, order=r)
        zFEM_tmp.solve()
        zFEM.append(zFEM_tmp)
        print('----------ordre: ' + str(r))

    TES = np.empty(len(subdivisions))
    eps = np.empty(len(subdivisions))
    ErrorTMM = np.empty(len(subdivisions))
    ErrorFEM = l2error(zFEM[0], zFEM[1])
    print("converged if value inf to 1e-12: " + str(ErrorFEM))
    zTMM = []
    curves = []
    for i, subdiv in enumerate(subdivisions):
        print(subdiv)
        zTMM_tmp = FrequentialSolver(phy_model, fs, compute_method='TMM', nb_sub=subdiv)
        zTMM_tmp.solve()
        zTMM.append(zTMM_tmp)
        if param == 1000:
            TES[i] = 0.002 / subdiv
        elif param == 5000:
            TES[i] = 5e-2 / subdiv
        else:
            TES[i] = param[0] / subdiv
        ErrorTMM[i] = l2error(zTMM[-1], zFEM[1])

    plt.loglog(TES, ErrorTMM)

plt.grid()
plt.ylabel(r'error')
plt.legend(names)

plt.show()
