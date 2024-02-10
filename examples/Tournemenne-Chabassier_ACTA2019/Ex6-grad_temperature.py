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

from openwind import ImpedanceComputation, InstrumentGeometry, Player


damping = True #  False # True
adim = False
F1 = 20
F2 = 1200
fs =  np.arange(F1, F2, 1)
print('param loaded')
player = Player()
mk_model = InstrumentGeometry('Tr_co_MP')
r = 8
N = 82
def GradT(x):
    return 37 - (37 - 21) * x / mk_model.get_main_bore_length()

temp = np.mean(GradT(np.linspace(0, mk_model.get_main_bore_length(), 100)))
print('temperature ' + str(temp))
#phy_model = InstrumentPhysics(mk_model, temp, player, damping)


print("nombre d'elements requis: " + str(N))
lenEle = mk_model.get_main_bore_length() / N

# Mean temperature
temp = np.mean(GradT(np.linspace(0, mk_model.get_main_bore_length(), 100)))
print('Mean temperature ' + str(temp))
zFEMCst = ImpedanceComputation(fs,'Tr_co_MP',temperature=temp,losses=damping,player=player,l_ele=lenEle,order=r,nondim = True)


# Variable temperature
#physicsNonConstant = Physics(GradT)
zFEMGrad = ImpedanceComputation(fs,'Tr_co_MP',temperature=GradT,losses=damping,player=player,l_ele=lenEle,order=r,nondim = True)

#%%
fig = plt.figure(6)
plt.clf()
zFEMCst.plot_impedance(figure=fig, label='Mean')
zFEMGrad.plot_impedance(figure=fig, label='Variable')
plt.show()

plt.show()
