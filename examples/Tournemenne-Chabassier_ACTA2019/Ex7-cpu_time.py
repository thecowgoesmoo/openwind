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

from timeit import timeit
from openwind import (InstrumentGeometry, Player, InstrumentPhysics,
                      Physics, FrequentialSolver, ImpedanceComputation)



def l2error(z1, z2):
    return np.linalg.norm(z1.impedance - z2.impedance) / np.linalg.norm(z2.impedance)


damping = True
F1 = 20
F2 = 2000
fs = np.arange(F1, F2, 1)
temp = 25
subdivisions = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

print('param loaded')
physics = Physics(temp)
player=Player()

# Trompette
xr = np.loadtxt('Tr_co_MP')
x = xr[:,0]
mk_model = InstrumentGeometry('Tr_co_MP')
phy_model = InstrumentPhysics(mk_model, temp, player, damping)

lbdmin = physics.c(0) / F2
lEleCvg = lbdmin / 10
ordreCvg = 14

##
print('--------- computation FEM adapt')
labels  = ['bore0', 'bore1', 'bore2', 'bore3', 'bore4', 'bore5', 'bore6', 'bore7',
 'bore8', 'bore9', 'bore10', 'bore11', 'bore12', 'bore13', 'bore14', 'bore15',
 'bore16', 'bore17', 'bore18', 'bore19', 'bore20', 'bore21', 'bore22', 'bore23',
 'bore24', 'bore25', 'bore26', 'bore27', 'bore28', 'bore29', 'bore30']

l_eles_values = list([0.002     , 0.004     , 0.004     , 0.004     , 0.00373405,
       0.06597   , 0.055     , 0.055     , 0.055     , 0.055     ,
       0.165     , 0.165     , 0.165     , 0.02      , 0.019     ,
       0.03      , 0.035     , 0.03      , 0.029     , 0.03      ,
       0.03100005, 0.03      , 0.061     , 0.03      , 0.031     ,
       0.032     , 0.03      , 0.025     , 0.036     , 0.03      ,
       0.034     , 0.029     , 0.028     ])

l_eles = dict(zip(labels, l_eles_values))
orders = dict(zip(labels, list([2, 3, 3, 3, 2, 5, 4, 4, 4, 4, 6, 6, 5, 2, 2, 2, 3, 2, 2, 2, 2, 2,
       3, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5])))

#%% erreurs

print('--------- computation FEM')
lEleClassic = lbdmin / 2
ordreCPUs = range(2, 6)  # 4 is ok for Lossy
zFEMCvg = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp, player=player, losses=damping, compute_method='FEM', l_ele = lEleCvg, order = ordreCvg)


zFEMAdapt = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp, player=player, losses=damping,  compute_method='FEM', l_ele = l_eles, order = orders)

errorFEMAdapt = l2error(zFEMAdapt, zFEMCvg)
print("error fem adapt: " + str(errorFEMAdapt))



##
zTMM = []
errorsTMM = []
for subdiv in subdivisions:
    print('-------- computation TMM ' + str(subdiv))
    zTMM_tmp =  ImpedanceComputation(fs,'Tr_co_MP', temperature=temp, player=player, losses=damping,  compute_method='TMM',  nb_sub=subdiv)
    zTMM.append(zTMM_tmp)
    errorsTMM.append(l2error(zTMM[-1], zFEMCvg))


#%% CPU times
def wrapper_FEM(lEleClassic, ordr):
    def FEM_CPU():
        zFEM = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp,player=player, losses=damping,  compute_method='FEM', l_ele = lEleClassic, order = ordr)
    return FEM_CPU

errorsFEM = []
CPUsFEM = []
for ordreCPU in ordreCPUs:
    zFEM = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp,player=player, losses=damping,  compute_method='FEM', l_ele = lEleClassic, order = ordreCPU)
    errorsFEM.append(l2error(zFEM, zFEMCvg))
    print('CPUFEM')
    repe = 5
    FEM_CPU = wrapper_FEM(lEleClassic, ordreCPU)
    CPUsFEM.append(timeit(FEM_CPU, number=repe))
    print(CPUsFEM[-1] / repe)
    print('--------------------')

##
def FEMAdapt():
    zFEM = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp,player=player, losses=damping,  compute_method='FEM', l_ele = l_eles, order = orders)

print('CPU Adapt')
zFEMadapt = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp,player=player, losses=damping,   compute_method='FEM', l_ele=l_eles,order=orders)
CPUFEMAdapt = timeit(FEMAdapt, number=repe)
print(CPUFEMAdapt / repe)
##
print(str((1 - CPUFEMAdapt / CPUsFEM[2]) * 100) + '%')
CPUFEM = CPUsFEM[2] / repe
CPUFEMAdapt = CPUFEMAdapt / repe
print('--------------------')


##
def wrapper_TMM(subdiv):
    def TMM():
        zTMM = ImpedanceComputation(fs,'Tr_co_MP', temperature=temp,player=player, losses=damping,  compute_method='TMM', nb_sub=subdiv)
    return TMM

print('CPU TMM')
CPUTMM = []
for subdiv in subdivisions:
    TMMGoodVal = wrapper_TMM(subdiv)
    CPUTMM.append(timeit(TMMGoodVal, number=repe) / repe)
print('--------------------')


#%%
plt.figure(7)
plt.clf()
plt.loglog(errorsTMM, CPUTMM, marker = '+')
plt.loglog(errorsFEM,np.array(CPUsFEM) / repe , marker = 'o')
plt.loglog(errorFEMAdapt, CPUFEMAdapt, marker = 'x')
plt.grid()
plt.xlabel("relative L2 error")
plt.ylabel('CPU time')
plt.legend(['TMM','FEM','FEM adapt'])

plt.show()
