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
Plot energy exchanges during the impulse response of a cylindrical pipe.
"""

import matplotlib.pyplot as plt
import numpy as np

from openwind import simulate, Player

#%% Run simulation: ~1 minute, depending on your CPU

# It takes more time than a simple simulation
# because it needs to calculate energy at each time step.

shape = 'simplified-trumpet'
rec = simulate(0.05,
               shape,
               player=Player("IMPULSE_400us"),
               losses='diffrepr', temperature=20,
               l_ele=0.04, order=10,
               radiation_category='perfectly_open',
               cfl_alpha=1.0,
               record_energy=True)

# Result is stored in rec
print(rec)



#%% Display the results


# Plot energy exchanges
plt.figure("Energy exchanges", figsize=(4.5,3))
ax = plt.axes()

dissip_bore = np.cumsum(rec.values['bore0_Q']+rec.values['bore1_Q'])
dissip_rad = np.cumsum(rec.values['bell_radiation_Q'])
dissip_cumul = dissip_bore + dissip_rad
source_cumul = np.cumsum(-rec.values['source_Q'])
e_bore = rec.values['bore0_E']+rec.values['bore1_E']
e_additional_vars = sum(rec.values['bore'+str(i)+'_E_'+var] for i in [0,1] for var in ['P0', 'Pi', 'Vi'])
e_tot = rec.values['bell_radiation_E']+e_bore

plt.plot(rec.ts, e_bore, label='$\mathcal{E}_h^n$')
#plt.plot(rec.ts, e_additional_vars, '-.', label=r'$\mathcal{E}_{h,v\theta}^n$')
plt.plot(rec.ts, e_additional_vars, '-.', label=r'$\mu \mathcal{E}_{h,{visc}}^n + \mathcal{E}_{h,{therm}}^n$')
# plt.plot(rec.ts, rec.values['bell_radiation_E'], label='$E_{radiation}$')
plt.plot(rec.ts, dissip_bore, ':', color='xkcd:grey', label='$\sum Q_h$')
# plt.plot(rec.ts, dissip_rad, label='$\int Q_{radiation}$ (radiated energy)')
plt.plot(rec.ts, e_tot+dissip_cumul, label='$\mathcal{E}_h^n + \sum Q_h$',
         marker='+', markevery=0.1)
plt.plot(rec.ts, source_cumul, label="$\sum \mathcal{S}_h$",
         linestyle='--', color='k',
         marker='x', markevery=0.11)
plt.xlabel("$t$ (s)")
plt.ylabel("Numerical energy")
plt.grid()
plt.legend()
ax.ticklabel_format(axis='y', scilimits=(0, 0))
plt.tight_layout()
plt.savefig('Figure5.pdf')



# Plot energy deviation
ref_energy = source_cumul[-1]

plt.figure("Energy balance", figsize=(4,2))
dt = rec.ts[1] - rec.ts[0]
plt.plot(rec.ts[:-1], np.diff(e_tot + dissip_cumul - source_cumul)/ref_energy, '.')
plt.xlabel("$t$ (s)")
plt.ylabel("$\Delta t (\delta \mathcal{E}_h + Q_h - \mathcal{S}_h) / \mathcal{E}_{max}$")
plt.tight_layout()
plt.savefig('Figure6.pdf')


#%% Discretization information

t_pipes = rec.t_solver.t_pipes
elements = sum(len(tp.mesh.elements) for tp in t_pipes)
n_dof = sum(tp.nH1 + tp.nL2 for tp in t_pipes)
element_lengths = np.concatenate([tp.mesh.get_lengths() for tp in t_pipes])
print(elements, "elements")
print("Element length is between", min(element_lengths), "and", max(element_lengths))
print(n_dof, "degrees of freedom")
