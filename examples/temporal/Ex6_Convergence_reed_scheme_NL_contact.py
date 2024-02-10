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
This file computes the convergence of the numerical schemes when their is a NL
contact force, for several strength of the contact force and several constant
used in the quadratization of this force.
"""


import numpy as np
import matplotlib.pyplot as plt

from openwind import simulate
from openwind.technical import Player



# %% Constants

Kc_list = [1e2, 1e4, 1e6, 1e8]
shape = [[0.0, .6, 5e-3, 5e-3, 'linear']]
temperature = 25

# player = Player('LIPS')
player = Player('LIPS_SCALED')
player.update_curve('contact_exponent',4)

# %% Loop on the force magnitude
for Kc in Kc_list:
    player.update_curve('contact_stifness', Kc)

    # C = 1
    # CFL = 0.05

    CFL_list = [0.32,.16, .08, .04, .02, .01]#, 0.005]#, 0.0025]#, 0.00125]
    C_list = [0, 1e-4, 1e-3, 5e-1, 1, 50, 1e3, 1e4]


    # %% Loop on the quadratization constant
    fig_conv, ax_conv = plt.subplots()
    ax_conv.loglog(np.logspace(-6,-4,5), np.logspace(-6,-4,5)**2 *1e9, 'k', label='ordre 2')
    ax_conv.loglog(np.logspace(-6,-4,5), np.logspace(-6,-4,5) * 1e2, 'k--', label='ordre 1')
    # fig_y, ax_y = plt.subplots()
    for C in C_list:
        print("=============================================================================\n"
              +f"{C}\n"
              +"=============================================================================")
        y_tot = list()
        time_tot = list()
        NRJ_max_tot = list()
        dt_list = list()

        for k, CFL in enumerate(CFL_list):
            rec  = simulate(0.13, shape, player=player, losses=False, temperature=20,
                            l_ele=0.1, order=2,
                            radiation_category='planar_piston',
                            theta_scheme_parameter=1/12,
                            contact_quadratization_cst=C,
                            cfl_alpha=CFL,
                            record_energy=True)

            signal_out= rec.values['bell_radiation_pressure']
            y = rec.values['source_y']
            time = rec.ts# - rec.dt

            assert(np.min(y)<0)
            dt_list.append(rec.dt)
            # ax_y.plot(time, y, '-o', label=f'CFL={CFL}; C={C}')
            # ax_y.legend()
            if k==0:
                time_tot.append(time)
            else:
                time_tot.append(np.interp(time_tot[0], time, time))
            y_tot.append(np.interp(time_tot[0], time, y))


            # Plot energy balance
            dt = rec.dt
            e_tot = rec.values['bell_radiation_E']+rec.values['bore0_E']+rec.values['source_E']
            dissip = rec.values['bore0_Q'] + rec.values['bell_radiation_Q']+rec.values['source_Q']
            relative_ener_var = (np.diff(e_tot)+dissip[1:])/(dt*np.max(np.abs(e_tot)))

            NRJ_max_tot.append(np.max(np.abs(relative_ener_var)))


        # norm
        norm = list()
        for k in range(len(CFL_list)-1):
            norm.append(np.linalg.norm(y_tot[k] - y_tot[k+1])/np.linalg.norm(y_tot[k+1]))



        ax_conv.loglog(dt_list[:-1], norm, '-o', label=f'C={C}')

    ax_conv.grid()
    ax_conv.set_ylabel('Norm on y')
    ax_conv.set_xlabel('dt')
    ax_conv.legend()
    ax_conv.set_xlim([min(dt_list), 2*max(dt_list)])
    ax_conv.set_ylim([0.5*min(norm), 2*max(norm)])
    fig_conv.savefig(f'Convergence_contact_Kc1e{np.log10(Kc)}.png')
