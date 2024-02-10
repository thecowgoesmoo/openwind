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

import matplotlib

"""
This file is related to the research report: A.Ernoult, 2023 "Effect of air
humidity and carbon dioxide in the sound propagation for the modeling of wind
musical instruments" RR-9500, Inria. 2023, pp.28. https://hal.inria.fr/hal-04008847

This file compare tabulated values and expressions founded in the literature for
the thermal conductivity of air (or other gases) .

The main papers used are:

    [1] W.M. Rohsenow, J.P. Hartnett and Y.I. Cho eds. 1998. Handbook of heat
        transfer. McGraw-Hill.

    [2] P.T. Tsilingiris 2008. "Thermophysical and transport properties of
        humid air at temperature range between 0 and 100°C". Energy Conversion
        and Management. 49(5), p.1098–1110. DOI:10.1016/j.enconman.2007.09.015
        https://www.sciencedirect.com/science/article/pii/S0196890407003329.

    [3] A.J. Zuckerwar and R.W. Meredith 1985. "Low‐frequency absorption of
        sound in air". The Journal of the Acoustical Society of America. 78(3),
        p.946–955. https://asa.scitation.org/doi/abs/10.1121/1.392927.

"""

# plotting options
colors = list(matplotlib.colors.TABLEAU_COLORS)
plt.close('all')
font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'


# plot and loop vectors
T = np.linspace(273.15, 313.15, 101)
HR = np.linspace(0, 1,501)
HR_loop = np.linspace(0, 1,6)
T_loop = np.linspace(273.15,313.15,5)


# %% References values

T0 = 273.15 #
T20 = T0 + 20
Patm = 101325
R = 8.31446261815324 # cste gaz parfait https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits
deg = lambda T: T - T0

Mdry = 28.9647e-3 # HANDBOOK chap.16.4.2, p.14 // masse molaire air sec https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits
Mv =  18.01534e-3 #https://fr.wikipedia.org/wiki/Eau

h = lambda T, HR: HR*10**(4.6142 -8073.0*T**(-1.261) + 0.3668/T + 100.35/T**2) # molar frac, combine Eq.(16.11) and Eq.(16.13.a), Handbook Zuck., Chap.16.3.3

xv_ref = h(T20,.5)


# %% Expressions for pure gases

# Dry air

# Chaigne and Kergomard
# kappa_chaigne = lambda T: 5.77e-5*(1 + .0033*(T-T0)) * 418.4#converted Cal/(cm.s.°C)
kappa_chaigne = lambda T: 2.57350e-2*(1 + 0.9075*(T/T20 - 1))

# Tsilingiris
poly_kappa_dry_tsi = np.array([2.47663035e-17, -1.066657e-13, 1.73550646e-10, -1.4815235e-7, 1.2598485e-4, -2.276501e-3])#*1e-3
kappa_dry_tsi = lambda T: np.polyval(poly_kappa_dry_tsi, T)
diff_kappa_dry = np.polyder(poly_kappa_dry_tsi, 2)
kappa_dry_tsi_lin = lambda T: 2.5562e-02*(1 + 0.8490*(T/T20 - 1) )#-0.2978*(T/T20 - 1)**2)
err_lin1 = np.max(np.abs(kappa_dry_tsi(T) - kappa_dry_tsi_lin(T))/kappa_dry_tsi(T))*100

# Water vapor
# Tsilingiris
# kappa_v_tsi = (17.61758242 + 5.558941059e-2*(T-T0) + 1.663336663e-4*(T-T0)**2)*1e-3
kappa_v_tsi = lambda T: (17.6 + 5.56e-2*(T-T0) + 1.66e-4*(T-T0)**2)*1e-3
d_kappa_v_tsi = lambda T: (5.56e-2 + 2*1.66e-4*(T-T0))*1e-3
kappa_v_lin = lambda T:  1.8778e-2*(1 +  0.972*(T/T20 -1))
# Rohsenow
temp_values_Rohsenow = np.linspace(273.15,373.15,11)
kappa_v_values_Rohsenow = np.array([17.6,18.2,18.8,19.4,20.1,20.9,21.6,22.3,23.1,23.9,24.8])*1e-3 # p.80 W/(m.°C) or J/(m.s.°C)

# Carbon
poly_kappa_c = np.array([2.68500151E-13, -4.75106178E-10, 3.14443715E-7, -1.33471677E-5, 2.971488E-3]) #Rohsenow p.59
kappa_c_roh  = lambda T: np.polyval(poly_kappa_c, T)
diff_kappa_c_roh  = lambda T: np.polyval(np.polyder(poly_kappa_c), T)
kappa_c_lin =  lambda T:  1.60945e-2*(1 + 1.3766*(T/T20 -1))

plt.figure()
plt.plot(deg(T), kappa_dry_tsi(T), label='Dry air Tsi')
plt.plot(deg(T), kappa_chaigne(T), '--', label='Dry air C&K')
plt.plot(deg(T), kappa_v_lin(T), label='water Tsi')
plt.plot(deg(temp_values_Rohsenow), kappa_v_values_Rohsenow,'*', label='Water Rohs.')
plt.plot(deg(T), kappa_c_lin(T), label='CO2 Rohs.')
plt.legend()
plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Thermal Conductivity $\kappa$ [W/(m.K)]')
plt.grid()
plt.tight_layout()
plt.savefig('Kappa_pure_temp.pdf')


# %% Tabulated value for humid air from Tsilingiris, Tab. 3

# [(temperature, [(viscosity, humidity rate)]

kappa_Gruss_Schmick = (80, [(2.9845, 0), (3.044, 15), (3.088, 32), (3.094, 37), (3.091, 42), (3.088, 48), (3.0949, 53), ( 3.062, 65), (3.074, 67), (2.9810, 95)])
kappa_Vargaftik = (80, [(2.989, 0), (3.069, 21), (3.103, 43), (3.078, 65), (3.019, 86)])
kappa_Touloukian = (60, [(2.92, 0), (2.96, 40), (2.96, 81), (2.92, 100)])


# %% Mixture

# Zuckerwar  eq.(12)
# kappa_Zuck = lambda T, xv: (60.054 + 1.846*T + 40*(xv) -1.775e-4*T*(xv) + 2.06e-6*T**2)*1e-8  *4.184e3 #converted kcal/(m.s.K)
# dxv_kappa_Zuck = lambda T, xv: (40 -1.775e-4*T)*1e-8  *4.184e3 #converted kcal/(m.s.K)
# dT_kappa_Zuck = lambda T, xv: (1.846 -1.775e-4*(xv) + 2*2.06e-6*T)*1e-8  *4.184e3 #converted kcal/(m.s.K)
kappa_Zuck = lambda T, xv: 2.5181e-2*(1 + 0.900*(T/T20 -1) + 0.0664*(xv - xv_ref) )

# From Tsilingiris
poly_mu_d = np.array([-5.79712299e-11, 1.2349703e-7,  -1.17635575e-4, 9.080125e-2, -9.8601e-1])*1e-6
eta_dry_tsi = lambda T: np.polyval(poly_mu_d, T)
eta_v_tsi = lambda T: 8.8582e-6*(1 + 1.3239*(T/T20 - 1))
def get_phi_coefs(T):
    """
    Phi coefficients from Tsilingiris eq.(22) and (23)
    """
    eta_av = eta_dry_tsi(T) / eta_v_tsi(T)
    M_av =  Mdry/Mv
    phi_av = np.sqrt(2)/4* 1/np.sqrt(1 + M_av) * (1 + np.sqrt(eta_av)/M_av**.25 )**2
    phi_va = np.sqrt(2)/4* 1/np.sqrt(1 + 1/M_av) * (1 + M_av**.25 / np.sqrt(eta_av))**2

    # phi_av = 0.21894 * (1 + 0.88806*np.sqrt(eta_av) )**2
    # phi_va = 0.27761 * (1 +  1.1260 / np.sqrt(eta_av))**2
    return phi_av, phi_va

def kappa_mix(T, xv):
    """Tsilingiris eq.(28)"""
    phi_av, phi_va = get_phi_coefs(T)
    kappa_tsi = ( (1 - xv)*kappa_dry_tsi(T) / ((1-xv) +  xv*phi_av)
               + xv*kappa_v_tsi(T) / (xv +  (1 - xv)*phi_va)
               )
    return kappa_tsi

# %% Figures

plt.figure()
plt.grid()
for k,temp in enumerate([60,80]):
    xv = h(temp+T0, HR)
    plt.plot(HR*100, kappa_Zuck(temp+T0, xv), '--',color=colors[k], label=f'Zuck. t={temp}°C')
    plt.plot(HR*100, kappa_mix(temp+T0, xv),  label=f'Tsi. t={temp}°C',color=colors[k])
kap, hv = tuple(zip(*kappa_Gruss_Schmick[1]))
plt.plot(hv, [a*1e-2 for a in kap],'*', color=colors[1], markersize=10, label='Gruss Schmick 80°C')
kap, hv = tuple(zip(*kappa_Vargaftik[1]))
plt.plot(hv, [a*1e-2 for a in kap],'o', color=colors[1], markersize=10, label='Vargaftik 80°C')
kap, hv = tuple(zip(*kappa_Touloukian[1]))
plt.plot(hv, [a*1e-2 for a in kap],'v', color=colors[0], markersize=10, label='Touloukian 60°C')

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r' $\kappa$ [W/(m.K)]')
plt.tight_layout()
plt.savefig('Kappa_mix_data.pdf')


plt.figure()
plt.plot(T, kappa_chaigne(T), 'k', label='Chaigne & Kergo.')
plt.plot(T, kappa_Zuck(T,0), label='Zuckerwar')
plt.plot(T, kappa_dry_tsi(T), label='Tsilingiris')
plt.plot(T, kappa_dry_tsi_lin(T),'--', label='lin from Tsi.')
plt.legend()
plt.grid()
plt.xlabel('Temperature [K]')
plt.ylabel('$\kappa_{dry}$ [W/(m.K)]')
plt.tight_layout()


T100 = np.linspace(0,100,100)+T0
plt.figure()
plt.grid()
for k,hv in enumerate(HR_loop):
    xv = h(T100, hv)
    plt.plot(T100-T0, (kappa_Zuck(T100, xv)), '--',color=colors[k])
    plt.plot(T100-T0, (kappa_mix(T100, xv)),  label=f'HR={100*hv:.0f}%',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Temperature [°C]')
plt.ylabel('$\kappa$ [W/(m.K)]')
plt.tight_layout()
plt.title('Repr. Fig.3 from Tsilingiris')



plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = h(temp, HR)
    plt.plot(HR*100, kappa_Zuck(temp, xv), '--',color=colors[k])
    plt.plot(HR*100, kappa_mix(temp, xv),  label=f't={deg(temp):.0f}°C',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r' $\kappa$ [W/(m.K)]')
plt.tight_layout()
plt.savefig('Kappa_mix_hv.pdf')


plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = h(temp, HR)
    plt.plot(HR*100, (kappa_Zuck(temp, xv)/kappa_Zuck(temp, 0) - 1)*100, '--',color=colors[k])
    plt.plot(HR*100, (kappa_mix(temp, xv)/kappa_mix(temp, 0) - 1)*100,  label=f'T={temp}K',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r' $100(\kappa/\kappa_{dry} - 1)$ [%]')
plt.tight_layout()



plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = np.linspace(0,5e-2,100)
    plt.plot(xv*100, (kappa_Zuck(temp, xv)/kappa_Zuck(temp, 0) - 1)*100, '--',color=colors[k])
    plt.plot(xv*100, (kappa_mix(temp, xv)/kappa_mix(temp, 0) - 1)*100,  label=f'T={temp}K',color=colors[k])

# plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Molar frac x_v [in %]')
plt.ylabel(r' $100(\kappa/\kappa_{dry} - 1)$ [%]')
plt.tight_layout()
