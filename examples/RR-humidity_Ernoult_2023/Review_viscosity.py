#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 17:09:39 2023

@author: augustin
"""
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
the viscosity of air (or other gases) .

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
eta_chaigne = lambda T: 1.708e-5*(1 + .0029*(T-T0)) #  Chaigne and Kergomard

# Tsilingiris
poly_mu_d = np.array([-5.79712299e-11, 1.2349703e-7,  -1.17635575e-4, 9.080125e-2, -9.8601e-1])*1e-6
eta_dry_tsi = lambda T: np.polyval(poly_mu_d, T)
poly_mu_adim = np.array([  0.43420097,  -3.15533074,  10.2526742 , -26.99606134,   1.])
eta_dry_tsi_adim = lambda T: -9.8601e-7*np.polyval(poly_mu_adim, T/T20)
eta_dry_tsi_lin1 = lambda T: 1.8206e-05*( 1 + 0.7701*(T/T20-1))
err_lin1 = np.max(np.abs(eta_dry_tsi(T) - eta_dry_tsi_lin1(T))/eta_dry_tsi(T))*100
print(f'Relative error between complete and linearised expression: {err_lin1:.2f}%')

# Water vapor
# tabulated values from Rohsenow # Chap2.26, Tab;2.15, (p.80)
temp_values_Rohsenow = np.linspace(273.15,373.15,11)
eta_v_values_Rohsenow = np.array([80.4,84.5,88.5,92.6,96.6,100,105,109,113,117,121])*1e-7

# from Tsilingiris
eta_v_tsi_orig = lambda T: (80.58131868 + 0.4000549451*(T-T0))*1e-7 #
eta_v_tsi = lambda T: 8.8582e-6*(1 + 1.3239*(T/T20 - 1))

# Carbon dioxide
poly_mu_c = np.array([-1.47315277E-12, 9.843776E-9, -2.824853E-5, 6.0395329E-2,-8.095191E-1])*1e-6 #Rohsenow p.59
eta_c_roh  = lambda T: np.polyval(poly_mu_c, T)
diff_eta_c = lambda T: np.polyval(np.polyder(poly_mu_c), T)
ceof_lin = diff_eta_c(T20)/eta_c_roh(T20)*T20
eta_c_lin = lambda T: 1.4705e-05*(1 + 0.921*(T/T20 - 1))

plt.figure()
plt.plot(deg(T), eta_dry_tsi(T), label='Dry air Tsi.')
plt.plot(deg(T), eta_chaigne(T), '--', label='Dry air, C&K')
plt.plot(deg(T), eta_v_tsi(T), label='Water Vapor Tsi.')
plt.plot(deg(temp_values_Rohsenow), eta_v_values_Rohsenow,'*', label='Water Vapor, Rohs.')
plt.plot(deg(T), eta_c_roh(T), label='CO2 Rohs.')
plt.legend()
plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('viscosity $\mu$ [kg/(m.s)]')
plt.grid()
plt.tight_layout()
plt.savefig('eta_pure_temp.pdf')

# %% Tabulated value for humid air from Tsilingiris, Tab. 2

# [(temperature, [(viscosity, humidity rate)]

# Kestin and Whitelaw
eta_KW = [(25, [(18.451, 19), (18.446, 25.5), (18.441, 35), (18.419, 51), (18.2, 54)]),
          (50, [(19.593, 15.5), (19.591,20), (19.575, 25), (19.539, 34), (19.474, 51), (19.247, 98)]),
          (75, [(20.632, 14), (20.588, 20), (20.497, 25), (20.357, 35), (20.046, 51), (19.586, 70), (19.252, 83), (18.792, 97), (18.781, 100)]),
          ]
t_KW = [t[0] for t in eta_KW]

# Hochrainer and Munczak
eta_HM = [(20, [(18.176,0), (18.150, 62), (18.136, 82), (18.134, 93)]),
          (30, [(18.647, 0), (18.620, 41), (18.617, 61), (18.586, 81), (18.569, 92)]),
          (40, [(19.111, 0), (19.111, 0), (19.080, 41), (19.053, 63), (19.017, 83), (18.995,  93)]),
          (50, [(19.588, 0), (19.553, 21), (19.483, 41), (19.426, 61), (19.363, 82), (19.343, 93)])
          ]
t_HM = [t[0] for t in eta_HM]

# Vargaftik
eta_V = [(50, [(19.550, 0), (19.140, 82)]),
         (60, [(20.01, 0), (19.600, 51), (19.04, 100)]),
         (70, [(20.46, 0), (20.05, 32), (19.500, 65), (18.800, 97)]),
         (80, [(20.910, 0), (20.510, 21), (19.950, 43), (19.250, 64), (18.430, 86), (17.500, 100)]),
         (90, [(21.350, 0), (20.950, 14), (20.350, 29), (19.690, 43), (18.870, 58), (17.920, 72), (16.890, 87), (15.770, 100)]),
         (100, [(21.800, 0), (21.400, 10), (20.840, 20), (20.140, 30), (19.310, 40), (18.360, 50), (17.320, 60), (16.180, 70), (14.990, 80), (13.750, 90), (12.470, 100)]),
        ]
t_V = [t[0] for t in eta_V]

#Mason and Monchick
eta_MM = [(25, [(18.300, 0), (18.000, 100)]),
         (50, [(19.550, 0), (19.100, 82), (19.070, 100)]),
         (75, [(20.850, 0), (20.170, 26), (19.660, 53), (19.100, 79), (18.400, 100)]),
        ]
t_MM = [t[0] for t in eta_MM]


eta_Roh = [()]

t_data = [20,25,30,40,50,60,70,75,80,90,100]




# %% Mixing rules

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


def eta_mix(T, xv):
    """
    mixing rule from Tsilingiris eq.(20)
    """
    phi_av, phi_va = get_phi_coefs(T)

    eta_tsi = ( (1 - xv)*eta_dry_tsi(T) / ((1-xv) +  xv*phi_av)
               + xv*eta_v_tsi(T) / (xv +  (1 - xv)*phi_va)
               )
    return eta_tsi

def eta_Zuck(T, xv):
    """
    from "low-freq absoprtion of sound in air", Zuckerwar 1985; refering to Mason 1965

    Simplified

    """
    # return (84.986 + 7*T + 113.157*xv - T*xv - 3.7501e-3*T**2 -100.015*xv**2)*1e-8
    # dxv_zuck = lambda T, xv: (113.157 - T -2*100.015*xv)*1e-8
    # dT_zuck = lambda T, xv: (7- xv - 2*3.7501e-3*T)*1e-8
    # dTdxv_zuck = lambda T, xv: -1e-8
    return 1.813e-05*(1 + 0.77*(T/T20 -1) -0.10*(xv - xv_ref))


plt.figure()
plt.grid()
for k,temp in enumerate([60,70,75,80,90,100]):
    xv = h(temp+T0, HR)
    plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
    plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
    if temp in t_KW:
        ind = t_KW.index(temp)
        mu, hv = tuple(zip(*eta_KW[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'*', color=colors[k], markersize=10)
    if temp in t_HM:
        ind = t_HM.index(temp)
        mu, hv = tuple(zip(*eta_HM[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'o', color=colors[k], markersize=10)
    if temp in t_V:
        ind = t_V.index(temp)
        mu, hv = tuple(zip(*eta_V[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'v', color=colors[k], markersize=10)
    if temp in t_MM:
        ind = t_MM.index(temp)
        mu, hv = tuple(zip(*eta_MM[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'s', color=colors[k], markersize=10)
plt.xlim([-0.1,101])
# plt.ylim([1.75e-5,2e-5])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
plt.title('High temperature comparison')
plt.tight_layout()


plt.figure()
plt.grid()
for k,temp in enumerate([20,25,30,40,50]):
    xv = h(temp+T0, HR)
    plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
    plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
    if temp in t_KW:
        ind = t_KW.index(temp)
        mu, hv = tuple(zip(*eta_KW[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'*', color=colors[k], markersize=10)
    if temp in t_HM:
        ind = t_HM.index(temp)
        mu, hv = tuple(zip(*eta_HM[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'o', color=colors[k], markersize=10)
    if temp in t_V:
        ind = t_V.index(temp)
        mu, hv = tuple(zip(*eta_V[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'v', color=colors[k], markersize=10)
    if temp in t_MM:
        ind = t_MM.index(temp)
        mu, hv = tuple(zip(*eta_MM[ind][1]))
        plt.plot(hv, [a*1e-6 for a in mu],'s', color=colors[k], markersize=10)
plt.xlim([-0.1,101])
plt.ylim([1.75e-5,1.975e-5])
plt.legend(loc=3)
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
plt.tight_layout()
plt.title('Low temperature comparison')
plt.savefig('eta_hv_data.pdf')

# Comparison for each set of data indepently
# plt.figure()
# plt.grid()
# for k,temp in enumerate(t_KW):
#     xv = h(temp+T0, HR)
#     plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
#     plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
#     if temp in t_KW:
#         ind = t_KW.index(temp)
#         mu, hv = tuple(zip(*eta_KW[ind][1]))
#         plt.plot(hv, [a*1e-6 for a in mu],'*', color=colors[k])

# plt.xlim([0,100])
# # ax_temp.set_ylim([0,0.4])
# plt.title('Kestin and Whitelaw')
# plt.legend()
# plt.xlabel('Humidity rate [in %]')
# plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
# plt.tight_layout()

# plt.figure()
# plt.grid()
# for k,temp in enumerate(t_HM):
#     xv = h(temp+T0, HR)
#     plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
#     plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
#     mu, hv = tuple(zip(*eta_HM[k][1]))
#     plt.plot(hv, [a*1e-6 for a in mu],'*', color=colors[k])

# plt.xlim([0,100])
# # ax_temp.set_ylim([0,0.4])
# plt.title('Hochrainer and Munczak')
# plt.legend()
# plt.xlabel('Humidity rate [in %]')
# plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
# plt.tight_layout()


# plt.figure()
# plt.grid()
# for k,temp in enumerate(t_V):
#     xv = h(temp+T0, HR)
#     plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
#     plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
#     mu, hv = tuple(zip(*eta_V[k][1]))
#     plt.plot(hv, [a*1e-6 for a in mu],'*:', color=colors[k])

# plt.xlim([0,100])
# # ax_temp.set_ylim([0,0.4])
# plt.title('Vargaftik')
# plt.legend()
# plt.xlabel('Humidity rate [in %]')
# plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
# plt.tight_layout()


# plt.figure()
# plt.grid()
# for k,temp in enumerate(t_MM):
#     xv = h(temp+T0, HR)
#     plt.plot(HR*100, (eta_Zuck(temp+T0, xv)), '--',color=colors[k])
#     plt.plot(HR*100, (eta_mix(temp+T0, xv)),  label=f't={temp}°C',color=colors[k])
#     mu, hv = tuple(zip(*eta_MM[k][1]))
#     plt.plot(hv, [a*1e-6 for a in mu],'*:', color=colors[k])

# plt.xlim([0,100])
# # ax_temp.set_ylim([0,0.4])
# plt.title('Mason and Monchick')
# plt.legend()
# plt.xlabel('Humidity rate [in %]')
# plt.ylabel(r'$\eta(T)$ [kg/(m.s)]')
# plt.tight_layout()


# %% Figures

plt.figure()
plt.plot(T, eta_chaigne(T), 'k', label='Chaigne & Kergo.')
plt.plot(T, eta_Zuck(T, 0), label='Zuckerwar')
plt.plot(T, eta_dry_tsi(T), label='Tsilingiris')
plt.plot(T, eta_dry_tsi_lin1(T),'--', label='lin from Tsi.')
plt.legend()
# plt.xlim([0,100])
plt.ylabel(r'$\eta_{dry}(T)$ [kg/(m.s)]')
plt.xlabel(r'Temperature [K]' )#+' \n' + r' $100(c/c_{dry} - 1)$')
plt.tight_layout()
plt.title('Dry air comparison')


T100 = np.linspace(0,100,100)+T0
plt.figure()
plt.grid()
for k,hv in enumerate(HR_loop):
    xv = h(T100, hv)
    plt.plot(T100-T0, (eta_Zuck(T100, xv)), '--',color=colors[k])
    plt.plot(T100-T0, (eta_mix(T100, xv)),  label=f'HR={100*hv:.0f}%',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Temperature [°C]')
plt.ylabel(r'$\eta$ [kg/(m.s)]')
plt.tight_layout()
plt.title('Repr. Fig.2 from Tsilingiris')



plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = h(temp, HR)
    plt.plot(HR*100, (eta_Zuck(temp, xv)), '--',color=colors[k])
    plt.plot(HR*100, (eta_mix(temp, xv)),  label=f'T={temp}K',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r'$\eta$ [kg/(m.s)]')
plt.tight_layout()


plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = h(temp, HR)
    plt.plot(HR*100, (eta_Zuck(temp, xv)/eta_Zuck(temp, 0) - 1)*100, '--',color=colors[k])
    plt.plot(HR*100, (eta_mix(temp, xv)/eta_mix(temp, 0) - 1)*100,  label=f'T={temp}K',color=colors[k])

plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Humidity rate [in %]')
plt.ylabel(r' $100(\eta/\eta_{dry} - 1)$ [%]')
plt.tight_layout()



plt.figure()
plt.grid()
for k,temp in enumerate(T_loop):
    xv = np.linspace(0,5e-2,100)
    plt.plot(xv*100, (eta_Zuck(temp, xv)/eta_Zuck(temp, 0) - 1)*100, '--',color=colors[k])
    plt.plot(xv*100, (eta_mix(temp, xv)/eta_mix(temp, 0) - 1)*100,  label=f'T={temp}K',color=colors[k])

# plt.xlim([0,100])
# ax_temp.set_ylim([0,0.4])
plt.legend()
plt.xlabel('Molar frac x_v [in %]')
plt.ylabel(r' $100(\eta/\eta_{dry} - 1)$ [%]')
plt.tight_layout()
