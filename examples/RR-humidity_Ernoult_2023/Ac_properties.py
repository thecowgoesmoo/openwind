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
In this file, the evolution of several acoustic quantities (speed of sound,
heat capacities, density, etc) w.r. to the temperature, the humidity rate and
the fraction of CO2.

This file is related to the research report: A.Ernoult, 2023 "Effect of air
humidity and carbon dioxide in the sound propagation for the modeling of wind
musical instruments" RR-9500, Inria. 2023, pp.28. https://hal.inria.fr/hal-04008847
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

colors = list(matplotlib.colors.TABLEAU_COLORS)

plt.close('all')
font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'



T = np.linspace(273.15, 313.15, 101)
HR = np.linspace(0, 1,501)
HR_loop = np.linspace(0, 1,6)
T_loop = np.linspace(273.15,313.15,5)
# T_loop = np.linspace(273.15,303.15,4)
xc = np.linspace(0, 0.1, 100)

T0 = 273.15 #
T20 = T0 + 20
Patm = 101325
R = 8.31446261815324 # cste gaz parfait https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits
xo_amb = .20946 # Ambiant portion of O2 in air: Cramer Tab.1
xc_amb = 420e-6 # Ambiant portion of CO2 in air: Nasa website
deg = lambda T: T- T0 # Conversion Kelvin to °C


# %% Cramer fit
def c_cramer(T, xv, xc=4e-4):
    t = T - T0
    a0, a1, a2 = (331.5024, 0.603055, -.000528)
    a3, a4, a5 = (51.471935, .1495874,-.000782)
    a6, a7, a8 = (-1.82e-7, 3.73e-8, -2.93e-10)
    a9, a10, a11 = (-85.20931, -.228525, 5.91e-5)
    a12, a13, a14, a15 = (-2.835149, -2.15e-13, 29.179762, 4.86e-4)

    c = (a0 + a1*t + a2*t**2
         + xv*(a3 + a4*t + a5*t**2)
         + Patm*(a6 + a7*t +a8*t**2)
         + xc*(a9 + a10*t + a11*t**2)
         + a12*xv**2
         + a13*Patm**2
         + a14*xc**2
         + a15*xv*xc*Patm
         )
    return c

def gamma_cramer(T, xv, xc):
    t = T - T0
    a0, a1, a2 = (1.400822, -1.75e-5, -1.73e-7)
    a3, a4, a5 = (-.0873629, -1.665e-4, -3.26e-6)
    a6, a7, a8 = (2.047e-8, -1.26e-10, 5.939e-14)
    a9, a10, a11 = (-.1199717, -8.693e-4, 1.979e-6)
    a12, a13, a14, a15 = (-.01104, -3.478e-16, .0450616, 1.82e-6)

    gamma = (a0 + a1*t + a2*t**2
         + xv*(a3 + a4*t + a5*t**2)
         + Patm*(a6 + a7*t +a8*t**2)
         + xc*(a9 + a10*t + a11*t**2)
         + a12*xv**2
         + a13*Patm**2
         + a14*xc**2
         + a15*xv*xc*Patm
         )
    return gamma

# %% Chaigne & kergomard values

c_chaigne = lambda T: 331.45*np.sqrt(T/T0)
gamma_chaigne = lambda T: 1.402*np.ones_like(T)
Cp_mass_chaigne = lambda T: .24*4.184e3*np.ones_like(T)
Cp_chaigne = lambda T: .24*4.184e3*Mdry()*np.ones_like(T)
rho_chaigne = lambda T: 1.2929*T0/T


# %% Reference value of pure gases

def get_virial_KL(a, b, c):
    r"""
    Apply expression from Kay & Laby for the second Virial coefficient

    .. math::
        B(T) = a - b \times \exp\left( \frac{c}{T}\right)

    It return B(T) and its first and second derivative with respect to T

    Parameters
    ----------
    a, b, c : float
        The value of the coefficeints.

    Returns
    -------
    B, dB, d2B : callable
        The 2nd virial coefficient, its first and second derivative w.r. to T

    """

    B = lambda T: a - b*np.exp(c/T)
    dB = lambda T: b*c*np.exp(c/T)/T**2
    d2B = lambda T: -b*c*(c +2*T)*np.exp(c/T)/T**4
    return B, dB, d2B

def get_ac_quantities_real(Cp0, B, dB, d2B, M):
    r"""
    Acoustic quantities for a real Gas with respect to the temperature and composition

    The formula given in the research report are used. For the molar heat capacity :math:`C_p`
    a,d the heat capacity ratio :math:`\gamma`

    .. math::
        C_p(T, x) = C_p^0(T, x) - P_{atm} T \frac{d^2B}{dT^2}(T, x) \\
        \tilde{R}(T, x) = R +2  P_{atm} \frac{dB}{dT}(T, x) \\
        \gamma(T, x) = 1 + \frac{1}{(C_p(T, x)/\tilde{R}) - 1}

    Where :math:`T` is the temperature (K) and :math:`x` are other variables
    such as the air composition. For the speed of sound :math:`c` and the
    density :math:`rho`:

    .. math::
        c^2 (T,x) = \frac{\gamma(T,x)}{M(x)} (RT + 2 P_{atm} B(T,x)) \\
        rho(T,x) =  M(x) \frac{P_{atm}}{RT} \times \left( 1 - B(T,x) \frac{P_{atm}}{RT}\right)

    Parameters
    ----------
    Cp0 : callable
        molar heat capacity for ideal gas.
    B, dB, d2B : callable
        2nd virial coeff and its first and 2nd derivative.
    M : callable
        Molar mass

    Returns
    -------
    Cp : callable
        molar heat capacity for real gas
    gamma : callable
        Heat capacity ratio for real gas.


    """

    Cp = lambda T, *x: Cp0(T, *x) - Patm * T *d2B(T, *x)
    gamma = lambda T, *x: 1 + 1 / ( Cp(T, *x)/(R + 2*Patm*dB(T, *x)) -1)
    c = lambda T, *x: np.sqrt(gamma(T, *x) /M(*x) * (R*T + 2*Patm*B(T,*x)))
    rho = lambda T, *x: Patm*M(*x)/(R*T)*(1 - B(T,*x)*Patm/(R*T))
    return Cp, gamma, c, rho


# Dry air
Mdry = lambda : 28.9647e-3 # HANDBOOK chap.16.4.2, p.14 // masse molaire air sec https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits
p_dry = np.flip([ 29.6170, -5.0950e-03,  1.1648e-05, -1.4104e-09]) # Touloukian p.293
Cp0_dry = lambda T: np.polyval(p_dry, T)
ad, bd, cd = (152.2e-6, 111.3e-6, 108.1) # Zuckerwar, fit of a lot of refereence, (see p.16)
Bd, dBd, d2Bd = get_virial_KL(ad, bd, cd)
Cp_dry , gamma_dry = get_ac_quantities_real(Cp0_dry, Bd, dBd, d2Bd, Mdry)[:2]

# water
Mv = lambda : 18.01534e-3 #https://fr.wikipedia.org/wiki/Eau
p_v = np.flip([ 3.40865e+01, -9.7404e-03,  3.1432e-05, -1.5105e-08]) # Touloukian p.105
Cp0_v = lambda T: np.polyval(p_v, T)
av, bv, cv =  (33.0e-6 , 15.2e-6 , 1300.7) # from Kaye and Laye  (31.5e-6, 13.6e-6, 1375.3)
Bv, dBv, d2Bv = get_virial_KL(av, bv, cv)
Cp_v , gamma_v = get_ac_quantities_real(Cp0_v, Bv, dBv, d2Bv, Mv)[:2]

# Carbon dioxide
Mc = lambda : 44.0095e-3#*0.96 # https://en.wikipedia.org/wiki/Carbon_dioxide ??? 44.009 95
p_c = np.flip([ 1.95026e+01,  7.4308e-02, -5.5836e-05,  1.5273e-08]) # Touloukian p.145
Cp0_c = lambda T:  np.polyval(p_c, T)
ac, bc, cc = (1.508e-04, 9.778e-05, 3.079e+02) # fit from Hilsenrath p.151 and Sengers p.54
Bc, dBc, d2Bc = get_virial_KL(ac, bc, cc)
Cp_c , gamma_c = get_ac_quantities_real(Cp0_c, Bc, dBc, d2Bc, Mc)[:2]

# dioxygen
Mo = lambda : 31.9988e-3 # Cramer
p_o = np.flip([ 2.97329e+01, -1.0299e-02,  3.7322e-05, -2.2774e-08]) # Touloukian p.50
Cp0_o = lambda T: np.polyval(p_o, T)
ao, bo, co = (152.8e-6, 117.0e-6, 108.8) # Kaye Laby
Bo, dBo, d2Bo = get_virial_KL(ao, bo, co)
Cp_o, gamma_o = get_ac_quantities_real(Cp0_o, Bo, dBo, d2Bo, Mo)[:2]


# %% mixture

# Molar fraction of Water vapor from relative humidity rate
h = lambda T, HR: HR*10**(4.6142 -8073.0*T**(-1.261) + 0.3668/T + 100.35/T**2) # molar frac, combine Eq.(16.11) and Eq.(16.13.a), Handbook Zuck., Chap.16.3.3

# effective molar mass
M_eff = lambda xv, xc: (Mdry())*(1-xv)  + (xc - xc_amb)*(Mc() - Mo())  + xv*Mv()

# Interaction Virial coeff, from Zuck 16.4.4, p.16
adv, bdv, cdv = (224.0e-6, 184.6e-6, 94.6)
Bdv, dBdv, d2Bdv = get_virial_KL(adv, bdv, cdv)

# Mixture properties
def get_mixture_virial(Dry, Water, Carbon, Oxygen, Dry_Water):
    mixture = lambda T, xv, xc: (Dry(T)*(1 -xv)**2 + 2*Dry_Water(T)*(1-xv)*xv + Water(T)*xv**2  # air water mixture with interaction
                                 + (Carbon(T) - Oxygen(T))*(xc - xc_amb)**2 - 2*(xc - xc_amb)*xo_amb*Oxygen(T)) # replacement of O2 by CO2 without interaction
    return mixture

B_mix   = get_mixture_virial(Bd, Bv, Bc, Bo, Bdv)
dB_mix  = get_mixture_virial(dBd, dBv, dBc, dBo, dBdv)
d2B_mix = get_mixture_virial(d2Bd, d2Bv, d2Bc, d2Bo, d2Bdv)
Cp0_mix = lambda T, xv, xc:  (Cp0_dry(T))*(1 -xv)  + (Cp0_c(T) - Cp0_o(T))*(xc- xc_amb) + Cp0_v(T)*xv


# Application of real gas formulae
Cp_mix, gamma_mix, c_mix, rho_mix = get_ac_quantities_real(Cp0_mix, B_mix, dB_mix, d2B_mix, M_eff)

# Mass heat capacity (specific heat)
Cp_mass_mix = lambda T, xv, xc: Cp_mix(T, xv, xc)/M_eff(xv, xc)



# %% dispersion

"""
Effect of the dispersion due to the relaxation frequency of O2 and N2.

[1] H.E. Bass, L.C. Sutherland, A.J. Zuckerwar, D.T. Blackstock and D.M. Hester 1995. "Atmospheric absorption of sound: Further developments". The Journal of the Acoustical Society of America. 97(1), p.680–683. DOI:10.1121/1.412989 HAL:http://asa.scitation.org/doi/10.1121/1.412989.
[2] C.L. Morfey and G.P. Howell 1980. "Speed of sound in air as a function of frequency and humidity". The Journal of the Acoustical Society of America. 68(5), p.1525–1527. DOI:10.1121/1.385080 HAL:http://asa.scitation.org/doi/10.1121/1.385080.


"""

def relax_freq(T, xv):
    h = 100*xv
    frO = 24 + 4.04e4*h*(.02+h)/(.391+h) # relaxation freq of O2 ; Atm. absorption of sound: Further dev., JASA 1995
    frN = np.sqrt(T20/T)*(9 + 280*h*np.exp(-4.17*((T20/T)**(1/3) -1)))
    return frO, frN

def c_disp(T, xv, f):
    """ Eq.(2) of Morfeay & Howell
    """
    frO, frN = relax_freq(T, xv)
    A_O2 = .01278/(2*np.pi)* np.exp(-2239.1/T)
    A_N2 = .1068/(2*np.pi) * np.exp(-3352/T)
    alpha_O =  A_O2 * f**2  / (frO**2 + f**2)
    alpha_N = A_N2 * f**2  / (frN**2 + f**2)
    c_disp =1/( 1/c_mix(T,xv, xc_amb) -alpha_O -alpha_N)
    return c_disp

f_loop = [100,500,5000,1e4]
xv_20 = h(T20, HR)

plt.figure()
plt.grid()
plt.plot(HR*100, (c_mix(T20, xv_20, xc_amb)/c_mix(T20, 0, xc_amb) - 1)* 1e4, 'k', label='wo disp: f=0 Hz')
for F in f_loop:
    plt.plot(HR*100, (c_disp(T20, xv_20, F)/c_mix(T20, 0, xc_amb) - 1)*1e4, label=f'f = {F*1e-3:>4.1f} kHz')
plt.xlim([0,100])
plt.ylim([-5,40])
plt.legend()
plt.xlabel('Humidity rate [%]')

plt.ylabel(r'$(c_{disp}(x_v, f)/c_{eff}(x_v)-1)\times 10^{-4}$' )
plt.tight_layout()
plt.savefig('Dev_c_percent_disp.pdf')


freq = np.linspace(0,5e3,1000)
plt.figure()
plt.grid()
for truc in np.append([.1], HR_loop[1:]):
    chose = h(T20, truc)
    plt.plot(freq*1e-3, (c_disp(T20, chose, freq)/c_mix(T20, chose, xc_amb) - 1)*1e2, label=f'$h_v$={truc*100:>3.0f} %')
plt.xlim([0,5])
plt.ylim([0,2.5e-2])
plt.legend()
plt.xlabel('Frequency [kHz]')
plt.ylabel('Celerity deviation [in %]' )
plt.tight_layout()
plt.savefig('Dev_c_freq_disp.pdf')


plt.figure()
plt.grid()
for temp in T_loop:
    chose = h(temp, .5)
    plt.plot(freq*1e-3, (c_disp(T20, chose, freq)/c_mix(T20, chose, xc_amb) - 1)*1e2, label=f't={deg(temp):.0f} °C')
plt.xlim([0,5])
plt.ylim([0,2.5e-2])
plt.legend()
plt.xlabel('Frequency [kHz]')
plt.ylabel('Celerity deviation [in %]' )
plt.tight_layout()
plt.savefig('Dev_c_freq_disp_temp.pdf')


# =============================================================================
# # %% Taylor development
# =============================================================================

# Reference point
Tref = T20  # Temperature
xv_ref = h(Tref, 0.5) # Molar frac of water
xc_ref = xc_amb # Molar frac of CO2
dx = 1e-5# step for finite diff



# %%% Molar frac of water and relative humidity rate

h_lin = lambda T, HR:HR*10**(5.21899 -5.8294*T20/T -1.0252*(T20/T)**2)

plt.figure()
for temp in T_loop:
    plt.plot(100*HR, 100*h(temp, HR), label=f't={deg(temp):.0f}°C')
    plt.plot(100*HR, 100*h_lin(temp, HR), 'k--')
plt.xlabel('Humidity rate $h_v$ [%]')
plt.ylabel('Molar frac. $x_v$ [%]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlim([0,100])
plt.ylim([0,8])
plt.savefig('Humidity_molar_frac_temp.pdf')

plt.figure()
for temp in T_loop:
    plt.plot(100*HR, h_lin(temp, HR)/h(temp, HR)-1, label=f't={deg(temp):.0f}°C')
plt.xlabel('Humidity rate $h_v$ [%]')
plt.ylabel('Relative error $h_{lin}/h_v$ [%]')
plt.legend()
plt.grid()
plt.tight_layout()
plt.xlim([0,100])


# %%% derivatives

"""
All the expressions are differentiate w.r to T, xv, and xc
"""

dM_eff_dxv = lambda xv, xc: np.ones_like(xv)*(Mv() - (Mdry() + (xc - xc_amb)*(Mc() - Mo()) ))
dM_eff_dxc = lambda xv, xc: np.ones_like(xc)*(Mc() - Mo())#*(1 - xv)

# Virial w.r. to T
def virial_derivatives(a,b,c):
    d3B = lambda T:  b*c/(T**6)*np.exp(c/T) * (c**2 + 6*c*T + 6*T**2)
    d4B = lambda T: -b*c/(T**8)*np.exp(c/T) * (c**3 + 12*c**2*T + 36*c*T**2 + 24*T**3)
    return d3B, d4B

d3Bd, d4Bd = virial_derivatives(ad, bd, cd)
d3Bv, d4Bv = virial_derivatives(av, bv, cv)
d3Bc, d4Bc = virial_derivatives(ac, bc, cc)
d3Bo, d4Bo = virial_derivatives(ao, bo, co)
d3Bdv, d4Bdv = virial_derivatives(adv, bdv, cdv)

d3B_mix = get_mixture_virial(d3Bd, d3Bv, d3Bc, d3Bo, d3Bdv)
d4B_mix = get_mixture_virial(d4Bd, d4Bv, d4Bc, d4Bo, d4Bdv)

# Virial w.r. to xv and xc
B_mix_dxv = lambda T, xv, xc: -2*Bd(T)*(1 -xv) + 2*Bdv(T)*(1-2*xv) + 2*Bv(T)*xv             + np.zeros_like(xc)
dB_mix_dxv = lambda T, xv, xc: -2*dBd(T)*(1 -xv) + 2* dBdv(T)*(1-2*xv) + 2*dBv(T)*xv        + np.zeros_like(xc)
d2B_mix_dxv = lambda T, xv, xc: -2*d2Bd(T)*(1 -xv) + 2* d2Bdv(T)*(1-2*xv) + 2*d2Bv(T)*xv    + np.zeros_like(xc)
d3B_mix_dxv = lambda T, xv, xc: -2*d3Bd(T)*(1 -xv) + 2* d3Bdv(T)*(1-2*xv) + 2*d3Bv(T)*xv    + np.zeros_like(xc)

B_mix_dxc = lambda T, xv, xc: 2*(Bc(T) - Bo(T))*(xc - xc_amb) - 2*xo_amb*Bo(T)                             + np.zeros_like(xv)
dB_mix_dxc = lambda T, xv, xc: 2*(dBc(T) - dBo(T))*(xc - xc_amb) - 2*xo_amb*dBo(T)                         + np.zeros_like(xv)
d2B_mix_dxc = lambda T, xv, xc: 2*(d2Bc(T) - d2Bo(T))*(xc - xc_amb) - 2*xo_amb*d2Bo(T)                     + np.zeros_like(xv)
d3B_mix_dxc = lambda T, xv, xc: 2*(d3Bc(T) - d3Bo(T))*(xc - xc_amb) - 2*xo_amb*d3Bo(T)                     + np.zeros_like(xv)

d2B_mix_dxv2 = lambda T, xv, xc: 2*d2Bd(T) - 4*d2Bdv(T) + 2*d2Bv(T)                         + np.zeros_like(xv) + np.zeros_like(xc)
dB_mix_dxc2 = lambda T, xv, xc: 2*(dBc(T) - dBo(T))                                         + np.zeros_like(xv) + np.zeros_like(xc)
d2B_mix_dxc2 = lambda T, xv, xc: 2*(d2Bc(T) - d2Bo(T))                                        + np.zeros_like(xv) + np.zeros_like(xc)

# Cp0 w.r. to T
dCp0_dry = lambda T: np.polyval(np.polyder(p_dry, 1), T)
d2Cp0_dry = lambda T: np.polyval(np.polyder(p_dry, 2), T)
dCp0_v = lambda T: np.polyval(np.polyder(p_v, 1), T)
d2Cp0_v = lambda T: np.polyval(np.polyder(p_v, 2), T)
dCp0_c = lambda T: np.polyval(np.polyder(p_c, 1), T)
d2Cp0_c = lambda T: np.polyval(np.polyder(p_c, 2), T)
dCp0_o = lambda T: np.polyval(np.polyder(p_o, 1), T)
d2Cp0_o = lambda T: np.polyval(np.polyder(p_o, 2), T)

# Cp mix w.r. to T
dCp_mix_dT = lambda T, xv, xc: (dCp0_dry(T))*(1 -xv) + (dCp0_c(T) - dCp0_o(T))*(xc - xc_amb) + dCp0_v(T)*xv +  - Patm*T*d3B_mix(T, xv, xc) - Patm*d2B_mix(T, xv, xc)
d2Cp_mix_dT2 = lambda T, xv, xc: (d2Cp0_dry(T))*(1 -xv) +  + (d2Cp0_c(T) - d2Cp0_o(T))*(xc- xc_amb)  + d2Cp0_v(T)*xv   - Patm*T*d4B_mix(T, xv, xc) - 2*Patm*d3B_mix(T, xv, xc)

# Cp mix w.r. to xv and xc
dCp_mix_dxv = lambda T, xv, xc: -Cp0_dry(T) + Cp0_v(T) - Patm * T *d2B_mix_dxv(T, xv, xc)
d2Cp_mix_dxv2 = lambda T, xv, xc: - Patm * T *d2B_mix_dxv2(T, xv, xc)
d2Cp_mix_dTdxv = lambda T, xv, xc: -dCp0_dry(T)  + dCp0_v(T) - Patm*T*d3B_mix_dxv(T, xv, xc) - Patm*d2B_mix_dxv(T, xv, xc)
dCp_mix_dxc = lambda T, xv, xc: (Cp0_c(T) - Cp0_o(T)) - Patm * T *d2B_mix_dxc(T, xv, xc)
d2Cp_mix_dxc2 = lambda T, xv, xc: - Patm * T *d2B_mix_dxc2(T, xv, xc)
d2Cp_mix_dTdxc = lambda T, xv, xc: (dCp0_c(T) - dCp0_o(T)) +  - Patm*T*d3B_mix_dxc(T, xv, xc) - Patm*d2B_mix_dxc(T, xv, xc)


dgamma_mix_dT = lambda T, xv, xc: (( Cp_mix(T, xv, xc)*2*Patm*d2B_mix(T,xv, xc) - (R + 2*Patm*dB_mix(T,xv, xc))*dCp_mix_dT(T, xv, xc) )
                                   /  (Cp_mix(T, xv, xc) - R - 2*Patm*dB_mix(T,xv, xc))**2)
dgamma_mix_dxv = lambda T, xv, xc: (( Cp_mix(T, xv, xc)*2*Patm*dB_mix_dxv(T,xv, xc) - (R + 2*Patm*dB_mix(T,xv, xc))*dCp_mix_dxv(T, xv, xc) )
                                    /  (Cp_mix(T, xv, xc) - R - 2*Patm*dB_mix(T,xv, xc))**2)
dgamma_mix_dxc = lambda T, xv, xc: (( Cp_mix(T, xv, xc)*2*Patm*dB_mix_dxc(T,xv, xc) - (R + 2*Patm*dB_mix(T,xv, xc))*dCp_mix_dxc(T, xv, xc) )
                                    /  (Cp_mix(T, xv, xc) - R - 2*Patm*dB_mix(T,xv, xc))**2)


# derivative of c**2/T
dc_mix2_dT = lambda T, xv, xc: (dgamma_mix_dT(T, xv, xc)*c_mix(T, xv, xc)**2/gamma_mix(T, xv, xc)
                                + gamma_mix(T, xv, xc)/M_eff(xv, xc)*(R + 2*Patm*dB_mix(T, xv, xc) )
                                )
dc_mix2_dxv = lambda T, xv, xc: (dgamma_mix_dxv(T, xv, xc)*c_mix(T, xv, xc)**2/gamma_mix(T, xv, xc)
                                 + gamma_mix(T, xv, xc)/M_eff(xv, xc)*2*Patm*B_mix_dxv(T, xv, xc)
                                 - dM_eff_dxv(xv, xc)*c_mix(T, xv, xc)**2/M_eff(xv, xc)
                                 )
dc_mix2_dxc = lambda T, xv, xc: (dgamma_mix_dxc(T, xv, xc)*c_mix(T, xv, xc)**2/gamma_mix(T, xv, xc)
                                 + gamma_mix(T, xv, xc)/M_eff(xv, xc)*2*Patm*B_mix_dxc(T, xv, xc)
                                 - dM_eff_dxc(xv, xc)*c_mix(T, xv, xc)**2/M_eff(xv, xc)
                                 )

# derivative of rho*T
drhoT_dT = lambda T,xv, xc: - (Patm/R)**2*M_eff(xv, xc) * ( dB_mix(T,xv,xc)/T - B_mix(T,xv,xc)/(T**2) )
drhoT_dxv = lambda T,xv, xc: Patm*dM_eff_dxv(xv, xc)/(R)*(1 - B_mix(T,xv,xc)*Patm/(R*T))  - (Patm**2)*M_eff(xv, xc)/(R**2)/T*B_mix_dxv(T,xv,xc)
drhoT_dxc = lambda T,xv, xc: Patm*dM_eff_dxc(xv, xc)/R*(1 - B_mix(T,xv,xc)*Patm/(R*T))  - ((Patm/R)**2)*M_eff(xv, xc)*B_mix_dxc(T,xv,xc)/T



def dl_ref(f_fit, Tref, xv_ref, xc_ref, dx):
    """
    Finite difference at the reference point.

    Parameters
    ----------
    f_fit : callable
        The function to differentiate
    Tref, xv_ref, xc_ref : float
        The reference point.
    dx : float
        The size of the finite difference step

    Returns
    -------
    df_dT, df_dx, df_dxc : float
        The value of the derivative w.r. to each variable

    """
    dT = dx

    df_dT = ( (f_fit(Tref+dT, xv_ref, xc_ref) - f_fit(Tref-dT, xv_ref, xc_ref)) / (2*dT) )
    df_dx = ( (f_fit(Tref, xv_ref + dx, xc_ref) - f_fit(Tref, xv_ref - dx, xc_ref)) / (2*dx))
    df_dxc = ( (f_fit(Tref, xv_ref, xc_ref+dx) - f_fit(Tref, xv_ref, xc_ref -dx)) / (2*dx))
    return df_dT, df_dx, df_dxc



# %%% c: the speed of sound

# The taylor dev is computed on c**2/T

cref = c_mix(Tref, xv_ref, xc_ref)

dc_dT = (dc_mix2_dT(Tref, xv_ref, xc_ref)*(Tref/Tref) - c_mix(Tref, xv_ref, xc_ref)**2*Tref/Tref**2)/cref**2
dc_dxv = dc_mix2_dxv(Tref, xv_ref, xc_ref)*(Tref/Tref)/cref**2
dc_dxc = dc_mix2_dxc(Tref, xv_ref, xc_ref)*(Tref/Tref)/cref**2

d2c_dxc2 = dl_ref(lambda T, xv, xc: dc_mix2_dxc(T, xv, xc)*Tref/T, Tref, xv_ref, xc_ref, 1e-8)[2]
d2c_dxcdT = dl_ref(lambda T, xv, xc: dc_mix2_dT(T, xv, xc) - c_mix(T, xv, xc)**2*Tref/T**2, Tref, xv_ref, xc_ref, 1e-8)[2]
d2c_dxc2 *= (Tref/Tref)/cref**2
d2c_dxcdT *= (Tref/Tref)/cref**2

c_lin = lambda T, xv, xc: cref * np.sqrt(T/Tref)* np.sqrt(1  + dc_dxv*(xv - xv_ref) + dc_dxc*(xc - xc_ref) + d2c_dxcdT*(xc - xc_ref)*(T-Tref)  + .5*d2c_dxc2*(xc - xc_ref)**2 )#+ dc_dT*(T - Tref) )
c_lin_num = lambda T, xv, xc:    343.986 * np.sqrt(T/Tref)* np.sqrt(1  + 0.314*(xv - xv_ref) -(xc - xc_ref)*(0.520 + 0.16*(T/Tref-1))  + 0.25*(xc - xc_ref)**2 )

c_linear = lambda T, xv, xc:    343.986 * np.sqrt(T/Tref)* np.sqrt(1  + 0.314*(xv - xv_ref) -(xc - xc_ref)*(0.520))


# %%%  gamma
gamma_ref = gamma_mix(Tref, xv_ref, xc_ref)

dg_dT = dgamma_mix_dT(Tref, xv_ref, xc_ref)/gamma_ref
dg_dxv = dgamma_mix_dxv(Tref, xv_ref, xc_ref)/gamma_ref
dg_dxc = dgamma_mix_dxc(Tref, xv_ref, xc_ref)/gamma_ref

d2g_dxcdT, d2g_dxvdxc, d2g_dxc2 = dl_ref(dgamma_mix_dxc, Tref, xv_ref, xc_ref, 1e-5)
d2g_dxcdT /= gamma_ref
d2g_dxc2 /= gamma_ref

gamma_lin = lambda T, xv, xc: gamma_ref * (1  + dg_dxv*(xv - xv_ref) + dg_dT*(T - Tref) + dg_dxc*(xc - xc_ref) + d2g_dxcdT*(xc - xc_ref)*(T-Tref) )#+ .5*d2g_dxc2*(xc - xc_ref)**2)
gamma_lin_num = lambda T, xv, xc:  1.40108 * (1  -0.060*(xv - xv_ref)  -0.0087*(T/Tref - 1) -0.104*(xc - xc_ref)   -0.154*(xc - xc_ref)*(T/Tref - 1) )

gamma_linear = lambda T, xv, xc:  1.40108 * (1  -0.060*(xv - xv_ref)  -0.0087*(T/Tref - 1) -0.104*(xc - xc_ref)  )

# %%% Cp mass

# In acoustic the interesting quantity is the MASS heat capacity
Cp_mass_ref = Cp_mass_mix(Tref, xv_ref, xc_ref)

dCp_mass_dT = (dCp_mix_dT(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref))/Cp_mass_ref - 0
d2Cp_mass_dT2 = (d2Cp_mix_dT2(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref) )/Cp_mass_ref - 0

dCp_mass_dxv = (dCp_mix_dxv(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)  -Cp_mix(Tref, xv_ref, xc_ref)*dM_eff_dxv(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2 )/Cp_mass_ref

d2Cp_mass_dxv2 = (d2Cp_mix_dxv2(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref) -2*dCp_mix_dxv(Tref, xv_ref, xc_ref)*dM_eff_dxv(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2  +2*Cp_mix(Tref, xv_ref, xc_ref)*dM_eff_dxv(xv_ref, xc_ref)**2/M_eff(xv_ref, xc_ref)**3 )/Cp_mass_ref

d2Cp_mass_dxvdT = (d2Cp_mix_dTdxv(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref) -  dCp_mix_dT(Tref, xv_ref, xc_ref)*dM_eff_dxv(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2)/Cp_mass_ref

dCp_mass_dxc = (dCp_mix_dxc(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref) -Cp_mix(Tref, xv_ref, xc_ref)*dM_eff_dxc(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2 )/Cp_mass_ref

d2Cp_mass_dxc2 = (d2Cp_mix_dxc2(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)
                  -2*dCp_mix_dxc(Tref, xv_ref, xc_ref)*dM_eff_dxc(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2
                  +2*Cp_mix(Tref, xv_ref, xc_ref)*dM_eff_dxc(xv_ref, xc_ref)**2/M_eff(xv_ref, xc_ref)**3
                  )/Cp_mass_ref

d2Cp_mass_dxcdT = (d2Cp_mix_dTdxc(Tref, xv_ref, xc_ref)/M_eff(xv_ref, xc_ref) -  dCp_mix_dT(Tref, xv_ref, xc_ref)*dM_eff_dxc(xv_ref, xc_ref)/M_eff(xv_ref, xc_ref)**2)/Cp_mass_ref

Cp_mass_lin = lambda T, xv, xc: Cp_mass_ref * (1  + dCp_mass_dxv*(xv - xv_ref) + dCp_mass_dxc*(xc - xc_ref) + dCp_mass_dT*Tref*(T/Tref-1) +  .5*d2Cp_mass_dT2*Tref**2*(T/Tref-1)**2 + d2Cp_mass_dxcdT*Tref*(xc - xc_ref)*(T/Tref-1) + .5*d2Cp_mass_dxv2*(xv - xv_ref)**2 + .5*d2Cp_mass_dxc2*(xc - xc_ref)**2) #+ d2Cp_mass_dxvdT*(xv - xv_ref)*(T-Tref)
Cp_mass_lin_num = lambda T, xv, xc: 1012.25 * (1  +0.5438*(xv-xv_ref)  -0.1594*(xc - xc_ref) + 9.52e-3*(T/Tref - 1) + 4.06e-2*(T/Tref - 1)**2 + 0.3976*(xc - xc_ref)*(T/Tref - 1) + 0.638 *(xv - xv_ref)**2 + 0.075*(xc - xc_ref)**2)
Cp_mass_linear = lambda T, xv, xc: 1012.25 * (1  +0.5438*(xv-xv_ref)  -0.1594*(xc - xc_ref) + 9.52e-3*(T/Tref - 1) )



# %%% rho

# the Taylor dev is computed on rho*T

rhoT_ref = rho_mix(Tref, xv_ref, xc_ref)*Tref

drho_dT = drhoT_dT(Tref, xv_ref, xc_ref)/rhoT_ref
drho_dxv = drhoT_dxv(Tref, xv_ref, xc_ref)/rhoT_ref
drho_dxc = drhoT_dxc(Tref, xv_ref, xc_ref)/rhoT_ref

rho_lin = lambda T, xv, xc: rhoT_ref/T * (1  + drho_dxv*(xv - xv_ref) + drho_dxc*(xc - xc_ref) + drho_dT*(T - Tref) )
rho_lin_num = lambda T, xv, xc: 1.19930*Tref/T * (1  -0.3767*(xv - xv_ref) + 0.4162*(xc - xc_ref) + -0.00291*(T/Tref- 1) )

# =============================================================================
# %% Figures
# =============================================================================


def plots(f_full, f_lin, f_linear, f_chaigne, f_cramer, ylabel_abs, ylabel_dev, filename):

    # CO2 plot
    fig_CO2_abs = plt.figure()
    ax_C02_abs = plt.subplot(1,1,1)
    plt.grid(which='major')
    plt.grid(which='minor', linewidth=.1, linestyle=':')
    ax_C02_abs.minorticks_on()

    fig_CO2_dev = plt.figure()
    ax_C02_dev = plt.subplot(1,1,1)
    plt.grid()

    for k, temp in enumerate(T_loop):

        ax_C02_abs.plot(xc*100, (f_full(temp, 0, xc)), label=f't={deg(temp):.0f}°C',color=colors[k])
        ax_C02_abs.plot(xc*100, (f_lin(temp, 0, xc)), '--', color=colors[k])
        if f_cramer:
            ax_C02_abs.plot(xc*100, (f_cramer(temp, 0, xc)), ':', color=colors[k])

        ax_C02_dev.plot(xc*100, (f_lin(temp, 0, xc)/f_full(temp,0,xc) - 1)* 100, '--', label=f't={deg(temp):.0f}°C', color=colors[k])
        ax_C02_dev.plot(xc*100, (f_linear(temp, 0, xc)/f_full(temp,0,xc) - 1)* 100, 'k:')#, label=f't={deg(temp):.0f}°C', color=colors[k])
        if f_cramer:
            ax_C02_dev.plot(xc*100, (f_cramer(temp, 0, xc)/f_full(temp,0, xc) - 1)* 100, ':', color=colors[k])

    ax_C02_abs.plot(xc*100, f_chaigne(T20)*np.ones_like(xc), 'k', label='C.&K.')
    ax_C02_abs.set_xlabel('Molar Frac. CO2 $x_c$ [%]')
    ax_C02_abs.set_ylabel(ylabel_abs)
    ax_C02_abs.legend()
    ax_C02_abs.set_xlim([0,10])
    fig_CO2_abs.tight_layout()
    fig_CO2_abs.savefig(filename + '_CO2.pdf')

    ax_C02_dev.set_xlabel('Molar Frac CO2 [%]')
    ax_C02_dev.set_ylabel(ylabel_dev)
    ax_C02_dev.legend()
    fig_CO2_dev.tight_layout()

    # humidity plot
    fig_abs = plt.figure()
    ax_abs = plt.subplot(1,1,1)
    plt.grid(which='major')
    plt.grid(which='minor', linewidth=.1, linestyle=':')
    ax_abs.minorticks_on()

    fig_dev = plt.figure()
    ax_dev = plt.subplot(1,1,1)
    plt.grid()

    for k, temp in enumerate(T_loop):
        xv = h(temp, HR)

        ax_abs.plot(HR*100, (f_full(temp, xv, xc_ref)), label=f't={deg(temp):.0f}°C',color=colors[k])
        ax_abs.plot(HR*100, (f_lin(temp, xv, xc_ref)), '--', color=colors[k])
        if f_cramer:
            ax_abs.plot(HR*100, (f_cramer(temp, xv, xc_ref)), ':', color=colors[k])

        ax_dev.plot(HR*100, (f_lin(temp, xv, xc_ref)/f_full(temp,xv, xc_ref) - 1)* 100, '--', label=f't={deg(temp):.0f}°C', color=colors[k])
        ax_dev.plot(HR*100, (f_linear(temp, xv, xc_ref)/f_full(temp,xv, xc_ref) - 1)* 100, 'k:')
        if f_cramer:
            ax_dev.plot(HR*100, (f_cramer(temp, xv, xc_ref)/f_full(temp,xv, xc_ref) - 1)* 100, ':', color=colors[k])

    ax_abs.plot(HR*100, f_chaigne(T20)*np.ones_like(HR), 'k', label='C.&K.')
    ax_abs.set_xlim([0,100])
    # ax_abs.set_ylim([1.385,1.403])
    ax_abs.legend()
    ax_abs.set_xlabel('Humidity rate $h_v$ [%]')
    ax_abs.set_ylabel(ylabel_abs)
    fig_abs.tight_layout()
    fig_abs.savefig(filename + '_humidity.pdf')

    ax_dev.set_xlim([0,100])
    # ax_temp.set_ylim([0,0.4])
    ax_dev.legend()
    ax_dev.set_xlabel('Humidity rate $h_v$ [%]')
    ax_dev.set_ylabel(ylabel_dev)
    fig_dev.tight_layout()


# %%% plots Cp mass

plots(Cp_mass_mix, Cp_mass_lin, Cp_mass_linear, Cp_mass_chaigne, None, '$C_p^m $ [J/(K.kg)]', 'Approx$(C_p^m) / C_{p}^m$ [%]', 'Cp_mass')


# %%% Plots gamma

plots(gamma_mix, gamma_lin, gamma_linear, gamma_chaigne, gamma_cramer, 'Heat Cap. Ratio $\gamma $ ', 'Approx$(\gamma) / \gamma_{full}$ [%]', 'gamma')


# %%% plot c


plots(c_mix, c_lin, c_linear, c_chaigne, c_cramer, 'Speed of sound $c$ [m/s]', 'Approx$(c) / c_{full}$ [%]', 'c')


c0 = c_mix(T0,0,xc_amb)

fig_T_c = plt.figure()
ax_T_c = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_T_c.minorticks_on()
ax_T_c.plot(deg(T), c_mix(T, 0, xc_amb), label='Full expression')
ax_T_c.plot(deg(T), c_lin(T, 0, xc_amb), '--', label='Taylor expansion')
ax_T_c.plot(deg(T), c_cramer(T, 0, 4e-4), ':', label='Cramer fit')
ax_T_c.plot(deg(T), c_chaigne(T) , 'k', label='C.&K.')
# ax_T_c.plot(deg(T), (np.sqrt(c_mix(T, 0, xc_amb)/(c0)) -1), label='Full expression')
# ax_T_c.plot(deg(T), (np.sqrt(c_lin(T, 0, xc_amb)/(c0)) -1), '--', label='Taylor expansion')
# ax_T_c.plot(deg(T), (np.sqrt(c_cramer(T, 0, 4e-4)/(c0)) -1), ':', label='Cramer fit')
# ax_T_c.plot(deg(T), (np.sqrt(c_chaigne(T)/(c0)) -1) , 'k', label='C.&K.')
ax_T_c.set_xlabel('Temperature [°C]')
ax_T_c.set_ylabel(r' $c$ [m/s] ')
ax_T_c.legend()
ax_T_c.set_xlim([0,40])
fig_T_c.tight_layout()
fig_T_c.savefig('c_abs_temperature.pdf')


fig_T_c = plt.figure()
ax_T_c = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_T_c.minorticks_on()
ax_T_c.plot(deg(T), 1e4*(c_mix(T, 0, xc_amb)/(c0*np.sqrt(T/T0)) -1), label='Full expression')
ax_T_c.plot(deg(T), 1e4*(c_lin(T, 0, xc_amb)/(c0*np.sqrt(T/T0)) -1), '--', label='Taylor expansion')
ax_T_c.plot(deg(T), 1e4*(c_cramer(T, 0, 4e-4)/(c0*np.sqrt(T/T0)) -1), ':', label='Cramer fit')
ax_T_c.plot(deg(T), 1e4*(c_chaigne(T)/(c0*np.sqrt(T/T0)) -1) , 'k', label='C.&K.')
ax_T_c.set_xlabel('Temperature [°C]')
ax_T_c.set_ylabel(r' $\left( c/c_0\sqrt{T_0/T} - 1 \right) \times 10^{4}$ ')
ax_T_c.legend()
ax_T_c.set_xlim([0,40])
fig_T_c.tight_layout()
fig_T_c.savefig('c_temperature.pdf')


print('Ref sound celerity at T0, in pure CO2: 259 m/s')
print(f'Current model: {c_mix(T0,0,1):.2f}' )
print(f'Cramer fit: {c_cramer(T0,0,1):.2f}' )


# cents
fig_CO2_c_cent = plt.figure()
ax_C02_c_cent = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_C02_c_cent.minorticks_on()

for k, temp in enumerate(T_loop):


    ax_C02_c_cent.plot(xc*100, 1200*np.log2(c_mix(temp, 0, xc)/c_mix(temp,0,0)), label=f't={deg(temp):.0f}°C',color=colors[k])
    ax_C02_c_cent.plot(xc*100, 1200*np.log2(c_lin(temp, 0, xc)/c_mix(temp,0,0)), '--',color=colors[k])
    ax_C02_c_cent.plot(xc*100, 1200*np.log2(c_cramer(temp, 0, xc )/c_mix(temp,0,0)), ':', color=colors[k])

ax_C02_c_cent.set_xlabel('Molar Frac CO2 [%]')
ax_C02_c_cent.set_ylabel('Dev. to CO2 free: $c / c_{pure}$ [cent]')
ax_C02_c_cent.legend()
ax_C02_c_cent.set_xlim([0,10])
fig_CO2_c_cent.tight_layout()
fig_CO2_c_cent.savefig('c_CO2_cent.pdf')


fig_cent = plt.figure()
ax_cent = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_cent.minorticks_on()


for k, temp in enumerate(T_loop):
    xv = h(temp, HR)

    ax_cent.plot(HR*100, 1200*np.log2(c_mix(temp, xv, xc_ref)/c_mix(temp,0, xc_ref)), label=f't={deg(temp):.0f}°C',color=colors[k])
    ax_cent.plot(HR*100, 1200*np.log2(c_lin(temp, xv, xc_ref)/c_mix(temp,0, xc_ref)), '--',color=colors[k])
    ax_cent.plot(HR*100, 1200*np.log2(c_cramer(temp, xv, xc_ref)/c_mix(temp,0, xc_ref)), '--',color=colors[k])
ax_cent.set_xlim([0,100])
# ax_temp.set_ylim([0,0.4])
ax_cent.legend()
ax_cent.set_xlabel('Humidity rate $h_v$ [%]')
ax_cent.set_ylabel(r'Dev. to dry air $c/c_{dry}$ [cents]')
fig_cent.tight_layout()
fig_cent.savefig('c_humidity_cents.pdf')


# %%% plot rho


plots(rho_mix, rho_lin, rho_lin, rho_chaigne, None, r' $\rho$ [kg.m$^{-3}$]', r'Approx$(\rho) / \rho_{full}$ [%]', 'c')


rho0 = rho_mix(T0,0,xc_amb)
fig_T_rho = plt.figure()
ax_T_rho = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_T_rho.minorticks_on()
ax_T_rho.plot(deg(T), rho_mix(T, 0, xc_amb), label='Full expression')
ax_T_rho.plot(deg(T), rho_lin(T, 0, xc_amb), '--', label='Taylor expansion')
ax_T_rho.plot(deg(T), rho_chaigne(T), 'k', label='C.&K.')
ax_T_rho.set_xlabel('Temperature [°C]')
ax_T_rho.set_ylabel(r' $\rho$ [kg.m$^{-3}$]')
ax_T_rho.legend()
ax_T_rho.set_xlim([0,40])
fig_T_rho.tight_layout()
fig_T_rho.savefig('rho_abs_temperature.pdf')

rho0 = rho_mix(T0,0,xc_amb)
fig_T_rho = plt.figure()
ax_T_rho = plt.subplot(1,1,1)
plt.grid(which='major')
plt.grid(which='minor', linewidth=.1, linestyle=':')
ax_T_rho.minorticks_on()
ax_T_rho.plot(deg(T), 1e4*(rho_mix(T, 0, xc_amb)/(rho0)*(T/T0) -1), label='Full expression')
ax_T_rho.plot(deg(T), 1e4*(rho_lin(T, 0, xc_amb)/(rho0)*(T/T0) -1), '--', label='Taylor expansion')
ax_T_rho.plot(deg(T), 1e4*(rho_chaigne(T)/(rho0)*(T/T0) -1) , 'k', label='C.&K.')
ax_T_rho.set_xlabel('Temperature [°C]')
ax_T_rho.set_ylabel(r' $\left[ (\rho T)/(\rho_0 T_0) - 1 \right] \times 10^{4}$ ')
ax_T_rho.legend()
ax_T_rho.set_xlim([0,40])
fig_T_rho.tight_layout()
fig_T_rho.savefig('rho_temperature.pdf')
