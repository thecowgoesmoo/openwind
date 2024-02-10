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
This file proposes a review of expressions and tabulated value of specific heat (Cp)
and heat capacity ratio (gamma) for different gases (dry air, steam, CO2 and O2)
with ideal gas assumptions or by taken into account the second coefficient of
the virial expansion (B)

[1] O. Cramer 1993. "The variation of the specific heat ratio and the speed of sound in air with temperature, pressure, humidity, and CO 2 concentration". The Journal of the Acoustical Society of America. 93(5), p.2510–2516. DOI:10.1121/1.405827 HAL:http://asa.scitation.org/doi/10.1121/1.405827.
[2] R.W. Hyland 1975. "A Correlation for the Second Interaction Virial Coefficients and Enhancement Factors for Moist Air". Journal of Research of the National Bureau of Standards. Section A, Physics and Chemistry. 79A(4), p.551–560. DOI:10.6028/jres.079A.017 HAL:https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6565410/.
[3] J. Hilsenrath 1955. Tables of Thermal Properties of Gases: Comprising Tables of Thermodynamic and Transport Properties of Air, Argon, Carbon Dioxide, Carbon Monoxide, Hydrogen, Nitrogen, Oxygen, and Steam. U.S. Department of Commerce, National Bureau of Standards.
[4] G.W.C. Kaye and T.H. Laby 1995. Tables of physical and chemical constants. Longman.
[5] W.M. Rohsenow, J.P. Hartnett and Y.I. Cho eds. 1998. Handbook of heat transfer. McGraw-Hill.
[6] H.L. Sengers, M. Klein and J.S. Gallagher 1971. "Pressure-Volume-Temperature relationships of gases. Virial coefficients..". national Bureau of Standards Washinghton D.C., Heat Division.
[7] A.J. Zuckerwar 2002. Handbook of the Speed of Sound in Real Gases. Academic Press.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.optimize import curve_fit

colors = list(matplotlib.colors.TABLEAU_COLORS)

plt.close('all')
font = {'family': 'serif', 'size': 14}
matplotlib.rc('font', **font)
matplotlib.rcParams['mathtext.fontset'] = 'cm'

# %% Tabulated values

T0 = 273.15 #
T20 = T0 + 20
Patm = 101325
R = 8.31446261815324 # cste gaz parfait https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits

deg = lambda T: T - T0 # Kelvin °C conversion

T = np.linspace(273.15,313.15, 101) # 373.15,100)#
HR = np.linspace(0, 1,501)
HR_loop = np.linspace(0, 1,6)
# T_loop = np.linspace(273.15,313.15,5)
T_loop = np.linspace(273.15,303.15,4)

# %% Cramer

# Fit for humid  air with CO2 from Cramer used here to compare
def gamma_cramer(T, xv, xc):
    t = T - T0
    a0 = 1.400822
    a1 = -1.75e-5
    a2 = -1.73e-7
    a3 = -.0873629
    a4 = -1.665e-4
    a5 = -3.26e-6
    a6 = 2.047e-8
    a7 = -1.26e-10
    a8 = 5.939e-14
    a9 = -.1199717
    a10 = -8.693e-4
    a11 = 1.979e-6
    a12 = -.01104
    a13 = -3.478e-16
    a14 = .0450616
    a15 = 1.82e-6

    # xc = 400e-6 # 400 ppm
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


# %% Dry air

T = np.linspace(273.15,313.15, 101) # 373.15,100)#
Mdry = 0.0289647 # HANDBOOK chap.16.4.2, p.14 // masse molaire air sec https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits

# Second Virial coefficient
ad, bd, cd = (152.2e-6, 111.3e-6, 108.1) # Zuckerwar, fit of a lot of refereence, (see p.16)
Bd = lambda T: ad- bd*np.exp(cd/T)
dBd = lambda T: bd*cd*np.exp(cd/T)/T**2
d2Bd = lambda T: -bd*cd*(cd +2*T)*np.exp(cd/T)/(T**4)
Bd_hyland = lambda T: (-13.5110 + 0.24311*(T-T0) -0.10846e-2*(T-T0)**2
                       + 0.42504e-5*(T-T0)**3 - 0.81851e-8*(T-T0)**4)*1e-6 # Hyland. eq.(4)


Cp0_dry_Toul = lambda T: (.244388 -4.20419e-5*T +9.61128e-8*T**2 -1.16383e-11*T**3)*4.184e3*Mdry # touloukian p.293
Cp1_dry_Toul = lambda T: (.249679 -7.55179e-5*T +1.69194e-7*T**2 -6.46128e-11*T**3)*4.184e3*Mdry # touloukian p.293
Cp_dry_Rohsen = lambda T: ( 1.03409 -.284887e-3*T +.7816818e-6*T**2 -.4970786e-9*T**3 +.1077024e-12*T**4 )*Mdry*1e3 # Rohsenow Tab2.10 (p.58) (Ref. to 1970)
Cp0_dry_Zuck = lambda T: (3.5623 -.0006128*T + 1.40e-6*T**2 -1.696e-10*T**3)*R #Handbook Zuck Tab16.6, Chap.16, p.14  (Ref. to 1850)
Cp_chaigne = lambda T: np.ones_like(T)*0.24 * 4.184e3 * Mdry

# Real gas by accouting for second virial coef
Cp_d = lambda T: Cp0_dry_Toul(T) - Patm * T *d2Bd(T)


gamma_chaigne = lambda T: np.ones_like(T)* 1.402
gamma_dry = lambda T: 1 + 1 / ( Cp_d(T)/(R + 2*Patm*dBd(T)) -1) # from cramer

plt.figure()
plt.plot(deg(T), Cp0_dry_Toul(T), label='Touloukian: $C_p^0$')
plt.plot(deg(T), Cp0_dry_Zuck(T), '--', label='Zuckerwar: $C_p^0$')
plt.plot(deg(T), Cp_chaigne(T), '--',label='Chaigne & Kergo.')
plt.plot(deg(T), Cp1_dry_Toul(T) ,label='Touloukian: $C_p^{1}$')
plt.plot(deg(T), Cp_dry_Rohsen(T), label='Rohsenow: $C_p^{1}$')
plt.plot(deg(T), Cp_d(T), '--',label='Toul. $C_p^0$ + virial: $C_p^{1}$')
plt.legend(prop={'size': 11})
plt.ylim([29.06,29.25])
plt.xlabel('Temperature [°C]')
plt.ylabel('Molar Heat Capacity [J/(mol.K)]')
plt.title('Dry Air')
plt.grid()
plt.tight_layout()
plt.savefig('App_Cp_dry.pdf')

plt.figure()
plt.plot(deg(T), Bd_hyland(T)*1e6, label='Hyland')
plt.plot(deg(T), Bd(T)*1e6, '--', label='Zuckerwar')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Second Virial $B$ [cm$^3$/mol]')
plt.title('Dry Air')
plt.grid()
plt.tight_layout()
plt.savefig('App_Virial_dry.pdf')

plt.figure()
plt.plot(deg(T), gamma_dry(T), label='From Virial')
plt.plot(deg(T), gamma_cramer(T, 0, 4e-4), label="Cramer's fit")
plt.plot(deg(T), gamma_chaigne(T), label='Chaigne & Kergomard')
plt.plot(0, 1.4027, 'k*',label='Hilsenrath')
plt.plot([0, 20], [1.403, 1.4], 'ko',label='Wikipedia')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Heat Capacity Ratio $\gamma$')
plt.title('Dry Air')
plt.grid()
plt.tight_layout()
plt.savefig('App_gamma_dry.pdf')

# %%  Water vapor / steam
T = np.linspace(273.15, 473.15,100)#313.15, 101)

Mv =  18.01534e-3 #https://fr.wikipedia.org/wiki/Eau

# Second Virial Coefficient
av, bv, cv = (31.5e-6, 13.6e-6, 1375.3) # not explain how it is obtained but valid for T>423K
Bv_Zuck = lambda T: av - bv*np.exp(cv/T) # virial from Zuck 16.4.4 p.16
dBv_Zuck = lambda T: bv*cv*np.exp(cv/T)/T**2
d2Bv_Zuck = lambda T: -bv*cv*(cv +2*T)*np.exp(cv/T)/T**4

av_KL, bv_KL, cv_KL = (33.0e-6 , 15.2e-6 , 1300.7) # from Kaye and Laye
Bv_KL = lambda T: av_KL - bv_KL*np.exp(cv_KL/T)
dBv_KL = lambda T: bv_KL*cv_KL*np.exp(cv_KL/T)/T**2
d2Bv_KL = lambda T: -bv_KL*cv_KL*(cv_KL +2*T)*np.exp(cv_KL/T)/T**4

Bv_hyland =lambda T: 33.97e-6 - 55306e-6/T*10**(72000/T**2) # Cramer and Hyland

# Specific heat
Cp0_v_Toul = lambda T: (.452219 - 1.29224e-4*T + 4.17008e-7*T**2 - 2.00401e-10*T**3)*4.184e3*Mv # touloukian p.105
Cp1_v_Toul = lambda T: (1.94480 - 8.92530e-3*T + 1.78830e-5*T**2 - 1.18442e-8*T**3)*4.184e3*Mv # touloukian p.105
Cp0_v = lambda T: (4.0996 -.001171*T + 3.78e-6*T**2 -1.817e-9*T**3)*R #Handbook Zuck Tab16.6, Chap.16.4.3, p.14
# Rohsenow p.2.26, Table2.15
t_roh = np.array(range(0,210,10))
Cp_roh = np.array([1.864, 1.868, 1.874, 1.883, 1.894, 1.907, 1.924, 1.944, 1.969, 1.999, 2.034, 2.075, 2.124, 2.180, 2.245, 2.320, 2.406, 2.504, 2.615, 2.741, 2.883])*Mv*1e3

# Cp fror real gas
Cp_v = lambda T: Cp0_v(T) - Patm * T *d2Bv_KL(T)

# Heat capacity ratio computed from specific heat assuming real (with virial) or ideal gas
gamma_v = lambda T: 1 + 1/(Cp_v(T)/(R + 2*Patm*dBv_KL(T)) -1)
gamma_v_ideal = lambda T: 1 + 1/(Cp0_v(T)/(R) -1)

plt.figure()
plt.plot(deg(T), Bv_hyland(T)*1e6, label='Hyland')
plt.plot(deg(T), Bv_Zuck(T)*1e6, '--', label='Zuckerwar (for T>150°C)')
plt.plot(deg(T), Bv_KL(T)*1e6, ':', label='Kaye & Laby')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Second Virial $B$ [cm$^3$/mol]')
plt.title('Water Vapor')
plt.grid()
plt.tight_layout()
plt.savefig('App_Virial_v.pdf')

plt.figure()
plt.plot(deg(T), Cp0_v_Toul(T), label='Touloukian: $C_p^0$')
plt.plot(deg(T), Cp0_v(T), '--', label='Zuckerwar: $C_p^0$')
plt.plot(t_roh, Cp_roh, 'o', label='Roshenow: ?')
plt.plot(deg(T), Cp1_v_Toul(T) ,label='Touloukian (for T>100°C): $C_p^{1}$')
plt.plot(deg(T), Cp_v(T), '--',label='Toul. + virial: $C_p^{1}$')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Molar Heat Capacity [J/(mol.K)]')
plt.title('Water Vapor')
plt.grid()
plt.tight_layout()
plt.savefig('App_Cp_v.pdf')

plt.figure()
plt.plot(deg(T), gamma_v(T), label='From Virial')
plt.plot(deg(T), gamma_v_ideal(T), label='Ideal gas ($C_p^{(0))}$ only)')
# plt.plot(deg(T), gamma_cramer(T, 1, 4e-4), label="Cramer's fit")
plt.plot([20,100,200], [1.330,1.324,1.310], 'ko',label='Wikipedia')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Heat Capacity Ratio $\gamma$')
plt.title('Water Vapor')
plt.grid()
plt.tight_layout()
plt.savefig('App_gamma_v.pdf')
# %% CO2

T = np.linspace(273.15, 373.15,100)#313.15, 101)
Mc = 44.0095e-3

# Second Virial coefficient
# Kaye Laby Chap.3.5
ac_KL, bc_KL, cc_KL = (137.6e-6, 87.7e-6, 325.7)
Bc_KL = lambda T: ac_KL - bc_KL*np.exp(cc_KL/T)
dBc_KL = lambda T: bc_KL*cc_KL*np.exp(cc_KL/T)/T**2
d2Bc_KL = lambda T: -bc_KL*cc_KL*(cc_KL +2*T)*np.exp(cc_KL/T)/(T**4)

# Sengers P44
t_ref = [T0] + list(range(275,305,5)) + list(range(310,380,10)) + [T0 +100]
B_ref = [b*1e-6 for b in [-149.7, -147.4, -141.7, -136.2, -131.1, -126.2, -121.5, -112.8, -104.8, -97.5, -90.8, -84.7, -79, -73.8, -72.2]]

# Hilsenrath p.151
t_ref_Hilse = list(range(270,380,10))
Z_ref_Hilse = [.99291, .99372, .99441, .99501, .99553, .99598, .99638, .99673, .99705, .99732, .99757]
B_ref_Hilse = list((np.array(Z_ref_Hilse) - 1)*R*np.array(t_ref_Hilse)/Patm)

# Global fit
Bc_fit = lambda T, ac, bc, cc: ac - bc*np.exp(cc/T)
coef, truc = curve_fit(Bc_fit, t_ref+t_ref_Hilse, B_ref+B_ref_Hilse, p0=[ac_KL, bc_KL, cc_KL])
ac, bc, cc = coef
Bc = lambda T: ac - bc*np.exp(cc/T)
dBc = lambda T: bc*cc*np.exp(cc/T)/T**2
d2Bc = lambda T: -bc*cc*(cc +2*T)*np.exp(cc/T)/(T**4)


# Specific heat from Touloukian p.145
Cp0_c = lambda T: (.105914 + 4.03552e-4*T - 3.03235e-7*T**2 + 8.2943e-11*T**3)*4.184e3*Mc
Cp1_c = lambda T: (.136812 + 2.15312e-4*T + 7.25177e-8*T**2 - 1.57146e-10*T**3)*4.184e3*Mc
Cp_c_virial = lambda T: Cp0_c(T) - Patm* T *d2Bc(T)

# Heat capacity ratio computed from specific heat assuming real (with virial) or ideal gas
gamma_c = lambda T: 1 + 1/(Cp_c_virial(T)/(R + 2*Patm*dBc(T)) -1)
gamma_c_ideal = lambda T: 1 + 1/(Cp0_c(T)/(R) -1)


plt.figure()
plt.plot(deg(T), Cp0_c(T), label='Touloukian: $C_p^0$')
plt.plot(deg(T), Cp1_c(T) ,label='Touloukian (for T>100°C): $C_p^{1}$')
plt.plot(deg(T), Cp_c_virial(T), '--',label='Toul. + virial: $C_p^{1}$')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Molar Heat Capacity [J/(mol.K)]')
plt.title('Carbon dioxide')
plt.grid()
plt.tight_layout()
plt.savefig('App_Cp_CO2.pdf')

plt.figure()
plt.plot(deg(T), Bc(T)*1e6, '--', label='Manual fit')
plt.plot(deg(T), Bc_KL(T)*1e6, ':', label='Kaye & Laby')
plt.plot(deg(np.array(t_ref)), np.array(B_ref)*1e6, '*', label='Sengers')
plt.plot(deg(np.array(t_ref_Hilse)), np.array(B_ref_Hilse)*1e6, 'o', label='Hilsenrath')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Second Virial $B$ [cm$^3$/mol]')
plt.title('Carbon dioxide')
plt.grid()
plt.tight_layout()
plt.savefig('App_Virial_CO2.pdf')

plt.figure()
plt.plot(deg(T), gamma_c(T), label='From Virial')
plt.plot(deg(T), gamma_c_ideal(T), label='Ideal gas ($C_p^{(0))}$ only)')
plt.plot(deg(T), gamma_cramer(T, 0, 1), ':', label="Cramer's fit (extrapolated)")
plt.plot([0,20,100], [1.310,1.300,1.281], 'ko',label='Wikipedia')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Heat Capacity Ratio $\gamma$')
plt.title('Carbon dioxide')
plt.grid()
plt.tight_layout()
plt.savefig('App_gamma_CO2.pdf')

# %% O2


# dioxygen
Mo = 31.9988e-3 # Cramer
Cp0_o = lambda T: (.222081 - 7.69230e-5*T + 2.78765e-7*T**2 - 1.70107e-10*T**3)*4.184e3*Mo # Touloukian p.50
Cp1_o = lambda T: (.227218 -1.06392e-4*T + 3.33354e-7*T**2 - 2.02242e-10*T**3)*4.184e3*Mo

ao, bo, co = (152.8e-6, 117.0e-6, 108.8) # Kaye Laby
Bo = lambda T: (ao - bo*np.exp(co/T))
dBo = lambda T: (bo*co*np.exp(co/T)/T**2)
d2Bo = lambda T: (-bo*co*(co +2*T)*np.exp(co/T)/T**4)
Cp_o_virial = lambda T: Cp0_o(T) - Patm * T *d2Bo(T)
gamma_o = lambda T: 1 + 1/(Cp_o_virial(T)/(R + 2*Patm*dBo(T)) -1)

# Nitrogen
aN, bN, cN = (185.4e-6, 141.8e-6, 88.7) # Kay & Laby
BN = lambda T: (aN - bN*np.exp(cN/T))

plt.figure()
plt.plot(deg(T), Cp0_o(T), label='Touloukian: $C_p^0$')
plt.plot(deg(T), Cp1_o(T) ,label='Touloukian (for T>100°C): $C_p^{1}$')
plt.plot(deg(T), Cp_o_virial(T), '--',label='Toul. $C_p^0$ + virial: $C_p^{1}$')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Molar Heat Capacity [J/(mol.K)]')
plt.title('Dioxygen')
plt.grid()
plt.tight_layout()
plt.savefig('App_Cp_O2.pdf')





# %% Virial Interaction Coefficient of Air-Water Mixture
adv, bdv, cdv = (224.0e-6, 184.6e-6, 94.6) # from Zuck 16.4.4, p.16, from Hyland values
Bdv = lambda T: adv - bdv*np.exp(cdv/T)
Bdv_hyland = lambda T: -1*(36.98928 - 0.331705*(T-T0) + 0.139035e-2*(T-T0)**2
                           - 0.574154e-5*(T-T0)**3 + 0.326513e-7*(T-T0)**4
                           - 0.142805e-9*(T-T0)**5)*1e-6 # Hyland 1975, eq.(9) / used in Cramer

plt.figure()
plt.plot(deg(T), Bc_KL(T)*1e6, label='Hyland fit')
plt.plot(deg(T), Bc(T)*1e6, '--', label='Zuckerwar fit')

plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Second Virial $B$ [cm$^3$/mol]')
plt.title('Water vapor - dry air interaction')
plt.grid()
plt.tight_layout()
plt.savefig('App_Virial_mix.pdf')

# %% Virial Comparison
plt.figure()
plt.plot(deg(T), Bo(T)*1e6, '-', label='Dioxygen')
plt.plot(deg(T), Bc(T)*1e6, '-', label='Carbon dioxyde')
plt.plot(deg(T), BN(T)*1e6, '--', label='Nitrogen')
plt.plot(deg(T), Bv_KL(T)*1e6, '-', label='water')
plt.plot(deg(T), Bdv(T)*1e6, ':', label='dry air-water interaction')
plt.legend()
# plt.xlim([-1,41])
plt.xlabel('Temperature [°C]')
plt.ylabel('Second Virial $B$ [cm$^3$/mol]')
# plt.title('Dioxygen')
plt.grid()
plt.tight_layout()
plt.savefig('App_Virial_comp.pdf')



# %%  Humidity rate and saturation pressure

# Saturation pressure scaled by atmospheric pressure
# Psat_Wexler = lambda T : 1/Patm*np.exp(1.2811805e-5*T**2 - 1.9509874e-2*T + 34.04926034 - 6353.6311/T) # Handbook Zuck Eq.16.10, Chap.16.3.3
Psat_Wexler = lambda T : 1/Patm*10**(5.56411e-06*T**2 - 8.4730e-3*T + 14.7874 - 2759.35/T) #


Psat_ISO = lambda T: 10**(-6.8346*(T/(T0 + .01))**(-1.261) + 4.6151) # Handbook Zuck ISO 9613-1 Eq.16.11, Chap.16.3.3
Psat_ANSI = lambda T: 10**(10.79586*(1 - (T0+.01)/T)
                           -5.02808 * np.log10(T/(T0+.01))
                           +1.50474e-4*(1 - 10**(-8.29692*(T/(T0+.01) - 1)) )
                           +0.42873e-3 * (10**(-4.76955*(1 - T/(T0+.01))) -1)
                           -2.2195983
                           )  # Handbook Zuck ANSI S1.26-1995 Eq.16.12.a, Chap.16.3.3

# Enhancement factor or correction for real gas
# ferr = lambda T: np.exp(-.00201 + 0.8446/T + 231.06/T**2)# correction for real gas; Handbook Zuck Eq.16.13.a, Chap.16.3.3
ferr = lambda T: 10**(-8.73e-4 + 0.3668/T + 100.35/T**2)# correction for real gas; Handbook Zuck Eq.16.13.a, Chap.16.3.3
ferr_cramer = lambda T: 1.00062 + 3.14e-8*Patm + 5.6e-7*(T-T0)**2 # Cramer 1993, Eq.(A2)


# Combination of Psat and ferr
# fact_humidity = lambda T: 10**(4.6142 -6.8346*(T/(T0 + .01))**(-1.261) + 0.3668/T + 100.35/T**2)
fact_humidity = lambda T: 10**(4.6142 -8073.0*T**(-1.261) + 0.3668/T + 100.35/T**2)
def molar_fraction(phi, T):
    return phi*fact_humidity(T)

plt.figure()
plt.plot(deg(T), Psat_Wexler(T), label="Wexler's equation")
plt.plot(deg(T), Psat_ANSI(T), label='$P_{sat}$ ANSI')
plt.plot(deg(T), Psat_ISO(T), '--', label='$P_{sat}$ ISO')
plt.xlabel('Temperature [°C]')
plt.ylabel(r'$P_{sat}/P_{atm}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('App_Psat.pdf')

plt.figure()
plt.plot(deg(T), (ferr(T) - 0)*1, label='Zuckerwar')
plt.plot(deg(T), (ferr_cramer(T) - 0)*1, label='Cramer')
plt.xlabel('Temperature [°C]')
plt.ylabel('Enhancement factor $f_{er}$')
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('App_ferr.pdf')
