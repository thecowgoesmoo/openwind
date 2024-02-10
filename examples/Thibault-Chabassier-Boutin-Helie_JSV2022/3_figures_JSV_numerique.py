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
This script is part of the numerical examples accompanying the article:
Thibault, A., Chabassier, J., Boutin, H., & Hélie, T. (2023).
Transmission line coefficients for viscothermal acoustics in conical tubes.
Journal of Sound and Vibration, 543, 117355.

This script runs the 1D impedance computations described in the
numerical section of the article:

    Transmission line coefficients for viscothermal acoustics in conical tubes,
    A. Thibault, J. Chabassier, H. Boutin, T. Hélie,
    submitted to Journal of Sound and Vibration (2022)

The following models are compared:
    - ZK. (Zwikker--Kosten) the loss coefficients are calculated from equation
          (55) using convention (ALT); this corresponds to the
          viscothermal model many authors use.
	- SH. (Spherical Harmonics) the loss coefficients are those derived in
          section 6, given by equations (50-51);
	- ZK-HR. (Zwikker--Kosten with Hydraulic Radius) the loss coefficients are
          calculated using the  equation (59) based on the hydraulic radius
          proposed in section 6.4.

As a reference, we use the 3D Sequential Linearized Navier--Stokes (SLNS) model
of viscothermal acoustics described in

    W. Kampinga, Y. H. Wijnant, A. de Boer, An efficient finite element model
    for viscothermal acoustics, Acta Acustica united with Acustica 97 (4)
    (2011) 618-631

The 3D results were computed with the Montjoie finite element software:

    https://www.math.u-bordeaux.fr/~durufle/montjoie/

and are stored in text files in the "IMPEDANCES_..." folders.


The script is organized as follows:
    - import the required modules
    - initialize the geometrical and physical parameters
    - import the impedance calculated with Montjoie
    - calculate the 1D models with Openwind
    - plot the geometry
    - plot magnitude and phase of the impedance computed with SLNS
    - plot magnitude and phase of Z/Z_SLNS
    - print the value of the error

This script generates Figures 5, 6a and 6b of the article.

Author: Alexis THIBAULT
"""

#%% Import the required modules

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse.linalg import MatrixRankWarning

from openwind import ImpedanceComputation, InstrumentGeometry
from openwind.continuous import Physics
from openwind.impedance_tools import plot_impedance

figsize_s = (4, 2.6) # For small article figures
figsize = (7, 4) # For larger article figures
# figsize = (10, 6) # For research

# 1D calculations with model SH are a bit expensive due to the need to
# compute Legendre functions of complex order, which requires summing
# hypergeometric series.
# Disable if you want to compare only ZK and ZK-HR (much faster).
use_SH = True

if use_SH:
    import mpmath # Module mpmath is required to run computations with model SH

#%% Initialize the geometrical and physical parameters

# Shape of the middle section of the Vox Humana pipe described in
# Rucz, P., Angster, J., Augusztinovicz, F., Miklós, A., & Preukschat, T. (2013).
# Modeling resonators of reed organ pipes.
shape = [[0, 4.95e-3], [61.60e-3, 24.60e-3]]

r0 = 4.95e-3 # radius of the entry circle
R0 = 24.60e-3 # radius of the exit circle
L_OW = 61.60e-3 # length along the axis
Theta = np.arctan((R0-r0)/L_OW) # half opening angle of the cone
L = L_OW / np.cos(Theta) # distance between the inner and the outer sphere

# loss_factor_alphainv is the ratio between R_HR and R_ALT
# See Equation (59)
loss_factor_alphainv = 2.0 / (1 + np.cos(Theta))

# Physical parameters
temperature = 20
physics = Physics(temperature)
c0 = physics.c(0)
# Cutoff frequency of the first nonplanar mode is given by
# k*R = 1.84  [Chaigne & Kergomard, (7.147)]
freq_cutoff = 1.84 * c0/R0 / (2*np.pi)
print("Cutoff frequency is",freq_cutoff)


#%% Import the impedance calculated with Montjoie

data = np.loadtxt("IMPEDANCES_Gertrude/"+"GERTRUDE-open-KAMP-CORR-2000-4500.txt")

imped_SLNS = data[:,1] + 1j*data[:,2]
fs_SLNS = data[:,0]

#%% Calculate the 1D models with Openwind

fs = fs_SLNS

# Discretization parameters
l_ele = 0.01
order = 10

print("Computing ZK")
result_ZK = ImpedanceComputation(fs, shape, temperature=temperature,
                                 spherical_waves='spherical_area_corr',
                                 losses='bessel',
                                 order=order, l_ele=l_ele,
                                 radiation_category='perfectly_open',
                                 nondim=False,
                                 )

Zc = result_ZK.Zc

print()
print("Computing HR")
result_HR = ImpedanceComputation(fs, shape, temperature=temperature,
                                 spherical_waves='spherical_area_corr',
                                 losses='bessel',
                                 order=order, l_ele=l_ele,
                                 radiation_category='perfectly_open',
                                 loss_factor_alphainv=loss_factor_alphainv,
                                 nondim=False,
                                 )


if use_SH:
    with warnings.catch_warnings():
        # Computation will fail at some frequencies and produce NaNs;
        # this is fine as they are dealt with later.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=MatrixRankWarning)

        print("\nComputing SH (takes a while)...", end="")
        result_SH = ImpedanceComputation(fs, shape, temperature=temperature,
                                          spherical_waves='spherical_area_corr',
                                          losses='sh',
                                          order=order, l_ele=l_ele,
                                          radiation_category='perfectly_open')
        print("OK!\n")




#%% Plot the geometry

fig = plt.figure("Figure 5", figsize=figsize_s)
InstrumentGeometry(shape).plot_InstrumentGeometry(fig,color="k")

ell0 = r0/np.sin(Theta)
centerx = -ell0*np.cos(Theta)
thetas = np.linspace(-Theta, Theta, 200)
xarc0 = centerx+np.cos(thetas)*ell0; yarc0 = np.sin(thetas)*ell0
ell1 = ell0+L
xarc1 = centerx+np.cos(thetas)*ell1; yarc1 = np.sin(thetas)*ell1
plt.plot(xarc0*1000, yarc0*1000, color="C0", linestyle=":")
plt.plot(xarc1*1000, yarc1*1000, color="C0", linestyle=":")

plt.title("Small cone")
plt.xlabel("z (mm)")
plt.ylabel("Radius (mm)")
plt.grid('both')
fig.tight_layout()
plt.savefig("Fig5.png", dpi=300)

#%% Plot magnitude and phase of the impedance computed with SLNS

fig, (ax1,ax2) = plt.subplots(2, 1, figsize=figsize, num="Figure 6 (a)")

for freqs, imp, mark, label, cface, cedge in [
        # (fs, result_ZK.impedance, "+", "ZK", "None", "C0"),
        # (fs, result_SH.impedance, "x", "SH", "None", "r"),
        # (fs, result_HR.impedance, "+", "ZK-HR", "None", "C1"),
        (fs_SLNS, imped_SLNS, ".", "SLNS (ref)", "k", "None"),
        ]:
    plot_impedance(
        freqs,
        imp,
        Zc0=Zc,
        figure=fig,
        marker=mark,
        linestyle="None",
        markerfacecolor=cface,
        markeredgecolor=cedge,
        label=label,
    )

ax1.grid("both")
ax2.grid("both")
ax1.legend(loc="lower right")

ax1.axvline(freq_cutoff, linestyle="--", color="gray")
ax2.axvline(freq_cutoff, linestyle="--", color="gray")


fig.tight_layout()
plt.savefig("Fig6a.png", dpi=300)

#%%


fig, (ax1,ax2) = plt.subplots(2,1,figsize=figsize, sharex=True, num="Figure 6 (b)")

ax1.set_yscale("symlog", linthresh=1e-6)
ax2.set_yscale("symlog", linthresh=1e-5)

cp = "#000000"
cm = "#000000"


r1 = result_ZK.impedance/imped_SLNS
r1m = (np.abs(r1) - 1)
r1p = np.angle(r1)
ax1.scatter(fs, (r1m),
            marker="x",
            c=[cp if v > 0 else cm for v in r1m],
            linewidths=0.8,
            label="ZK"
        )
ax2.scatter(fs, (r1p),
            marker="x",
            c=[cp if v > 0 else cm for v in r1p],
            linewidths=0.8,
            label="ZK"
        )

if use_SH:
    r2 = result_SH.impedance/imped_SLNS
    r2m = np.abs(r2) - 1
    r2p = np.angle(r2)
    ax1.scatter(fs, (r2m),
                marker="o",
                facecolor="None",
                edgecolors=[cp if v > 0 else cm for v in r2m],
                linewidths=0.5,
                label="SH"
            )
    ax2.scatter(fs, (r2p),
                marker="o",
                facecolor="None",
                edgecolors=[cp if v > 0 else cm for v in r2p],
                linewidths=0.5,
                label="SH"
            )

r3 = result_HR.impedance/imped_SLNS
r3m = np.abs(r3) - 1
r3p = np.angle(r3)
ax1.scatter(fs, (r3m),
            marker=".",
            # facecolor="None",
            # edgecolor
            c=[cp if v > 0 else cm for v in r3m],
            linewidths=0.5,
            label="ZK-HR"
        )
ax2.scatter(fs, (r3p),
            marker=".",
            # facecolor="None",
            # edgecolor
            c=[cp if v > 0 else cm for v in r3p],
            linewidths=0.5,
            label="ZK-HR"
        )

ax1.axhline(0, c='gray')
ax2.axhline(0, c='gray')
ax2.set_xlabel("Frequency (Hz)")
ax1.set_ylabel("$|Z / Z_{SLNS}| - 1$")
ax2.set_ylabel("angle$(Z / Z_{SLNS})$")
# ax1.set_title("$Z/Z_{SLNS}$ (cyan=positive, red=negative)")
ax1.grid(True, which='both', axis='both', color="#cccccc")
ax2.grid(True, 'both', color="#cccccc")
ax1.legend()
ax1.axvline(freq_cutoff, linestyle="--", color="gray")
fig.tight_layout()
plt.savefig("Fig6b-symlog.png", dpi=300)



#%% Print the value of the error

err1 = (result_ZK.impedance/imped_SLNS - 1)
err3 = (result_HR.impedance/imped_SLNS - 1)
if use_SH:
    err2 = (result_SH.impedance/imped_SLNS - 1)


# From what frequency did the SH calculation fail?
if use_SH:
    print("SH calculation failed for frequencies",fs[np.isnan(result_SH.impedance)][0], "and above")


print("Error between ZK and SLNS",f"max={np.max(abs(err1))*100:.5g}%",
      f"mean={np.mean(abs(err1))*100:.5g}%")
if use_SH:
    err2b = err2[~np.isnan(err2)] # Remove NaNs
    print("Error between SH and SLNS",f"max={np.max(abs(err2b))*100:.5g}%",
          f"mean={np.mean(abs(err2b))*100:.5g}%")

print("Error between HR and SLNS",f"max={np.max(abs(err3))*100:.5g}%",
      f"mean={np.mean(abs(err3))*100:.5g}%")


err_ZK_HR = np.abs(result_ZK.impedance / result_HR.impedance - 1)
print("Error between ZK and HR is",f"max={np.max(abs(err_ZK_HR))*100:.5g}%",
      f"mean={np.mean(abs(err_ZK_HR))*100:.5g}%")

if use_SH:
    with warnings.catch_warnings():
        # Do not produce warnings when encountering NaNs
        warnings.simplefilter("ignore", category=RuntimeWarning)

        err_HR_SH = np.abs(result_HR.impedance / result_SH.impedance - 1)
        err_HR_SH = err_HR_SH[~np.isnan(err_HR_SH)] # Remove NaNs
        print("Error between HR and SH",f"max={np.max(abs(err_HR_SH))*100:.5g}%",
              f"mean={np.mean(abs(err_HR_SH))*100:.5g}%")



