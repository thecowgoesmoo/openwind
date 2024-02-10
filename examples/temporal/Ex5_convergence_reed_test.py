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
Convergence curves of a simple reed instrument.
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import simpleaudio as sa
except:
    print('To play the sound please install simpleaudio')

import scipy.signal as signal

from openwind import Player, simulate
from openwind.temporal.utils import export_mono
from openwind.technical.temporal_curves import constant_with_initial_ramp


# cylinder
instrument = [[0.0, 1e-2],
              [0.3, 1e-2]]
# with no hole
holes = []

# a reed model for the oscillator
player = Player('CLARINET')

# the series of time refinements
fact = [1,2,4,8,16,32,64]
#fact = [4]

# the series of mouth pressure targets
mouth_pressures = [1900,1950,2000,2050,2100, 2200,2400,2600,2800,3000]
#mouth_pressures = [2200]

# simulation time in seconds
duration = 1
outputs = {}

# first simulation is run with 17221 time steps in each second
n_time_step_base = 17221 * duration

# loop on the target mouth_pressures
for jj in mouth_pressures:
    # loop on the refinements
    for ii in fact:

        n_time_step = ii * n_time_step_base #51661  # power of 2 * n_time_step_base
        # launch simulatio
        rec = simulate(duration,
                       instrument,
                       holes,
                       player = player,
                       losses='diffrepr',
                       temperature=20,
                       l_ele=0.25, order=4,  # Discretization parameters
                       n_steps=n_time_step
                       )

        # signal = rec.values['bell_radiation_pressure']
        outputs['rec_{}_pm_{}'.format(ii, jj)] = rec



output_values = {}
for i in outputs:
    output_values[i] = outputs[i].values

if 1:
    np.save('output_values.npy', output_values)

# plt.plot(np.linspace(0, duration, ii * CFL),
#          output_values['rec_4_pm_2200']['bell_radiation_pressure'])

read_outputs = np.load('output_values.npy',allow_pickle=True).item()


#raise SystemExit(0)
#%% compute the consecutive errors

abs_norm = []
for jj in mouth_pressures:
    pm_norm = []
    for ii in fact:
        new = read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure'][::ii]
        ref = read_outputs['rec_{}_pm_{}'.format(fact[-1], jj)]['bell_radiation_pressure'][::fact[-1]]

        pm_norm.append(np.max(np.abs(new - ref)) / np.max(np.abs(ref)))

    abs_norm.append(pm_norm)



mouth_pressures.reverse()



# CONVERGENCE
plt.figure(2)
for jj in range(len(mouth_pressures)):
    plt.loglog([duration/(ii*n_time_step_base) for ii in fact[:-1]], abs_norm[jj][:-1], 'o-')
plt.xlabel('Dt [s]')
plt.ylabel('sup norm of the difference')
plt.title('convergence')
plt.grid(True, 'major', linewidth=2)
plt.grid(True, 'minor', linewidth=0.5)
plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])


#%% plots and sounds
plt.figure(1)
for ii in [fact[0]]:
    for jj in [2200]:
        plt.title('Bell Radiation pressure ; tuyau L = 30cm, r = 1cm')
        plt.plot(np.linspace(0, duration, ii * n_time_step_base),
                 read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure'],
                 linewidth=1)
        # plt.plot(np.linspace(0, duration, ii * n_time_step_base),
        #          ADSR(2500, 0.2, 0.2, 0.65, 0.015)(np.linspace(0, duration, ii * n_time_step_base)))

plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
plt.xlabel('temps [s]')





NP = fact[0]
PM = mouth_pressures[0]
sound = outputs['rec_{}_pm_{}'.format(NP, PM)].values['bell_radiation_pressure']

sound = np.interp(np.linspace(0, duration, int(44100 * duration)),
                  np.linspace(0, duration, int(NP * n_time_step_base)),
                  sound)

audio = sound * (2**15 - 1) / np.max(np.abs(sound))
audio = audio.astype(np.int16)

try:
    # Start playback
    play_obj = sa.play_buffer(audio, 1, 2, 44100)

    # Wait for playback to finish before exiting
    play_obj.wait_done()
except Exception as e:
    print(e)



#%%
yoplait = 2
[print(np.polyfit(np.log([duration/(ii*n_time_step_base) for ii in fact[yoplait:-1]]),np.log(abs_norm[kk][yoplait:-1]),1)) for kk in range(len(abs_norm))]



# BELL RADIATION + Y REED, LINKED AXIS
plt.figure(3)
for ii in [1]:
    for jj in mouth_pressures[:]:
        ax1 = plt.subplot(2,1,1)
        plt.ylabel('Delta p')
        t = np.linspace(0, duration, ii * n_time_step_base)
        loc_plot = (constant_with_initial_ramp(jj)(t) -
                 read_outputs['rec_{}_pm_{}'.format(ii, jj)]['source_pressure'])
        plt.plot(t,
                 loc_plot,
                 linewidth=1)
        plt.fill_between(t,
                         loc_plot,
                         0,
                         loc_plot<0)

        ax2 = plt.subplot(2,1,2, sharex=ax1)
        plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
             read_outputs['rec_2_pm_{}'.format(jj)]['source_y'])
        plt.grid(True)
plt.ylabel('y anche')

plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))],
           loc='lower right')
plt.xlabel('temps [s]')



# Y REED
plt.figure(33)
for jj in mouth_pressures:
    plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
             read_outputs['rec_2_pm_{}'.format(jj)]['source_y'])
plt.xlabel('temps')
plt.ylabel('y anche')
plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
plt.grid(True)


# DELTA P FILLED
plt.figure(4)
for ii in [2]:
    for jj in mouth_pressures:
        t = np.linspace(0, duration, ii * n_time_step_base)
        loc_plot = (constant_with_initial_ramp(jj)(t) -
                 read_outputs['rec_{}_pm_{}'.format(ii, jj)]['source_pressure'])
        plt.plot(t,
                 loc_plot,
                 linewidth=1)
        plt.fill_between(t,
                         loc_plot,
                         0,
                         loc_plot<0)
plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
plt.xlabel('temps')
plt.ylabel('Delta p')
plt.grid(True,'major')


#%%
# d/dt DELTA P
plt.figure(4)
for ii in [fact[-1]]:
    for jj in [0]:
        jjj = mouth_pressures[jj]
        t = np.linspace(0, duration, ii * n_time_step_base)
        loc_plot = np.gradient((constant_with_initial_ramp(jjj)(t) -
                 read_outputs['rec_{}_pm_{}'.format(ii, jjj)]['source_pressure']))
        plt.plot(t,
                 loc_plot,
                 linewidth=1)
        plt.xlabel('temps')
        plt.ylabel('d/dt Delta p ; pm = {}'.format(jjj))


# plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])




#%%
# RAMP MOUTH PRESSURE
plt.figure(5)
for jj in mouth_pressures:
    plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
             constant_with_initial_ramp(ii)(np.linspace(0, duration, 2 * n_time_step_base)))

plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])




t = np.linspace(0, duration, fact[-1] * n_time_step_base)
loc_plot = np.gradient((constant_with_initial_ramp(PM)(t) -
                        read_outputs['rec_{}_pm_{}'.format(fact[-1], PM)]['source_pressure']))
plt.plot(t,
         loc_plot,
         linewidth=1)
plt.xlabel('temps [s]')
plt.ylabel('Delta p [Pa]')
plt.title(f'pm = {PM} Pa ; Delta P')



# Hilbert does not work ? So :
def swmax(vect, N):
    """
    Local Window Maximum with linear interpolation

    Parameters
    ----------
    vect : np array
        input signal.
    N : int
        size of window.

    Returns
    -------
    np array
        Local maximum.

    """

    L = int(np.ceil(vect.shape[0] / N))

    vect = np.append(vect,
                     np.zeros([(L * N) - vect.shape[0], 1]))

    vect = vect.reshape([L, N])

    env_sup = np.zeros(vect.shape)

    for i in range(vect.shape[0]):
        if i == 0:
            env_sup[i, :] = np.linspace((max(vect[i, :]))/N,
                                        max(vect[i, :]),
                                        N)
        else:
            env_sup[i, :] = np.linspace(((N-1)*max(vect[i-1, :]) + max(vect[i, :]))/N,
                                        max(vect[i, :]),
                                        N)
    return np.squeeze(env_sup.reshape((L * N, 1)))



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# enveloppes
ii = 8

for jj in mouth_pressures:
    loc_temp =read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure']
    loc_env = swmax(loc_temp,200*ii)
    #plt.plot(loc_temp)
    plt.plot(np.linspace(0,
                         duration * loc_env.shape[0]/ (ii*n_time_step_base),
                         loc_env.shape[0]),
             loc_env)
plt.xlabel('temps [s]')
plt.ylabel('[Pa]')
plt.title('enveloppe de Bell radiation pressure')
plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))],
           loc = 'lower right',
           framealpha = 1)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dérivées

ii = 8
t = np.arange(0,duration, duration/(ii*n_time_step_base))

for jj in mouth_pressures:
    loc_temp =read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure']
    loc_env = swmax(loc_temp,200*ii)
    #plt.plot(loc_temp)
    plt.plot(np.linspace(0,
                         duration * loc_env.shape[0]/ (ii*n_time_step_base),
                         loc_env.shape[0]),
             np.gradient(loc_env))
plt.xlabel('temps [s]')
plt.ylabel('[Pa]')
plt.title('dérivée de l\'enveloppe de Bell radiation pressure')
plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))],
           loc = 'lower right',
           framealpha = 1)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 # enveloppes en fonction de pm
plt.plot(mouth_pressures,
         [np.max(swmax(read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure'],510)) for jj in mouth_pressures],
         'o')
plt.xlabel('pm [Pa]')
plt.ylabel('pression [Pa]')
plt.title('maximum de bell radiation pressure en fonction de pm')
plt.grid(True,'major')

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# dérivée en fonction de PM

ii = 8
t = np.arange(0,duration, duration/(ii*n_time_step_base))

loc_grad =[]
loc_vitesse_exp =[]

for jj in mouth_pressures:
    loc_temp =read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure']
    loc_env = swmax(loc_temp,180*ii)
    env_t = np.linspace(0, duration * loc_env.shape[0]/ (ii*n_time_step_base), loc_env.shape[0])

    fit_window = np.where((loc_env < 0.3*np.max(loc_env))*
                          (env_t > 0.06))
    pfit=np.polyfit(env_t[fit_window], np.log(loc_env[fit_window]),1)

    # plt.plot(env_t, np.log(loc_env))
    # plt.plot(env_t[fit_window], np.log(loc_env[fit_window]))

    # plt.plot(t, loc_temp)
    # plt.plot(env_t, loc_env)
    # plt.plot(env_t[fit_window], np.exp(pfit[1]) * np.exp( pfit[0] * t[fit_window]))


    loc_pm = constant_with_initial_ramp(jj)(t)
    loc_pm_grad = np.gradient(loc_pm)

    loc_grad.append(np.max(np.gradient(loc_env)) / np.max(loc_pm_grad))
    loc_vitesse_exp.append(pfit[0])

    #plt.plot(loc_temp)
plt.plot(mouth_pressures,
         loc_vitesse_exp,
         'o')
plt.xlabel('PM [Pa]')
plt.ylabel(r'$\alpha$ dans P = a + exp[$\alpha$ t]')
plt.grid(True)
plt.title('taux de croissance exponentiel')


#%%
loc_grad =[]

for jj in mouth_pressures:
    loc_temp = read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure']
    loc_env = swmax(loc_temp,200*ii)

    t = np.linspace(0, duration * loc_env.shape[0]/ (ii*n_time_step_base), loc_env.shape[0])

    loc_pm = constant_with_initial_ramp(jj)(t)
    loc_pm_grad = np.gradient(loc_pm)

    loc_grad.append(t[int(np.mean(np.where(
        np.gradient(loc_env)==np.max(np.gradient(loc_env)))
        ))])


    #plt.plot(loc_temp)
plt.plot(mouth_pressures,
         loc_grad,
         'o')
plt.xlabel('PM [Pa]')
plt.ylabel('temps [s]')
plt.title('temps d\'attaque')
plt.grid(True)









# import scipy
# scipy.io.wavfile.write('pm_2000_60cm_moche.wav', 44100, audio)

plt.show()
