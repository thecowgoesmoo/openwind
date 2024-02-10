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


"""Methods to properly export temporal simulated data to an audio file."""

import numpy as np
import warnings
from scipy.io import wavfile
from scipy.signal import butter, lfilter

from openwind.continuous import Reed1dof, InstrumentPhysics, Scaling
from openwind import Player



# ========== SIGNAL PROCESSING ==========

def antialias(signal, dt, cutoff_freq=18000, order=5):
    """
    Decimate high frequencies using a digital Butterworth filter.

    Parameters
    ----------
    signal : array
        the rough signal to treat.
    dt : float
        the step time of the input signal (1/sample rate).
    cutoff_freq : float, optional
        The highest frequency over which the signal is filtered. The default is
        18000.
    order : int, optional
        the filter order. The default is 5.

    Returns
    -------
    array
        The filtered signal.

    """
    """"""
    nyq = 0.5 / dt
    normal_cutoff = cutoff_freq / nyq
    if normal_cutoff > 1:
        # Can't antialias beyond the Nyquist frequency!
        return signal
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, signal)


def resample(signal, dt, samplerate=44100, cutoff_freq=18000):
    """
    Change sampling rate of signal, applying anti-aliasing if needed.

    Parameters
    ----------
    signal : array
        the rough signal to treat.
    dt : float
        the step time of the input signal (1/(old sample rate)).
    samplerate : float, optional
        The *new* sample rate. The default is 44100.
    cutoff_freq : float, optional
        The highest frequency over which the signal is filtered. The default is
        18000.

    Returns
    -------
    array
        The resampled signal.

    """
    signal = antialias(signal, dt)
    Tmax = dt*len(signal)
    ts_before = np.linspace(0, Tmax, len(signal))
    ts_after = np.arange(0, Tmax, 1.0/samplerate)
    return np.interp(ts_after, ts_before, signal)


def export_mono(filename, out, ts, verbose=True):
    """
    Export simulation data to an audio file.

    Resamples with antialiasing, and normalizes amplitude so that the highest
    peak is at 0 dB.

    Parameters
    ----------
    filename : str
        The name of the file in which write the signal.
    out : array(float)
        1D array of data to export.
    ts : array(float)
        1D array of times of the simulation. We assume constant increment.
    """
    dt = ts[1] - ts[0]
    assert np.allclose(ts[1:], ts[:-1] + dt)

    samplerate = 44100
    out_interp = resample(out, dt, samplerate=samplerate)

    out_interp /= np.max(np.abs(out_interp))  # Normalize to [-1,1]
    wavfile.write(filename, samplerate, out_interp)

    if verbose:
        print("Wrote audio to", filename)

# ========== SCALING OF PLAYER ==========

def scaling_player(unsclaled_dic, instru_geom, temperature, humidity=.5,
                   carbon=4e-4, **physics_opt):
    """
    Convert a excitator dic with dimensionfull parameters to dimensionless dic

    .. warning::
        When the mouth pressure varies with time (=callable) it can not be
        easily converted. This must be done afterward by reinstanciating a
        callable and using the "closing_pressure" value from the new dic to
        scaled the pressure (see example).

    Parameters
    ----------
    unsclaled_dic : dic
        The dictionnary with dimensionfull parameters.
    instru_geom : :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
        The instrument geometry object associated to the player.
    temperature : float
        The temperature in the instrument.
    humidity : float, optional
        the relative humidity rate in the instrument. The default is .5.
    carbon : float, optional
        the carbon rate. The default is 4e-4.
    **physics_opt : kwargs
        other option for :py:class:`Physics<openwind.continuous.physics.Physics>`.

    Raises
    ------
    ValueError
        Error if the excitator type is different than 'Reed1dof'.

    Returns
    -------
    scaled_dict : dic
        The player dic with dimensionless parameters.

    Examples
    --------
    Here the conversion of a player dictionnary with a time varying mouth
    pressure:

    .. code-block:: python
        :emphasize-lines: 14,16

        from openwind import InstrumentGeometry
        from openwind.temporal.utils import scaling_player

        pm = lambda t: t*5000 # here the pressure increases linearly
        dim_player = {"excitator_type" : "Reed1dof", "model" : "inwards",
                      "mouth_pressure" : pm,
                      "opening" : 1e-4, "mass" : 3e-6, "section" : 14.6e-5,
                      "pulsation" : 2*np.pi*3700, "dissip" : 3000,
                      "width" : 3e-2, "contact_pulsation": 0,
                      "contact_exponent": 4}

        cylinder = [[0, .5, 5e-3, 5e-3, 'linear']]
        instru_geom = InstrumentGeometry(cylinder)
        scaled_player = scaling_player(dim_player, instru_geom, 20)
        gamma = lambda t: t*5000/scaled_player['closing_pressure']
        scaled_player['gamma'] = gamma
        print(scaled_player)

    """
    if unsclaled_dic['excitator_type']!='Reed1dof':
        raise ValueError("Only 'Reed1dof' player can be scaled.")

    reed_unscaled = Reed1dof(unsclaled_dic, 'label', Scaling(), 'PH1')
    phy = InstrumentPhysics(instru_geom, temperature, Player(), False,
                            humidity=humidity, carbon=carbon, **physics_opt)

    rho, c = phy.get_entry_coefs('rho', 'c')
    radius = instru_geom.get_main_bore_radius_at(0)
    Zc = rho*c/(np.pi*radius**2)

    scaled_list = reed_unscaled.get_dimensionless_values(1e-8, Zc, rho) #for some enveloppe an error occurs at t=0
    params_list = ['gamma', 'zeta', 'kappa', 'qfactor', 'pulsation',
                   'contact_stifness', 'contact_exponent', 'epsilon']

    scaled_dict = dict()
    scaled_dict["excitator_type"] = "Reed1dof_scaled"
    scaled_dict["model"] = unsclaled_dic["model"]
    if callable(unsclaled_dic['mouth_pressure']):
        warnings.warn('The mouth pressure is not converted because it is a '
                      'callable. Please instanciate manually the "gamma" by '
                      'using: gamma=mouth_pressure/closing_pressure')
    else:
        scaled_dict['gamma'] = scaled_list[0]

    for k, param in enumerate(params_list[1:-1]):
        scaled_dict[param] = scaled_list[k+1]
    scaled_dict["opening"] = unsclaled_dic["opening"]
    scaled_dict["closing_pressure"] = reed_unscaled.get_Pclosed(0)

    return scaled_dict
