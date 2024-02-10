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

"""Tools for input/output of impedance data, and impedance visualization."""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import warnings
from openwind.continuous import Physics

def write_impedance(frequencies, impedance, filename, column_sep=' '):
    """
    Write the impedance in a file.

    The file has the format

    .. code-block:: shell

        (frequency)  (real part of impedance)  (imaginary part of impedance)


    Parameters
    ----------
    frequencies : list or np.array of float
        The frequency at which is evaluated the impedance
    impedance : list or np.array of float
        The complexe impedance at each frequency
    filename : string
        The name of the file in which is written the impedance (with the
        extension)
    column_sep : str, optional
        The column separator. Default is ' ' (space)

    """
    f = open(filename, "w")
    assert len(frequencies) == len(impedance)
    for k in range(len(frequencies)):
        f.write('{:e}{sep}{:e}{sep}{:e}\n'.format(frequencies[k],
                                                       np.real(impedance[k]),
                                                       np.imag(impedance[k]),
                                                       sep=column_sep))


def read_impedance(filename, df_filt=None, column_sep=None):
    """
    Read an impedance file.

    The impedance file must have the following format:

    * first column: the frequency in Hertz
    * second column: the real part of the impedance
    * third column: the imaginary part of the impedance

    .. code-block:: shell

        # any comment
        (frequency)  (real part of impedance)  (imaginary part of impedance)

    The file can contain comments line beginning with a #.
    It is possible to filter the impedance by fixing the cutoff frequency step.
    It is performed by `scipy.signal.butter() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html>`_
    and `scipy.signal.filtfilt() <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.filtfilt.html>`_

    Parameters
    ----------
    filename : string
        The name of the file containing the impedance (with the extension).
    df_filt : float, optional
        The frequency step in Hz use to filter the impedance with a low-pass
        filter. Use `None` (Default value) to not filter the signal.
    column_sep : str, optional
        Column separator. Default is None corresponding to merged whitespace.

    Returns
    -------
    frequencies : np.array of float
        The frequencies at which is evaluated the impedance.
    impedance : np.array of float
        The complexe impedance at each frequency.

    Warnings
    -------
    The NaN values are excluded of the returned arrays

    """
    def parse_line(line):
        # Anything after a '#' is considered to be a comment
        line = line.split('#')[0]
        # Split the lines according to whitespace
        return line.split(column_sep)
    with open(filename) as file:
        lines = file.readlines()
    file_freq = []
    file_imped = []
    for line in lines:
        contents = parse_line(line)
        if len(contents) > 0:
            file_freq.append(float(contents[0]))
            file_imped.append(float(contents[1]) + 1j*float(contents[2]))
    frequencies = np.array(file_freq)
    impedance = np.array(file_imped)
    frequencies = frequencies[~np.isnan(file_imped)]
    impedance = impedance[~np.isnan(impedance)]
    delta_f = np.mean(np.diff(frequencies))

    if not not df_filt:
        ratio = delta_f/df_filt
        b, a = signal.butter(2, ratio)
        impedance = signal.filtfilt(b, a, impedance)

    return frequencies, impedance

def convert_frequencies_temperature(frequencies, temperature_in, temperature_out,
                                    humidity_in=.5, humidity_out=.5,
                                    carbon_in=4e-4, carbon_out=4e-4):
    """
    Convert a frequency axis according to the temperature.

    Only the modification of the sound celerity is accouted:

    .. math::
        f_{out} = f_{in} \\frac{c_{out}}{c_{in}}}

    with the :math:`c_i` the speed of sound at the given temperature and air composition

    Parameters
    ----------
    frequencies : array of float
        The "input" frequencies in Hz.
    temperature_in : float
        The input temperature in °C.
    temperature_out : float
        The output temperature in °C.

    Returns
    -------
    freq_ref : array of float
        The output frequencies in Hz.

    """
    if humidity_out>1 or humidity_in>1:
        raise ValueError('The humidity rate must be within [0,1].')
    phys_meas = Physics(temperature_in, humidity=humidity_in, carbon=carbon_in)
    phys_comp = Physics(temperature_out, humidity=humidity_out, carbon=carbon_out)
    ratio_temp = phys_comp.c(0) / phys_meas.c(0)

    freq_ref = frequencies*ratio_temp
    return freq_ref

def plot_impedance(frequencies, impedance, Zc0=1, dbscale=True, figure=None,
                   label=None, modulus_only=False, remarkable_freqs =None,
                   admittance=False, **kwargs):
    """
    Plot the impedance of the instrument.

    Parameters
    ----------
    impedance : array(complex)
        Sequence of impedance values for successive frequencies.
    Zc0 : float
        Characteristic impedance (or 1 if the impedance
        is already nondimensionalized).
    dbscale : bool, optional
        If True, the modulus is plotted with a dB scale, else a linear scale is
        used. The default is True.
    figure : matplotlib.Figure, optional
        Where to plot. By default, opens a new figure.
    label : str, optional
        The label associated to the curve. Default is None
    modulus_only: bool, optional
        If true, only the modulus of the impedance is plotted
    remarkable_freqs: tuple of (table of floats, table of labels), optional
        Two tables of the same size of remarkable frequencies that will appear on the graph
    admittance: bool, optional
        If true plot the admittance instead. Default is False.
    **kwargs : key word arguments
        Passed to `plt.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.
    """
    if not figure:
        fig = plt.figure()
    else:
        fig = figure
    ax = fig.get_axes()

    if admittance:
        ylabel = '|Y|'
        ylabel_angle = "angle(Y)"
        if Zc0 != 1:
            ylabel += '/Yc'
        y = Zc0/impedance
    else:
        ylabel = '|Z|'
        ylabel_angle = "angle(Z)"
        if Zc0 != 1:
            ylabel += '/Zc'
        y = impedance/Zc0

    if remarkable_freqs:
        F = remarkable_freqs[0]
        Lab = remarkable_freqs[1]
    else:
        F   = []
        Lab = []

    if len(ax) < 1 and not modulus_only:
        ax = [fig.add_subplot(2, 1, 1)]
        ax.append(fig.add_subplot(2, 1, 2, sharex=ax[0]))
    elif len(ax) < 1:
        ax.append(fig.add_subplot(1, 1, 1))
    [a.grid() for a in ax]


    if dbscale:
        ax[0].plot(frequencies, 20*np.log10(np.abs(y)), label=label, **kwargs)
        ylabel += ' (dB)'

    else:
        ax[0].plot(frequencies, np.abs(y), label=label, **kwargs)
    ax[0].set_ylabel(ylabel)
    ax[0].grid('on')

    limi = ax[0].get_ylim()
    for (f,lab) in zip(F,Lab):
        ax[0].plot([f,f],limi,'k--')
        ax[0].text(f,limi[1]*0.9, lab)

    if label:
        ax[0].legend(loc='upper right')

    if len(ax)>1 and not modulus_only:
        ax[1].plot(frequencies, np.angle(y), **kwargs)
        ax[1].grid('on')
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel(ylabel_angle)
        ax[1].get_yaxis().set_ticks([-np.pi/2, -np.pi/4, 0, np.pi/4, np.pi/2])
        ax[1].get_yaxis().set_ticklabels(
            ['$-\pi/2$', '$-\pi/4$', '0', '$-\pi/4$', '$\pi/2$']
        )
    else:
        ax[0].set_xlabel("Frequency (Hz)")


def plot_reflection(frequencies, impedance, Zc0,
                    complex_plane=True, figure=None, **kwargs):
    """Plot the reflection function of the instrument.

    Parameters
    ----------
    impedance : array(complex)
        Sequence of impedance values for successive frequencies.
    Zc0 : float
        Characteristic impedance (or 1 if the impedance
        is already nondimensionalized).
    complex_plane : bool, optional
        Whether to plot in the complex plane,
        i.e. with axes (x, y) = (real, imag),
        rather than (x, y) = (freq, unwrapped argument).
        The default is True.
    figure : matplotlib.Figure, optional
        Where to plot. By default, opens a new figure.
    **kwargs : key word arguments
        Passed to `plt.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.
    """
    if not figure:
        fig = plt.figure()
    else:
        fig = figure
    ax = fig.get_axes()
    if len(ax) < 1 and complex_plane:
        ax = [fig.add_axes([.1, .1, .8, .8])]
    elif len(ax) < 2 and not complex_plane:
        ax = [fig.add_subplot(2, 1, 1)]
        ax.append(fig.add_subplot(2, 1, 2, sharex=ax[0]))

    Ref = (impedance - Zc0)/(impedance + Zc0)
    if complex_plane:
        ax[0].plot(np.real(Ref), np.imag(Ref), **kwargs)
        ax[0].legend()
        ax[0].set_xlabel("real(R)")
        ax[0].set_ylabel("imag(R)")
    else:
        ax[0].plot(frequencies, (np.abs(Ref)), **kwargs)
        ax[0].set_ylabel("|R|")
        ax[0].legend()
        ax[1].plot(frequencies, (np.unwrap(np.angle(Ref))), **kwargs)
        ax[1].set_xlabel("Frequency (Hz)")
        ax[1].set_ylabel("angle(R)")

def resonance_frequencies(frequencies, impedance, k=5, display_warning=True):
    """
    Find the first k resonance frequencies of an impedance without noise.

    We define a resonance frequency as a frequency where the phase is zero
    and decreasing.

    Parameters
    ----------
    k : int, optional
        The number of resonance included. The default is 5.

    Returns
    -------
    list of float
    """

    return resonance_peaks_from_phase(frequencies, impedance, k, display_warning)[0]


def resonance_peaks_from_phase(frequencies, impedance, k=5, display_warning=True):
    r"""
    Find the first k resonance of an impedance without noise from phase considerations.

    Here, a resonance frequency is defined as a frequency where the phase is zero
    and decreasing, and the Q-factor is related to the slope at this frequency:

    .. math::
        Q = - \frac{\omega_0}{2} \frac{d \phi}{d \omega}

    Parameters
    ----------
    frequencies: array
        The frequential axis
    impedance: array of complex
        The impedance array
    k : int, optional
        The number of resonance included. The default is 5.

    Returns
    -------
    f_res : list of float
        The frequencies of resonance
    Q : list of float
        The quality facors
    Z_res : list of float
        The impedance value at the frequency of resonance (real by def)
    """
    if display_warning:
        warnings.warn("Here, the res charac. are estimated from the phase, assuming"
                      " no interaction between mode. If you can, use 'modal' "
                      "computation to have a more rigorous estimation.")

    phase = np.angle(impedance)
    signchange = np.diff(np.sign(phase)) < 0
    no_discontinuity = np.abs(np.diff(phase)) < np.pi
    valid_indices, = np.where(signchange & no_discontinuity)
    indi = valid_indices[:k]

    # Find the zero crossing by linear interpolation
    df = frequencies[indi+1] - frequencies[indi]
    dphi = phase[indi+1] - phase[indi]
    f_res = frequencies[indi] - phase[indi] * (df/dphi)

    # compute the Q-factor
    Q = -.5*f_res*dphi/df

    # at the resonance, by def, the imaginary part is 0
    # the modulus is estimated by fitting a parabolic

    Z_res = list()
    for p, n in enumerate(indi):
        pol = np.polyfit(frequencies[n-1:n+2], np.abs(impedance[n-1:n+2]), 2)
        Z_res.append(np.polyval(pol, f_res[p]))

    return f_res, Q, Z_res


def antiresonance_frequencies(frequencies, impedance, k=5, display_warning=True):
    """
    Find the first k anti-resonance frequencies of an impedance without noise.

    We define a resonance frequency as a frequency where the phase is zero
    and increasing.

    Parameters
    ----------
    k : int, optional
        The number of resonance included. The default is 5.

    Returns
    -------
    list of float
    """
    return resonance_frequencies(frequencies, np.conjugate(impedance), k, display_warning)

def antiresonance_peaks_from_phase(frequencies, impedance, k=5, display_warning=True):
    """
    Find the first k anti-resonances of an impedance without noise.

    Here, an anti-resonance frequency is defined as a frequency where the phase is zero
    and increasing,

    Parameters
    ----------
    k : int, optional
        The number of resonance included. The default is 5.

    Returns
    -------
    f_antires : list of float
        The frequencies of anti-resonance
    Q : list of float
        The quality facors
    Z_antires : list of float
        The impedance value at the frequency of antiresonance (real by def)
    """

    f_antires, Q, Y_antires = resonance_peaks_from_phase(frequencies, 1/impedance, k, display_warning)
    Z_antires = [1/y for y in Y_antires]
    return f_antires, Q, Z_antires

def find_peaks_measured_impedance(frequencies, impedance, Npeaks=10, fmin=0, display_warning=True):
    """
    Find peaks frequency and magnitudes, for data with noise.

    The peaks are found with `signal.find_peaks <https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.find_peaks.html>`_.
    The magnitudes are estimated by fitting a parabol on the modulus in dB.
    The frequencies are estimated by the zero of the linear regression of the
    imaginary part.

    Parameters
    ----------
    frequencies : np.array
        Frequencies values
    impedance : np.array
        The impedance value at each frequency.
    Npeaks : int, optional
        The number of peaks researched. The default is 10.
    fmin : float, optional
        The minimal frequency considered. The default is 0.

    Returns
    -------
    res_freq : np.array
        Array of peaks frequency.

    res_Q : np.array
        Array of the peaks Q-factors.

    res_mag : np.array
        Array of the peaks magnitude.

    """
    if display_warning:
        warnings.warn("This function has been modified after version 0.9.0")
        warnings.warn("Here, the res charac. are estimated from the phase, assuming"
                      " no interaction between mode.")

    impedance_db = 20*np.log10(np.abs(impedance))
    ind_peaks, info = signal.find_peaks(impedance_db, prominence=10, height=0)
    amp_peaks = info['peak_heights']
    f_peaks = frequencies[ind_peaks]

    ind_peaks = ind_peaks[f_peaks > fmin]
    amp_peaks = amp_peaks[f_peaks > fmin]
    f_peaks = f_peaks[f_peaks > fmin]

    if len(f_peaks) > Npeaks:
        f_peaks = f_peaks[:Npeaks]
        amp_peaks = amp_peaks[:Npeaks]
    res_mag = np.zeros(Npeaks)
    res_freq = np.zeros(Npeaks)
    res_Q = np.zeros(Npeaks)

    for k, f_peak in enumerate(f_peaks):
        # the sample range where to do the fit
        ind_min = max(ind_peaks[k]-2, 0)
        ind_max = min(ind_peaks[k]+2, len(frequencies)-1)
        f1 = np.min([f_peak*2**(-10/1200), frequencies[ind_min]])
        f2 = np.max([f_peak*2**(+10/1200), frequencies[ind_max]])
        ind_fit = np.logical_and(frequencies >= f1, frequencies <= f2)

        # frequency estimation by linear fitting of the phase
        x = frequencies[ind_fit]
        y = np.angle(impedance[ind_fit])
        polf = np.polyfit(x, y, 1)
        res_freq[k] = -polf[1]/polf[0]

        # Q-factor from the phase
        res_Q[k] = 0.5*polf[1] # -0.5*polf[0]*res_freq

        # amp with parabola fit
        x_a = frequencies[ind_fit] - res_freq[k] # center and scaled to improve fit performance
        y_a = impedance_db[ind_fit] / amp_peaks[k]
        pol = np.polyfit(x_a, y_a, 2)
        res_mag[k] = 10**((pol[2] * amp_peaks[k])/20)

    return  res_freq, res_Q, res_mag

def get_equal_temperament_frequency(midi, tuning=440, nb_harmo=1):
    # midi 69 corresponds to {tuning} Hz
    r = np.power(2,1/12)
    midi = np.array(midi)
    freq = tuning*np.power(r,midi-69)
    if(midi.size>1 and nb_harmo>1):
        print("Harmonics only implemented for one midi number")
    freqs = (np.arange(0,nb_harmo)+1)*freq
    return freqs

def match_freqs_with_notes(f_, concert_pitch_A=440, transposition = 0, display=False, simple_name=True):
    """
    Matches provided freqs with notes frequencies in Hz, deviation in cents and notes names
    The user can specify a concert pitch and a transposing behavior for the instrument.

    Parameters
    ----------
    f_: list of float
        Frequencies to match in Hz.
    concert_pitch_A: float, optional
        Frequency of the concert A4, in Hz.
        Default value is 440 Hz.
    transposition: int or string, optional
        indicates if the instrument is transposing.
        If an integer is given, it must be the number of semitones between the played C and the actual heard C
        If a note name is given, the number of semitones will be deduced (-2 for "Bb" instrument, for instance)
        Availables notes are : {"Eb":-9,"F":-7,"A":-3,"Bb":-2,"C":0,"F+":5,"A+":9}
        Default is 0.
    display : boolean, optional
        If true, display the result for each mode. Default is False.

    Returns
    -------
    tuple of 3 lists
       - The closest notes frequencies (float)
       - The deviation of the resonance frequencies (float)
       - The names of the closest notes, in the given concert pitch and transposition system (string)

    """
    if(isinstance(transposition, str)):
        # convert note name in a number of semitones with respect to C
        conversion = dict({"Eb":-9,"F":-7,"A":-3,"Bb":-2,"C":0,"F+":5,"A+":9})
        if(transposition in conversion):
            nb_semitones_transp = conversion[transposition]
        else:
            nb_semitones_transp = 0
    else:
        nb_semitones_transp = transposition
    concert_pitch_A = concert_pitch_A*2**(nb_semitones_transp/12)
    if(concert_pitch_A==415 or concert_pitch_A==392 or simple_name):
        # for historical tunings only use usual alterations
        notes = ['A','Bb','B','C','C#','D','Eb','E','F','F#','G','G#']
    else:
        notes = ['A','A#/Bb','B','C','C#/Db','D','D#/Eb','E','F','F#/Gb','G','G#/Ab']

    devs   = []
    pitches = []
    names = []
    for (i,f) in enumerate(f_):
        abs_dev = 1200*np.log2(f/concert_pitch_A)
        # how much semitones above the pitch A ?
        (octave, oct_dev) = divmod(abs_dev/100, 12)
        note_nb = round(oct_dev)
        cent = (oct_dev-note_nb)*100
        if(note_nb == 12):
            note_nb = 0
            octave += 1
        fn = concert_pitch_A*2**((12*octave+note_nb)/12)

        octave_disp = octave + 4 # notation A4 for pitch A
        if(note_nb >=3): # english notation changes at C
            octave_disp += 1
        note_name = notes[note_nb] + str(int(octave_disp))

        msg =f"{i+1} - frequency {f:.2f} Hz : {cent:3.2f} cents away from note {note_name} ({fn:.2f} Hz)"
        if nb_semitones_transp!=0:
            msg += f" (transposition: {nb_semitones_transp:+d} semitones)"
        if display:
            print(msg)

        names.append(note_name)
        pitches.append(fn)
        devs.append(cent)
    return (pitches, devs, names)
