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
import warnings
from numpy import pi


# ===============     CONTROL ENVELOPES     =============== #


# COMPOSITE FUNCTIONS (most used):

#     gate(t1, t2, t3, t4, shape='fast', a=1)
#         from 0 to 'a' value and back, with different ramps

#     ADSR(t1, t2, amplitude, A, D, S, R, shape='fast',
#          trem_a=0, trem_freq=5, trem_gate=gate(0, 0.1, 0.8, 1, 'slow'))
#         complete ADSR with possible tremolo

# (OLD) -- can be replaced by gate
#     constant_with_initial_ramp(pm_max=5000, t_car=5e-2, t_down=1e10)
#         from 0 to constant value with cosine fade


# SIMPLE FUNCTIONS :

# (OLD) -- can be replaced by gate
#     triangle(pm_max=5000, duration=3.0)
#         linear ramp from 0 to max value and back

#     bump(A, t0, width)
#         C-inf rise and fall

# BASIC FUNCTIONS (not for direct use):

#     fade(t1, x1, t2, x2, shape='fast')
#         C-inf fade between two values

#     exp(t1, x1, t2, x2)
#         joining two points with exponential func

#     cos(t1, x1, t2, x2)
#         joining two points with cosine func

#     tremolo(A, frequency, phase=0)
#         sine oscillation of given amplitude, frequency and phase

#     constant(t1, t2, C)
#         just a constant between t1 and t2

#     ramp(t1, x1, t2, x2)
#         ramp between 2 values, 'fast', 'slow', 'lin' or 'cos'

#     from_file()
#         not yet implemented


# %%  FUNCTION OPERATIONS


def add(f, g):
    """
    Adding two functions (temporal curves) together

    Parameters
    ----------
    f : function
    g : function

    Returns
    -------
    function
        return new function h = f + g

    """
    def env(t):
        return f(t) + g(t)
    return env


def substract(f, g):
    """
    Substracting a function (temporal curve) from another

    Parameters
    ----------
    f : function
    g : function

    Returns
    -------
    function
        return new function h = f - g

    """
    def env(t):
        return f(t) - g(t)
    return env


def multiply(f, g):
    """
    Multiplying two functions (temporal curves) together

    Parameters
    ----------
    f : function
    g : function

    Returns
    -------
    function
        return new function h = f * g

    """
    def env(t):
        return f(t) * g(t)
    return env

# %%  COMPOSITE FUNCTIONS (PROBABLY MOST USED)  %% #


def gate(t1, t2, t3, t4, shape='fast', a=1):
    """
    On-Off gate between 0 and 'a' with fade-in and fade-out ramps

    Parameters
    ----------
    t1 : float
        begin time of first (increasing) ramp.
    t2 : float
        end time of first ramp.
    t3 : float
        begin time of second (decreasing) ramp.
    t4 : float
        end time of second ramp.
    shape : string -- by default 'fast'
        ramp shape : ('fast', 'slow', 'linear', 'cos').
    a : float, optional
        amplitude of gate. The default is 1.

    Returns
    -------
    np.array :
        env(t) function, with a gate between t1 and t4, else 0

    """
    if [t1, t2, t3, t4] != sorted([t1, t2, t3, t4]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    if (isinstance(shape, str) is False) & (len(shape) != 2):
        warnings.warn('shape should be str or list of 2.' +
                      ' Proceeding with default "fast" ')
        shape = 'fast'

    if t1 == t2:
        shape = ['fast', shape]
    if t3 == t4:
        if isinstance(shape, str):
            shape = [shape, 'fast']
        else:
            shape[1] = 'fast'

    if isinstance(shape, str):
        def env(t):
            return (fade(t1, 0, t2, a, shape)(t) +
                    constant(t2, t3, a)(t) +
                    fade(t3, a, t4, 0, shape)(t))
    else:
        def env(t):
            return (fade(t1, 0, t2, a, shape[0])(t) +
                    constant(t2, t3, a)(t) +
                    fade(t3, a, t4, 0, shape[1])(t))

    return env


def ADSR(t1, t2, amplitude,
         A, D, S, R,
         shape='fast',
         trem_a=0, trem_freq=5, trem_gate=gate(0, 0.1, 0.8, 1, 'slow')):
    """
    Attack-Decay-Sustain-Release envelope

    .. image :: https://files.inria.fr/openwind/pictures/ADSR.png

    Parameters
    ----------
    t1 : float
        begin time.
    t2 : float
        end time.
    amplitude : float
        peak amplitude.
    A : float
        attack time.
    D : float
        decay time.
    S : float
        sustain level relative to amplitude.
    R : float
        release time.
    shape : string -- by default 'fast'
        ramp shape : ('fast', 'slow', 'linear', 'cos').
    trem_a : float, optional
        amplitude of tremolo during the sustain, relative to peak amplitude.
        The default is 0.
    trem_freq : float, optional
        frequency of tremolo during the sustain. The default is 5.
    trem_gate : env(t) function, optional
        envelope of tremolo during the sustain.
        The default is gate(0, 0.1, 0.8, 1, 'slow').

    Returns
    -------
    np.array :
        env(t) function returning the ADSR envelope when t1 < t < t2, else 0.

    """
    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    if shape == 'linear':
        def env(t):
            sustain = (constant(t1 + A + D, t2 - R, S)(t) +
                       (tremolo(trem_a, trem_freq, 0)(t) *
                        trem_gate(((t - t1 - A - D) /
                                  ((t2 - R) - (t1 + A + D))))))

            return (amplitude * (ramp(t1, 0, t1 + A, 1)(t) +
                                 ramp(t1 + A, 1, t1 + A + D, S)(t) +
                                 sustain +
                                 ramp(t2 - R, S, t2, 0)(t)))

    elif isinstance(shape, str):
        def env(t):
            sustain = (constant(t1 + A + D, t2 - R, S)(t) +
                       (tremolo(trem_a, trem_freq, 0)(t) *
                        trem_gate(((t - t1 - A - D) /
                                  ((t2 - R) - (t1 + A + D))))))

            return (amplitude * (fade(t1, 0, t1 + A, 1, shape)(t) +
                                 fade(t1 + A, 1, t1 + A + D, S, shape)(t) +
                                 sustain +
                                 fade(t2 - R, S, t2, 0, shape)(t)))

    elif (isinstance(shape, list) * [isinstance(i, str) for i in shape] *
          (len(shape) == 3)):
        def env(t):
            sustain = (constant(t1 + A + D, t2 - R, S)(t) +
                       (tremolo(trem_a, trem_freq, 0)(t) *
                        trem_gate(((t - t1 - A - D) /
                                  ((t2 - R) - (t1 + A + D))))))

            return (amplitude * (fade(t1, 0, t1 + A, 1, shape[0])(t) +
                                 fade(t1 + A, 1, t1 + A + D, S, shape[1])(t) +
                                 sustain +
                                 fade(t2 - R, S, t2, 0, shape[2])(t)))
    else:
        def env(t):
            sustain = (constant(t1 + A + D, t2 - R, S)(t) +
                       (tremolo(trem_a, trem_freq, 0)(t) *
                        trem_gate(((t - t1 - A - D) /
                                  ((t2 - R) - (t1 + A + D))))))

            return (amplitude * (fade(t1, 0, t1 + A, 1, 'fast')(t) +
                                 fade(t1 + A, 1, t1 + A + D, S, 'fast')(t) +
                                 sustain +
                                 fade(t2 - R, S, t2, 0, 'fast')(t)))
    return env


def dirac_flow(duration, amplitude=1e-7, sign=-1, shape='cos4'):
    """Short smooth impulse.

    Approximates a Dirac distribution, with a smooth function.


    Parameters
    ----------
    max_frequency : float
        A characteristic frequency of the pulse.
        The actual duration of the pulse is 2/max_frequency.
        Setting this too high may result in aliasing during the simulation.
        Default is 2e4 (20kHz).
    amplitude : float, optional
        Injected volume of air, in m^3. Default is 1e-7 = 100 mm^3.
    sign : {-1, 1}, optional
        Sign of the output.
        1 means it is an impulse of exiting flow,
        -1 means incoming flow. Default is -1.
    shape : {'cos', 'bump'}, optional
        Shape of the signal.
        'cos2' : cos(x)**2, -pi/2<x<pi/2 \
            Fairly low frequency content, but only C1 regularity
        'cos4' : cos(x)**4, -pi/2<x<pi/2 \
            C3 regularity
        'bump' : exp(-1/(1-t)**2), t<1 \
            C-infinite, but with quite a bit of high frequencies

    Returns
    -------
    dirac : Callable[Float]
        Function of time approximating a Dirac distribution.

    See Also
    --------
    openwind.continuous.Flow
    """
    if shape == 'bump':
        integral_of_pulse = 0.443993816168079437823048921170552663
        def _pulse(t, pulse_length):
            x = 2*t/pulse_length - 1
            pulse = np.nan_to_num(np.exp(-1 / (1 - abs(x)**2))) * (abs(x) < 1)
            normalized = pulse / (integral_of_pulse * pulse_length)
    #        if normalized != 0:
    #            print("Dirac is giving away non-zero values!", normalized)
            return sign * amplitude * normalized
    elif shape == 'cos2':
        def _pulse(t, pulse_length):
            x = np.pi/2*(2*t/pulse_length - 1)
            pulse = (abs(x) < np.pi/2) * np.cos(x)**2
            normalized = pulse/pulse_length
            return sign * amplitude * normalized
    elif shape == 'cos4':
        def _pulse(t, pulse_length):
            x = np.pi/2*(2*t/pulse_length - 1)
            pulse = (abs(x) < np.pi/2) * np.cos(x)**4
            normalized = pulse/pulse_length * 4/3
            return sign * amplitude * normalized

    return lambda t: _pulse(t, duration)


def chirp_flow():
    """Logarithmic chirp.

    Not yet implemented.
    """
    raise NotImplementedError()



# OLD : can easily be replaced by gate
def constant_with_initial_ramp(pm_max=5000, t_car=5e-2, t_down=1e10):
    """Curve starting at zero and reaching a constant mouth pressure.

    Parameters
    ----------
    pm_max : float
        Maximal value
    t_car : float
        Time to reach the value
    t_down : float
        Time when pressure source turns off
    """
    def pm(t):
        if np.isscalar(t):  # For scalars
            if 0 < t < t_car:  # Lighter computation for scalars
                return pm_max * (1 - np.cos(pi * t / t_car)) / 2
            elif 0 < t - t_down < t_car:
                return pm_max * (1 + np.cos(pi * (t - t_down) / t_car)) / 2
            elif t_car <= t <= t_down:
                return pm_max
            else:
                return 0
        ramp = 1.0 * (0 < t) * (t < t_car) * (1 - np.cos(pi * t / t_car)) / 2
        ramp2 = ((0 < t - t_down) * (t - t_down < t_car) *
                 (1 + np.cos(pi * (t - t_down) / t_car)) / 2)
        return pm_max * (ramp + (t_car <= t) * (t <= t_down) + ramp2)

    return pm


# %% SIMPLE FUNCTIONS  %% #

# OLD : can easily be replaced by gate
def triangle(pm_max=5000, duration=3.0):
    """
    Curve starting at zero, linearly increasing to pm_max until duration/2,
    and decreasing afterwards.

    Parameters
    ----------
    pm_max : float, optional
        The maximal value. The default is 5000.
    duration : float, optional
        The total duration. The default is 3.0.

    Returns
    -------
    np.array
        The triangle env(t).

    """

    half_dur = duration / 2

    def pm(t):
        ramp1 = pm_max * (t / half_dur) * (0 < t) * (t <= half_dur)
        ramp2 = (pm_max * ((duration-t) / half_dur) *
                 (t >= half_dur) * (t < duration))
        return ramp1 + ramp2

    return pm


def bump(A, t0, width):
    """
    C-inf bump function, with zero value and derivative at t0 +/- time_scale/2

    Parameters
    ----------
    A : float
        maximum amplitude.
    t0 : float
        centre time.
    width : float
        full width of bump.

    Returns
    -------
    np.array :
        env(t) function, with bump when :math:`t0-width/2 < t < t0+width/2`,
        else 0.

    """
    def env(t):
        with np.errstate(all='ignore'):
            return A * ((np.nan_to_num(np.exp(-1 / (1 - abs((t - t0) /
                                                            (width/2))**2)) *
                                       (abs((t - t0)/(width / 2)) < 1))) /
                        (np.exp(-1)))
    return env

# note : errstate(all='ignore') allows to handle warnings about division by
#      zero that are handle within the function by the ()<1 logical condition


def from_file():  # not implemented yet
    return 0


# %% BASIC FUNCTIONS (PROBABLY NOT FOR DIRECT USE) %% #


def fade(t1, x1, t2, x2, shape='fast'):
    """
    technically half of bump function :
        useful for C-inf ramp between two values

    Parameters
    ----------
    t1 : float
        begin time.
    x1 : float
        begin value.
    t2 : float
        end time.
    x2 : float
        end value.
    shape : string 'fast', 'slow', 'cos'
        rise shape.

    Returns
    -------
    np.array :
        env(t) function, with C-inf ramp from x1 to x2 when t1 < t < t2, else 0.

    """

    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')
    if shape == 'fast':

        def env(t):
            with np.errstate(all='ignore'):  # /0 err handled in code
                return (x1 + (x2 - x1) * ((np.nan_to_num(
                    np.exp(-1 / (1 - abs((t - t2) / (t2 - t1))**2)) *
                        (abs((t - t2) / (t2 - t1)) < 1))) /
                        (np.exp(-1)))) * (t <= t2) * (t > t1)

    elif shape == 'slow':

        def env(t):
            with np.errstate(all='ignore'):  # /0 err handled in code
                return (x2 - (x2 - x1) * ((np.nan_to_num(
                        np.exp(-1 / (1 - abs((t - t1) / (t2 - t1))**2)) *
                        (abs((t - t1) / (t2 - t1)) < 1))) /
                            (np.exp(-1)))) * (t <= t2) * (t > t1)

    elif shape == 'linear' or shape == 'lin' or shape == 'ramp':
        return ramp(t1, x1, t2, x2)

    elif shape == 'cos':  # duplicate of cos(), for use in ADSR
        return cos(t1, x1, t2, x2)

    else:
        warnings.warn('shape of fade should be ''fast'' or ''slow'' ;' +
                      'proceeding with default ''fast'' ')

        def env(t):
            with np.errstate(all='ignore'):  # see note above
                return (x1 + (x2 - x1) * ((np.nan_to_num(
                    np.exp(-1 / (1 - abs((t - t2) / (t2 - t1))**2)) *
                        (abs((t - t2) / (t2 - t1)) < 1))) /
                        (np.exp(-1)))) * (t <= t2) * (t > t1)
    return env


def exp(t1, x1, t2, x2):
    """
    piece of exponential function between two points

    Parameters
    ----------
    t1 : float
        begin time.
    x1 : float
        begin value.
    t2 : float
        end time.
    x2 : float
        end value.

    Returns
    -------
    np.array :
        env(t) function, with exponential when t1 < t < t2, else 0.

    """
    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    def env(t):
        power = (t - t1) / (t2 - t1)
        return (x1 * (x2 / x1)**power) * (t <= t2) * (t > t1)
    return env


def cos(t1, x1, t2, x2):
    """


    Parameters
    ----------
    t1 : float
        begin time.
    x1 : float
        begin value.
    t2 : float
        end time.
    x2 : float
        end value.

    Returns
    -------
    np.array :
        env(t) function, with cosine ramp from x1 to x2 when t1 < t < t2, else 0.

    """
    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    def env(t):
        return ((x1 - x2) / 2 * np.cos(2 * pi * (t - t1) / (2 * (t2 - t1)))
                + np.mean([x1, x2])) * (t <= t2) * (t > t1)
    return env


def tremolo(A, frequency, phase=0):
    """
    sine oscillation of given amplitude, frequency and phase

    Parameters
    ----------
    A : float
        amplitude of oscillation.
    frequency : float
        frequency.
    phase (optionnal) : float
        phase

    Returns
    -------
    np.array :
        env(t) function, oscillation on the whole range of t

    """
    def env(t):
        return A * np.sin(2 * np.pi * frequency * t + phase)

    return env


def constant(t1, t2, C):
    """

    constant value over the range [t1, t2], else 0

    Parameters
    ----------
    t1 : float
        begin time.
    t2 : float
        end time.
    C : float
        Constant value.

    Returns
    -------
    np.array :
        env(t) function, equals C over the range [t1, t2], else 0

    """
    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    def env(t):
        if isinstance(t, float) or isinstance(t, int):
            return C * (t <= t2) * (t > t1)
        else:
            return C * np.ones(t.shape) * (t <= t2) * (t > t1)
    return env


def ramp(t1, x1, t2, x2):
    """

    linear ramp x = a*t + b between two points defined by (t, x)

    Parameters
    ----------
    t1 : float
        begin time.
    x1 : float
        begin value.
    t2 : float
        end time.
    x2 : float
        end value.

    Returns
    -------
    np.array :
        env(t) function returning the ramp when t1 < t < t2, else 0.

    """
    if [t1, t2] != sorted([t1, t2]):
        warnings.warn('time arguments not increasing, ' +
                      ' might cause unexpected results')

    def env(t):
        a = (x2 - x1) / (t2 - t1)
        b = x1 - a*t1

        return (a*t + b) * (t <= t2) * (t > t1)

    return env
