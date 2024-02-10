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
This examples shows how to define parameter controllers using temporal
functions.
"""

import matplotlib.pyplot as plt
import numpy as np

from openwind.technical.temporal_curves import gate, ADSR, fade
import openwind.technical.temporal_curves as tc

# For temporal simulations, one might want to change the simulation parameters
# with respect to time. For instance, the blowing pressure starts generally
# at 0, rises to a certain constant value, and then fades back to 0 to avoid
# any discontinuities


# Say we have a temporal simulation running from t=0 to t=5 second, with a
# sampling period of T = 0.01.
t = np.arange(0, 5, 0.01)

# The Temporal Curves module offers two main controller types : Gate and ADSR

# -----------------------------------------------------------------------------
# --- GATE(t) ---


# the GATE function acts as a basic on-off controller, but has multiple options
# for more advanced shapes :


# basic ON-OFF on the interval t = [2, 4]
# here the start and end times for both ramps are the same
plt.figure(1)
plt.plot(t, gate(2, 2, 4, 4)(t))
plt.ylim([0, 1.1])
plt.title('simple ON/OFF')


# if the time of the decreasing ramp is greater than the end time of the
# simulation, we obtain a constant value with rising ramp
# the additional "a" parameter gives the maximum amplitude ; note that
# gate(2, 2, 4, 4, a=2) is equivalent to 2 * gate(2, 2, 4, 4, a=1)
plt.figure(2)
plt.plot(t, gate(1, 2, 6, 6, a=1)(t))
plt.plot(t, gate(1, 2, 6, 6, a=2)(t))
plt.plot(t, 4*gate(1, 2, 6, 6)(t))
plt.ylim([0, 4.4])
plt.title('constant with initial ramp, different maximum values')


# the shape option of the ramps offers different results :
# note that the 'fast' and 'slow' functions are based on exponentials and are
# C-infinite
plt.figure(3)
plt.plot(t, gate(0, 2, 3, 5, shape='fast')(t), label='fast')
plt.plot(t, gate(0, 2, 3, 5, shape='slow')(t), label='slow')
plt.plot(t, gate(0, 2, 3, 5, shape='cos')(t), label='cos')
plt.plot(t, gate(0, 2, 3, 5, shape='linear')(t), label='linear')
plt.ylim([0, 1.1])
plt.legend()


# the shape can be a list to combine different shapes
plt.figure(4)
plt.plot(t, gate(1, 2, 3, 4, shape=['lin', 'fast'])(t))
plt.title('linear rise, fast fall')
plt.ylim([0, 1.1])

plt.show()
# if the end of the first ramp and the start of the second are the same, you
# obtain 'bump'-like function
plt.plot(5)
plt.plot(t, gate(1, 2.5, 2.5, 4, shape='linear')(t), label='shape=[lin, lin]')
plt.plot(t, gate(1, 2.5, 2.5, 4, shape=['fast', 'slow'])(t), label='shape=[fast, slow]')
plt.plot(t, gate(1, 2.5, 2.5, 4, shape=['slow', 'fast'])(t), label='shape=[slow, fast]')
plt.title('bump-like functions')
plt.ylim([0, 1.1])
plt.legend()


# -----------------------------------------------------------------------------
# --- ADSR(t)

# The ADSR function is built to work like the Attack-Decay-Sustain-Release
# function widely used in additive sound synthesis.

# The resulting function resembles the GATE behaviour but has more 'musical'
# options.

# ADSR envelope over the inverval [0.2, 4.5], with peak amplitude 2.5.
plt.figure(6)
plt.plot(t, ADSR(0.2, 4.5, 2.5, 0.5, 1, 0.85, 1, shape='lin')(t))
plt.title('Attack = 0.5, Decay = 1, Sustain = 0.85, Release = 1, shape = linear')

# Again, the shapes offer different results :
plt.figure(7)
plt.plot(t, ADSR(0.2, 4.5, 2.5, 0.5, 1, 0.85, 1, shape=['fast', 'slow', 'cos'])(t))
plt.title('ADSR with shape = [''fast'', ''slow'', ''cos'']')

# ADSR also offers an option for tremolo on the sustain part. This is done by
# replacing the constant sustain by a sine function, that can be gated in
# different advanced ways (please refer to the documentation of the ADSR
# function for more details)
plt.figure(8)
plt.plot(t, ADSR(0.2, 4.5, 2.5, 0.5, 1, 0.85, 1,
                 shape=['fast', 'slow', 'cos'],
                 trem_a=0.05, trem_freq=2,
                 trem_gate=gate(0, 0.75, .9, 1, shape='slow'))(t), label='ADSR')

# calculations to represent the envelope of tremolo
plt.plot(t, 0.85*2.5 + gate(0.2+0.5+1,
                            0.2+0.5+1 + 0.8*((4.5-1) - (0.2+0.5+1)),
                            0.2+0.5+1 + 0.9*((4.5-1) - (0.2+0.5+1)),
                            4.5-1,
                            shape='slow',
                            a=0.05*2.5)(t),
         linestyle='--', label='envelope of the tremolo')
plt.legend()
plt.title('ADSR with tremolo')


# -----------------------------------------------------------------------------
# --- ADVANCED 'TRICKS' : OPERATIONS

# The temporal curves module contains a large variety of 'building blocks'
# functions, that will no be discussed in this example file. Please refer to
# the specific documentation for more details.
# It is however useful to know that the functions may be combined in
# various ways using operators :


# Multiply a GATE with a linear fade :

my_gate = gate(0.5, 1.5, 3.5, 4.5)
nice_fade = fade(0, 1, 5, 0.3, shape='lin')
new_curve = tc.multiply(my_gate, nice_fade)

plt.figure(9)
plt.plot(t, new_curve(t))
plt.plot(t, my_gate(t))
plt.plot(t, nice_fade(t), linestyle='--')
plt.legend(['multiplication', 'gate', 'fade'])
plt.title('A GATE function multiplied by a linear fade')

# Add different envelopes together

gate1 = gate(0, 0.5, 1, 1.5)
gate2 = gate(2, 2, 3, 5)
gate3 = gate(2.2, 4, 4.5, 5)
sum_of_gates = tc.add( tc.add(gate1, gate2), gate3 )

plt.figure(10)
plt.plot(t, sum_of_gates(t))
plt.plot(t, gate1(t), linestyle='--')
plt.plot(t, gate2(t), linestyle='--')
plt.plot(t, gate3(t), linestyle='--')
plt.legend(['sum', 'gate1', 'gate2', 'gate3'])
plt.title('Three GATE functions added together')


# Have fun !!

# -----------------------------------------------------------------------------
