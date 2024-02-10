
Ex. 4 : Temporal Curves
=======================

.. code-block:: python

   from openwind.technical.temporal_curves import gate, ADSR, fade
   import matplotlib.pyplot as plt
   import numpy as np

For temporal simulations, one might want to change the simulation parameters with respect to time. For instance, the blowing pressure generally starts at 0, rises to a certain constant value, and then fades back to 0 to avoid any discontinuities.

The :py:mod:`temporal curves <openwind.technical.temporal_curves>` module offers two main controller types, Gate and ADSR. All the functions in the module need to call a certain time, generally a vector with time values. 

.. code-block:: python

   time_vector = np.arange(0, 1, 0.01)
   my_control = some_function(some_arguments)  # this is a function
   my_values = my_control(t)  # this is a vector

Gate(t)
-------

The gate function can act as a basic ON/OFF controller, but has multiple options for more advanced shapes.

.. code-block:: python

   my_gate = gate(t1, t2, t3, t4, shape='linear', a=1)

The ``shape`` can be set to ``'linear'``\ , ``'fast'``\ , ``'slow'``\ , or ``'cos'``.

``linear`` is a simple straight line, the other three are smooth functions.

ADSR(t)
-------

The ADSR function is built to work like the Attack-Decay-Sustain-Release function widely used in modular sound synthesis. It resembles the ``gate`` function but has more 'musical' options.

Use first two arguments to define the start and end times of your envelope, then come the four ADSR parameters. The shape can be a list, in which case its elements refer to the different fades in the envelope.

.. code-block:: python

   plt.plot(t, ADSR(0.2, 4.5, 2.5, 0.5, 1, 0.85, 1, shape=['fast', 'slow', 'cos'])(t))
   plt.title('ADSR with shape = [''fast'', ''slow'', ''cos'']')

For extra-fancy simulations, add a tremolo to your envelope ! The tremolo option is defined by its amplitude *relative to the peak amplitude of the ADSR*\ , and its frequency. The effect can also be given its own envelope, in which case the times are defined from 0 to 1 on the 'sustain' interval.

.. code-block:: python

   my_tremolo = ADSR(0.2, 4.5, 2.5, 0.5, 1, 0.85, 1,
                    shape=['fast', 'slow', 'cos'],
                    trem_a=0.05, trem_freq=2,
                    trem_gate=gate(0, 0.75, .9, 1, shape='slow'))
