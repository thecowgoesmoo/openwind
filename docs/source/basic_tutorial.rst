
Basic tutorial
==============

This tutorial is recommended as a first introduction
to using wind instrument acoustics toolbox ``openwind``.
It covers the basic use of the main features.
We will:


* import the toolbox,
* build a simplified trumpet,
* compute its input impedance,
* listen to the trumpet coupled to a reed.

At the end of this tutorial, you will be able to:


* enter an instrument's geometry into ``openwind``\ ,
* run a frequency-domain simulation,
* run a time-domain simulation.

Importing openwind
------------------

It is assumed that you have already installed ``openwind`` and its dependencies as described
:doc:`here <installation>`.
In a new python script, we will first import ``openwind``\ , along with ``numpy``
and ``matplotlib``.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   import openwind
   print("OK!")

Run this script ; if Python responds with an error message, `openwind` may not be in your :ref:`PythonPATH <configure-your-pythonpath>`.

Building the instrument
-----------------------

Now that we've made sure ``openwind`` is working properly,
let us build a simplified trumpet.

Description of the geometry
^^^^^^^^^^^^^^^^^^^^^^^^^^^

In openwind, the geometry of an instrument is described by the evolution of air column radius
as one advances through the instrument.
Imagine "unfolding" the trumpet into a straight tube with varying cross-section:
``openwind`` needs to know the inner radius ``r`` for each abscissa ``x``.

Our simplified trumpet will comprise two parts: the lead pipe, and the horn.
The lead pipe will be represented by a cylinder, and the horn by an inverse
power function (also called "Bessel horn").
In the same folder as your script, create a new file named
``'simplified-trumpet.csv'`` containing the following two lines:

.. code-block:: shell

   0.0 0.716 6e-3 6e-3 linear
   0.716 1.335 6e-3 6e-2 bessel 0.7

Each line describes the bore radius on a part of the instrument;
units are in meters:


* between ``x=0`` and ``x=0.716``\ , radius evolves linearly from ``6e-3`` to ``6e-3``\ ; in other words it remains constant, which is what we want for a cylinder.
* between ``x=0.176`` and ``x=1.335``\ , radius evolves from ``6e-3`` to ``6e-2`` following a Bessel horn shape with parameter ``0.7``.

Plotting the instrument
^^^^^^^^^^^^^^^^^^^^^^^

Let us check that this looks like the bore profile of a trumpet. In your script, add:

.. code-block:: python

   instru_geom = openwind.InstrumentGeometry('simplified-trumpet.csv')
   instru_geom.plot_InstrumentGeometry()
   plt.show()

These lines create an :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
object from the geometry file, and plot it.
After running the script, you should now see a graph displaying the bore
profile of your instrument.

If instead Python responds with a ``FileNotFoundError``\ ,
it likely means that the filename is not exactly 'simplified-trumpet.csv',
or that the file is not located in the working directory where Python is
running.

Computing the impedance
-----------------------

Once the instrument has been built, we are able to compute its input impedance
at a given frequency using the
`finite elements method <https://hal.archives-ouvertes.fr/hal-01963674>`_.
The input impedance is a useful indicator of how the instrument responds to periodic
oscillation.

First we need to specify at what frequencies we want to know the impedance.
Let us choose frequencies ranging between 50 Hz and 2 kHz by steps of 1 Hz.

.. code-block:: python

   frequencies = np.arange(50,2000,1)

Then, we can use :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>`
to compute and plot the impedance of the instrument.

.. code-block:: python

   result = openwind.ImpedanceComputation(frequencies, 'simplified-trumpet.csv')
   result.plot_impedance()
   plt.show()

The resulting curve displays several peaks,
corresponding to the resonant frequencies of the simplified trumpet.

Changing the temperature
^^^^^^^^^^^^^^^^^^^^^^^^

Notice that the script printed a warning message:

.. code-block:: shell

   UserWarning: The default temperature is 25 degrees Celsius.

If the experiment was instead conducted in a rather hot room, we could change
the temperature to 30°C using an optional argument of ``ImpedanceComputation``.

.. code-block:: python

   result = openwind.ImpedanceComputation(frequencies, 'simplified-trumpet.csv', temperature=30)

The peaks and dips of the resulting curve are slightly shifted toward higher
frequencies, because the speed of sound has increased.

Exporting the impedance to a file
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The impedance can be written to a file using function :py:meth:`write_impedance <openwind.impedance_tools.write_impedance>`:

.. code-block:: python

   result.write_impedance('impedance.txt')

This should create in the working directory a file ``impedance.txt`` containing
the calculated data:

.. code-block:: shell

   5.000000e+01 4.096864e+05 4.887119e+06
   5.100000e+01 4.323662e+05 5.077191e+06
   5.200000e+01 4.571048e+05 5.277118e+06
   ...

Each line is formatted as
``"(frequency) (real part of impedance) (imaginary part of impedance)"``.
This allows to easily re-import this data later
using :py:meth:`read_impedance <openwind.impedance_tools.read_impedance>`,
or to use it in other software.

"Blowing" the trumpet
---------------------

Time-domain simulations are possible through the use of :py:meth:`simulate <openwind.temporal_simulation>`.

Running the simulation
^^^^^^^^^^^^^^^^^^^^^^

First need to tell what it is we put at the end of the instrument: is it a
woodwind-reed, a pair of lips, an idealized flow source?
This is what the :py:class:`Player <openwind.technical.player.Player>` object is for. Here we want to use a
woodwind-reed model, so we build the ``Player`` from a default
set of parameters called ``"TUTORIAL_REED"``.

.. code-block:: python

   my_player = openwind.Player("TUTORIAL_REED")

Using this ``Player``, we can launch a very short simulation of 0.5 seconds.

.. code-block:: python

   simulation = openwind.simulate(0.5, 'simplified-trumpet.csv', player=my_player)

This line takes a few seconds to run, and then returns an object that contains
the simulation data. To know which data are available, run

.. code-block:: python

   print(simulation)

Plotting the waveform
^^^^^^^^^^^^^^^^^^^^^

The sound that can be heard from the outside is the radiated pressure,
so let us plot the evolution of ``'bell_radiation_pressure'``
as a function of time. Time information is contained in ``simulation.ts``\ ,
the array of the times at which data was sampled.

.. code-block:: python

   plt.figure()  # opens a new figure
   plt.plot(simulation.ts, simulation.values['bell_radiation_pressure'])
   plt.show()

Exporting as an audio file
^^^^^^^^^^^^^^^^^^^^^^^^^^

To listen to the corresponding audio, a convenience method is :py:meth:`export_mono <openwind.temporal.utils.export_mono>`, which resamples and scales the data,
and exports it to a ``.wav`` file.
Let us listen to the waveform of the bell pressure:

.. code-block:: python

   from openwind.temporal.utils import export_mono
   export_mono('my_trumpet_reed_simulation.wav', simulation.values['bell_radiation_pressure'], simulation.ts)

More information
----------------


* The syntax of the geometry file is described in the documentation for :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`.
* More options for impedance computation can be found in the documentation for :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>`.
* Options for time-domain simulation can be found in the documentation for :py:meth:`simulate <openwind.temporal_simulation.simulate>`.
* The following section lists a number of use-case examples of the various features of the toolbox.
