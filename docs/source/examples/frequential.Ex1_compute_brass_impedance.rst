
Ex. 1:  Compute brass impedance
===============================

How to easily compute impedance of instrument without tone holes.

This example uses the :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>` class.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import ImpedanceComputation

Basic computation
-----------------

Frequencies of interest: 20Hz to 2kHz by steps of 1Hz

.. code-block:: python

   fs = np.arange(20, 2000, 1)

Find file 'trumpet' describing the bore, and compute its impedance

.. code-block:: python

   geom_filename = 'Geom_trumpet.txt'
   result = ImpedanceComputation(fs, geom_filename)

Plot the instrument geometry

.. code-block:: python

   result.plot_instrument_geometry()

you can get the characteristic impedance at the entrance of the instrument which can be useful to normalize the impedance

.. code-block:: python

   Zc = result.Zc

You can plot the impedance which is automatically normalized

.. code-block:: python

   fig = plt.figure()
   result.plot_impedance(figure=fig, label='my label')

here the option 'figure' specifies on which window plot the impedance (you can use any matplotlib keyword!)

Other useful features
---------------------

You can modify the frequency axis without redoing everything

.. code-block:: python

   freq_bis = np.arange(20, 2000, 100)
   result.recompute_impedance_at(freq_bis)
   result.plot_impedance(figure=fig, label='few frequencies', marker='o', linestyle='')

You can print the computed impedance in a file (it is automatically normalized by Zc).

.. code-block:: python

   result.write_impedance('computed_impedance.txt')
