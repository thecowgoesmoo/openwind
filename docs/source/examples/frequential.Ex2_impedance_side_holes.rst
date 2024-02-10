
Ex. 2: Impedance of instrument with side holes
==============================================

How to compute impedances of instrument with side holes and so several fingerings.
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
   temperature = 25

The three files describing the geometry and the

.. code-block:: python

   geom = 'Geom_trumpet.txt'
   holes = 'Geom_holes.txt'
   fing_chart = 'Fingering_chart.txt'

Find file 'trumpet' describing the bore, and compute its impedance

.. code-block:: python

   result = ImpedanceComputation(fs, geom, holes, fing_chart, temperature=temperature)
   result.technical_infos()

Plot the instrument geometry

.. code-block:: python

   result.plot_instrument_geometry()

Plot the impedance

.. code-block:: python

   result.plot_impedance(label='Default Fingering: all open')
   plt.suptitle('Default Fingering: all open')

without indication the impedance computed correspond to the one with all holes open

Chose the fingering
-------------------

It is possible to fix the fingering when the object ``ImpedanceComputation`` is created with the option ``note``

.. code-block:: python

   result_note = ImpedanceComputation(fs, geom, holes, fing_chart,
                                      temperature=temperature, note='A')

   result_note.plot_impedance(label='A')

or to modify it after the instanciation

.. code-block:: python

   fig = plt.figure()
   notes = result_note.get_all_notes()
   for note in notes:
       result_note.set_note(note)
       result_note.plot_impedance(figure=fig, label=note)
