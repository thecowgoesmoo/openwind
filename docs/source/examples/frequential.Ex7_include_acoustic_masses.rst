
Ex. 7: Include asoustic masses
==============================

Present the option relative to inclusion or not of acoustic masses
(discontinuity and matching volume).
This example uses the :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>` class.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import ImpedanceComputation, InstrumentGeometry

   fs = np.arange(20, 2000, 1)
   temp = 25

Mass due to cross section discontinuity
---------------------------------------

We chose an instrument with a cross section discontinuity:

.. code-block:: python

   geom = 'Oboe_instrument.txt'
   holes = 'Oboe_holes.txt'
   fing_chart = 'Oboe_fingering_chart.txt'

   instru_geom = InstrumentGeometry(geom, holes)
   instru_geom.plot_InstrumentGeometry()

There is a discontinuity at 0.45m before the "bell". It is possible to chose to include or not the supplementary acoustic mass  due to this discontinuity (by default it is included)

.. code-block:: python

   result_with_masses = ImpedanceComputation(fs, geom, holes, fing_chart,
                                             note='C', temperature=temp,
                                             discontinuity_mass=True)

   fig = plt.figure()
   result_with_masses.plot_impedance(figure=fig, label='with discontinuity mass')

or to exclude it

.. code-block:: python

   result_wo_masses = ImpedanceComputation(fs, geom, holes, fing_chart,
                                           note='C',temperature=temp,
                                           discontinuity_mass=False)
   result_wo_masses.plot_impedance(figure=fig, label='without discontinuity mass')

Matching Volume
---------------

It is possible to include the masses due to the matching volume between the circular pipe of  the main bore and the circular pipe of the side hole by default these masses are excluded, it can be including throug the keyword ``matching_volume``

.. code-block:: python

   result_with_matching_volume = ImpedanceComputation(fs, geom, holes, fing_chart,
                                                    note='C', temperature=temp,
                                                    matching_volume=True)
   result_with_matching_volume.plot_impedance(figure=fig, label='with matching volume')
