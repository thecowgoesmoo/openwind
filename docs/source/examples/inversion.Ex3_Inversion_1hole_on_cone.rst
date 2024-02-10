
Ex. 3: Inversion 1hole On Cone
==============================

By treating the inversion of on hole on a conical pipe, this example illustrates how to guarantee the side hole to be smaller that the main pipe.

This example uses the :py:class:`InverseFrequentialResponse <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse>` class.

Imports
-------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind.inversion import InverseFrequentialResponse
   from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                         InstrumentPhysics)
   plt.close('all')

Global Options
--------------

.. code-block:: python

   frequencies = np.linspace(50, 500, 100)
   temperature = 20
   losses = True

Targets Definitions
-------------------

For this example we use simulated data
The geometry is 0.5m conical part with 1 side hole.

.. code-block:: python

   geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
   target_hole = [['label', 'position', 'radius', 'chimney'],
                  ['hole1', .25, 3e-3, 5e-3]]
   fingerings = [['label', 'open', 'closed'],
                 ['hole1', 'o', 'x']]
   notes = ['open', 'closed']
   noise_ratio = 0.01
   target_computation = ImpedanceComputation(frequencies, geom, target_hole,
                                             fingerings, note=notes[0],
                                             temperature=temperature,
                                             losses=losses)

The impedance used in target must be normalized

.. code-block:: python

   Zopen = target_computation.impedance/target_computation.Zc

noise is added to simulate measurement

.. code-block:: python

   Zopen *= 1 + noise_ratio*np.random.randn(len(frequencies))
   target_computation.set_note(notes[1])
   Zclosed = target_computation.impedance/target_computation.Zc
   Zclosed *= 1 + noise_ratio*np.random.randn(len(frequencies))
   Ztarget = [Zopen, Zclosed]

Definition Of The Optimized Geometry
------------------------------------

Here we want to adjust only the hole location and radius
During the optimization process we have to guarantee that:


* the hole stays on the main bore (here its location is in [0, 0.5])
* its radius stays smaller than the one of the main pipe at its location
  this can not be guarantee with boundaries!

.. code-block:: python

   inverse_hole = [['label', 'position', 'radius', 'chimney'],
                   ['hole1', '0<~.1<.5', '~2e-3%', 5e-3]]

By using '~2e-3%' the hole radius is defined as a ratio of the main bore
radius at its location. This ratio is in [0,1].
Similar notation can be used to define the location as a ratio of the length
of the main bore pipe for the cases where both the hole location and the pipe
length are optimized.

.. code-block:: python

   instru_geom = InstrumentGeometry(geom, inverse_hole, fingerings)
   print(instru_geom.optim_params)

We can see that the value adjusted by the algorithm for the radius does not
correspond to the geometric value. It is the ratio.
We can compare the two bore at the initial state

.. code-block:: python

   fig_geom = plt.figure()
   target_computation.plot_instrument_geometry(figure=fig_geom, label='Target')
   instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')

The Optimization
----------------

.. code-block:: python

   instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)
   inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget,
                                        notes=notes)

Optimization process

.. code-block:: python

   result = inverse.optimize_freq_model(iter_detailed=True)

and the geometry

.. code-block:: python

   instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Final Geometry')
   print('='*30 + '\nCompare holes geometry')
   print('Target Geometry')
   print(target_computation.get_instrument_geometry().print_holes())
   print('Optimization result:')
   print(instru_geom.print_holes())
