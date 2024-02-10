
Ex. 7: Ensure Radius Continuity
===============================

This example present how to ensure to keep the radius continuity of the main bore
during an inversion, or inversely to give the possibility to have a radius jump.

This example uses the `InverseFrequentialResponse <../modules/openwind.inversion.inverse_frequential_response>`_ class.

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

   frequencies = np.linspace(100, 500, 10)
   temperature = 20
   losses = True

Targets Definitions
-------------------

For this example we use simulated data instead of measurement
The geometry is composed of 2 cones, with a discontinuity of radius at the junction

.. code-block:: python

   target_geom = [[0, 0.25, 2e-3, 3e-3, 'linear'],
                  [0.25, .5, 3.2e-3, 7e-3, 'linear']]
   target_computation = ImpedanceComputation(frequencies, target_geom,
                                             temperature=temperature,
                                             losses=losses)

The impedance used in target must be normalized

.. code-block:: python

   Ztarget = target_computation.impedance/target_computation.Zc

noise is added to simulate measurement

.. code-block:: python

   noise_ratio = 0.01
   Ztarget = Ztarget*(1 + noise_ratio*np.random.randn(len(Ztarget)))

Ensure Radius Continuity
------------------------

We would like to find the geometry without discontinuity of section which fit
the target impedance

.. code-block:: python

   geom_continuous = [[0, 0.25, 2e-3, '~4e-3', 'linear'],
                  [0.25, .5, '~4e-3', 7e-3, 'linear']]
   instru_geom_con = InstrumentGeometry(geom_continuous)
   print(instru_geom_con.optim_params)

Here the exact same initial value is indicated for the right end radius of the first
cone and the left end radius of the second pipe.
You can notice that they are treated as a unique design parameter
this Ensure that during the optimization these two raddi will be always equal
Inverse problem

.. code-block:: python

   con_phy = InstrumentPhysics(instru_geom_con, temperature, Player(), losses)
   inverse_con = InverseFrequentialResponse(con_phy, frequencies, Ztarget)
   result_con = inverse_con.optimize_freq_model(iter_detailed=True)
   print(instru_geom_con.optim_params)

The optimization process stops at a value in between the two 3mm and 3.2mm

Give The Possibility To Have A Discontinuity
--------------------------------------------

If we want to authorize the discontinuity, the two initial value must different,
of at least 0.001% (1e-5) or 1e-5mm

.. code-block:: python

   geom_disccontinuous = [[0, 0.25, 2e-3, '~4e-3', 'linear'],
                          [0.25, .5, '~4.0001e-3', 7e-3, 'linear']]
   instru_geom_disc = InstrumentGeometry(geom_disccontinuous)
   print(instru_geom_disc.optim_params)

This time two different design variables are isntanciated
Inverse problem

.. code-block:: python

   disc_phy = InstrumentPhysics(instru_geom_disc, temperature, Player(), losses)
   inverse_disc = InverseFrequentialResponse(disc_phy, frequencies, Ztarget)
   result_disc = inverse_disc.optimize_freq_model(iter_detailed=True)
   print(instru_geom_disc.optim_params)

Now the optimization process converge to 3mm and 3.2mm
