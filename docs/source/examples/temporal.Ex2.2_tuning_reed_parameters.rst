
Ex. 2.2: Tuning Reed Parameters
===============================

This file illustrates how to update the controle parameters for a time
domain simulation, and how to use the low-level of the time domain

Source file available `here <https://gitlab.inria.fr/openwind/openwind/-/blob/master/examples/temporal/Ex2.2_tuning_reed_parameters.py>`_.

This example uses the classes:

- `InstrumentGeometry <../modules/openwind.instrument_geometry>`_,
- `InstrumentPhysics <../modules/openwind.instrument_physics>`_,
- `TemporalSolver <../modules/openwind.temporal_solver>`_,
- `Player <../modules/openwind.player>`_.

Imports
-------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import InstrumentGeometry, InstrumentPhysics, TemporalSolver, Player
   from openwind.temporal import RecordingDevice
   from openwind.technical.temporal_curves import ADSR

"Low Level" Implementation Of Scaled Reed Model
-----------------------------------------------

This time we will use the scaled models and so the dimensionless parameters.
We build a dictionnary with the intersting fields then instanciate the ``Player``.
This is also possible to use the default dict given in :py:mod:`default_excitator_parameters<openwind.technical.default_excitator_parameters>`.

.. code-block:: python

   gamma_amp = 0.45 # the amplitude of gamma, the dimensionless supply pressure
   transition_time = 2e-2 # the characteristic time of the time eveolution of gamma
   gamma_time = ADSR(0, 0.4, gamma_amp, transition_time, transition_time, 1, transition_time) # the time evolution of gamma
   zeta = 0.35 # the value of zeta, the "reed" opening dimensionless paramters
   dimless_reed = {"excitator_type" : "Reed1dof_scaled",
                   "gamma" : gamma_time,
                   "zeta": zeta,
                   "kappa": 0.35,
                   "pulsation" : 2*np.pi*2700, #in rad/s
                   "qfactor": 6,
                   "model" : "inwards",
                   "contact_stifness": 1e4,
                   "contact_exponent": 4,
                   "opening" : 5e-4, #in m
                   "closing_pressure": 5e3 #in Pa
                   }
   reed_player = Player(dimless_reed)

We instanciate the other objects necessary to compute the sound

.. code-block:: python

   instrument = [[0.0, 5e-3],
                 [0.5, 5e-3]]
   my_geom = InstrumentGeometry(instrument) # the geometry of the instrument
   temperature = 25
   my_phy = InstrumentPhysics(my_geom, temperature, reed_player, 'diffrepr')
   my_temp_solver = TemporalSolver(my_phy)
   rec = RecordingDevice()

we can now compute the sound for the indicated control parameters

.. code-block:: python

   my_temp_solver.run_simulation(0.5, callback=rec.callback)

we extract and plot the reed displacement

.. code-block:: python

   y_reed = rec.values['source_y']
   time = rec.ts
   plt.figure()
   plt.plot(time, y_reed, label=f'zeta={zeta}')
   plt.legend()
   plt.grid()
   plt.xlabel('Time [s]')
   plt.ylabel('Reed displacement [m]')

Change The Value Of Constant Control Parameters
-----------------------------------------------

We can recompute the sound for different control parameters without redoing everything
We first modify the value of the control parameters in the Player bject then only restart the time simulation.

.. code-block:: python

   zeta_list = [0.3, 0.4, 0.5]
   for zeta in zeta_list:
       reed_player.update_curve('zeta', zeta)
       rec = RecordingDevice()
       my_temp_solver.reset()
       my_temp_solver.run_simulation(0.5, callback=rec.callback)
       y_reed = rec.values['source_y']
       time = rec.ts
       plt.plot(time, y_reed, label=f'zeta={zeta}')
   plt.legend()

Change The Value Of Time Varying Control Parameters
---------------------------------------------------

`Gamma` , the dimensionless parameters linked to the supply pressure is the only one parameters which can vary with thime.
It is necessary to reinstanciate a new time varying function

.. code-block:: python

   gamma_amp_list =  [0.3, 0.4, 0.5]
   plt.figure()
   plt.grid()
   plt.xlabel('Time [s]')
   plt.ylabel('Reed displacement [m]')
   for gamma_amp in gamma_amp_list:
       gamma_time = ADSR(0, 0.4, gamma_amp, transition_time, transition_time, 1, transition_time)
       reed_player.update_curve('gamma', gamma_time)
       rec = RecordingDevice()
       my_temp_solver.reset()
       my_temp_solver.run_simulation(0.5, callback=rec.callback)
       y_reed = rec.values['source_y']
       time = rec.ts
       plt.plot(time, y_reed, label=f'gamma={gamma_amp}')
   plt.legend()
   plt.show()
