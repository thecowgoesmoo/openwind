
Ex. 2: Simulation of a reed simple instrument with one side hole
================================================================

How to simulate the coupling of a reed oscillator with a 50cm cylinder

This example uses the :py:meth:`simulate <openwind.temporal_simulation.simulate>` method
and the :py:class:`Player <openwind.technical.player.Player>` class.

.. code-block:: python

   from openwind import simulate
   from openwind import Player
   from openwind.temporal.utils import export_mono

Describe the geometry
---------------------

.. code-block:: python

   instrument = [[0.0, 5e-3],
                 [0.5, 5e-3]]

One small hole positioned at 45cm
1cm long, 2mm of radius, radiating by default.

.. code-block:: python

   holes = [['x', 'l', 'r', 'label'],
            [0.45, 0.01, 2e-3, 'hole1']]

   player = Player('CLARINET')

Tune the parameters
-------------------

Parameters of the reed can be changed manually
Available parameters are:
"opening", "mass", "section", "pulsation", "dissip", "width",
"mouth_pressure", "model", "contact_pulsation", "contact_exponent"

.. code-block:: python

   player.update_curve("width", 2e-2)


   duration = 0.2  # simulation time in seconds
   rec = simulate(duration,
                  instrument,
                  holes,
                  player = player,
                  losses='diffrepr',
                  temperature=20,
                  l_ele=0.01, order=4 # Discretization parameters
                  )

Show the discretization infos and export the simulated data
-----------------------------------------------------------

.. code-block:: python

   rec.t_solver.discretization_infos()

   signal = rec.values['bell_radiation_pressure']
   export_mono('Ex2_cylinder_reed.wav', signal, rec.ts)
