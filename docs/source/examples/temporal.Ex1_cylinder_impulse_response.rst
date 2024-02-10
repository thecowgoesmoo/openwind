
Ex. 1 : Impulse response of a cylinder
======================================

How to easily simulate acoustics in a 20cm cylinder of radius 5mm

This example uses the :py:meth:`simulate <openwind.temporal_simulation.simulate>` method
and the :py:class:`Player <openwind.technical.player.Player>` class.

Describe the instrument
-----------------------

.. code-block:: python

   instrument = [[0.0, 5e-3],
                 [0.2, 5e-3]]

The input signal is a flow impulse at the entrance of the tube.
The impulse lasts 400µs.

.. code-block:: python

   player = Player('IMPULSE_400us')

.. code-block:: python

   duration = 0.2  # simulation time in seconds

Run the simulation
------------------

The user must provide the duration, the instrument (and optionnaly its holes
and fingering chart).
the user can specify some other paramteres as :
the player (default is an impulse flow)
the type of pipe visco-thermal losses (default is none)
the temperature inside the instrument (default is 25°C)
the radiation condition at the bell and open holes (default is 'unflanged')
the spatial discretization
and many more, see openwind.temporal_simulation.py

.. code-block:: python

   rec , _ = simulate(duration,
                  instrument,
                  # use the pre-instanciated player
                  # (if not given, the source is an impulse flow)
                  player=player,
                  # Use diffusive representation of boundary layer losses
                  # (if not given, default will be False)
                  losses='diffrepr',
                  # Assume a temperature of 20°C
                  temperature=20,
                  # Finite elements discretization parameters
                  # (if not given, a default discretization will be made)
                  l_ele=0.01, order=4
                  )

Export the signal that is radiated at the exit
----------------------------------------------

.. code-block:: python

   signal = rec.values['bell_radiation_pressure']
   export_mono('Ex1_cylinder_impulse_response.wav', signal, rec.ts)
