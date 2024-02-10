
Ex. 9: Low Level Implementation And Ac Fields
=============================================

This presents low level implementation giving access to the acoustic fields
in the entire instrument.
It presents also how to interpolate data to a specific grid.

This example uses the `(ImpedanceComputation, InstrumentGeometry, Player, <../modules/openwind.impedance_computation, _instrument_geometry, _player,>`_ class.

Imports
-------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                         InstrumentPhysics, FrequentialSolver)
   fs = np.arange(20, 2000, 1) # frequencies of interest: 20Hz to 2kHz by steps of 1Hz

Low Level Implementation
------------------------

Load and process the instrument geometrical file

.. code-block:: python

   instr_geom = InstrumentGeometry('Oboe_instrument.txt','Oboe_holes.txt')

Create a player using the default value : unitary flow for impedance computation

.. code-block:: python

   player = Player()

Choose the physics of the instrument from its geometry. Default models are chosen when they are not specified.
Here losses = True means that Zwikker-Koster model is solved.

.. code-block:: python

   instr_physics = InstrumentPhysics(instr_geom, temperature=25, player = player, losses=True)

Perform the discretisation of the pipes and put all parts together ready to be solved.

.. code-block:: python

   freq_model = FrequentialSolver(instr_physics, fs)

Visualization
-------------

Solve the linear system underlying the impedance computation.
interp_grid allows to interpolate the data on a uniform grid with a given spacing.

.. code-block:: python

   freq_model.solve(interp=True, interp_grid=0.01)

You can observe at the flow and pressure for all interpolated points at one
given frequency, e.g. 500Hz and 1500Hz:

.. code-block:: python

   plt.figure()
   freq_model.plot_flow_at_freq(500, label='Flow: 500 Hz')
   freq_model.plot_flow_at_freq(1500, label='Flow: 1500 Hz')
   plt.figure()
   freq_model.plot_pressure_at_freq(500, label='Pressure: 500 Hz')
   freq_model.plot_pressure_at_freq(1500, label='Pressure: 1500 Hz')

You can also display these ac. fields for all frequencies.

.. code-block:: python

   freq_model.plot_var3D(var='pressure')
   plt.title('Main Bore')
   freq_model.plot_var3D(var='flow')
   plt.title('Main Bore')

by default, only the main bore pipes are observed and plotted.

.. code-block:: python

   freq_model.solve(interp=True, interp_grid=0.01, pipes_label='main_bore')

you can observe also a given hole by giving the right pipe label obtained in

.. code-block:: python

   print(instr_physics.netlist.pipes.keys())
   freq_model.solve(interp=True, interp_grid=0.001, pipes_label='hole1')
   freq_model.plot_var3D(var='pressure')
   plt.title('Hole 1')

or a series of pipe labels

.. code-block:: python

   freq_model.solve(interp=True, interp_grid=0.001,
                        pipes_label=['bore0', 'bore1', 'bore2_slice0', 'bore2_slice1'])
   freq_model.plot_var3D(var='pressure')
   plt.title('Pipes 0 to 2')
   plt.show()

With Plotly
-----------

if you have plotly install on you computer you can display this plot in 3D

.. code-block:: python

   freq_model.plot_var3D(var='pressure', with_plotly=True)
