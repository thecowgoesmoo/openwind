
Ex. 8: Conical Holes and Valves
===============================

This example illustrates how to specify conical tone holes or conical deviation
pipes for the valves.

This example uses the `InstrumentGeometry <../modules/openwind.instrument_geometry>`_ class.

Conical Holes
-------------

To use conical side- holes in openwind, simply add a column labeled "radius_out"
in the hole file (or hole list). This column will correspond to the radius of
the holes at the external wall of the instrument and the column "radius" to the
radius inside the instrument. If you add this column, for cylindrical hole you
must indicate twice the same value.

Let's try for a simple instrument with 3 holes, 2 conical with inverse conicity
and one cylindrical:

.. code-block:: python

   main_bore = [[0, 0.25, 5e-3, 7e-3, 'linear']]
   holes = [['label',     'position', 'length',    'radius',   'radius_out'],
            ['hole_cone', .1,          5e-3,       4e-3,       2e-3],
            ['hole_cyl', .15,          5e-3,       4e-3,       4e-3],
            ['hole_inv_cone', .17,          5e-3,       2e-3,       4e-3],
            ]
   my_instru_with_holes = InstrumentGeometry(main_bore, holes)
   print(my_instru_with_holes)
   my_instru_with_holes.plot_InstrumentGeometry()
   plt.show()

Conical Valves
--------------

With the valves it is possible to specify conical deviation pipe similarly.
This time, the "radius_out" column correspond to the radius at the reconnection
point the farthest to the entrance. Here again, let's try with a simple
instrument with 3 valves: two with conical deviation pipe and one with a cylindrical pipe.

.. code-block:: python

   valves = [['label',         'variety',  'position', 'length',   'radius',   'radius_out',   'reconnection'],
             ['piston_cone',   'valve',    0.07,       0.05,       4e-3,       2e-3,           0.08],
             ['piston_cyl',    'valve',    0.11,       0.05,       4e-3,       4e-3,           0.12],
             ['piston_cone2',  'valve',    0.15,       0.05,       2e-3,       4e-3,           0.16],
             ]
   my_instru_with_valves = InstrumentGeometry(main_bore, valves)
   print(my_instru_with_valves)
   my_instru_with_valves.plot_InstrumentGeometry()
   plt.show()
