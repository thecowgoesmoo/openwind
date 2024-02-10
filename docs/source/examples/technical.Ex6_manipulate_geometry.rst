
Ex. 6: Manipulate Geometry
==========================

This example illustrates how to manipulate geometries with openwind. More precisely
how to shift x-axis, concatenate several geometries or slice one geometry.

This example uses the `InstrumentGeometry <../modules/openwind.instrument_geometry>`_ class.

Imports
-------

.. code-block:: python

   This example illustrates how to manipulate geometries with openwind. More precisely
   how to shift x-axis, concatenate several geometries or slice one geometry.
   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import InstrumentGeometry

Shift The X-Axis
----------------

we build am instrument with side holes

.. code-block:: python

   main_bore = [[-5e-2, 0, 3e-3, 7e-3, 'cone'],
                [0, 7e-2, 8e-3, 8e-3, 'spline', 4e-2, 6e-2, 13e-3, 11e-3],
                [7e-2, 50e-2, 7e-3, 15e-3, 'cone'],
                [50e-2, 70e-2, 15e-3, 110e-3, 'bessel', 0.3]
               ]
   holes = [['label', 'x', 'r', 'l'],
            ['thumb', 10e-2, 5e-3, 12e-3],
            ['indexL', 11e-2, 3e-3, 4e-3],
            ['middleL', 19e-2, 4e-3, 4e-3],
            ['ringL', 26e-2, 5e-3, 4e-3],
            ['indexR', 35e-2, 6e-3, 4e-3],
            ['middleR', 45e-2, 7e-3, 4e-3],
            ['ringR', 56e-2, 8e-3, 4e-3]
            ]
   my_clar = InstrumentGeometry(main_bore, holes)
   fig = plt.figure()
   my_clar.plot_InstrumentGeometry(figure=fig, color='k')
   plt.suptitle('My clarinet')

Even if it does not change anything for the acoustic computation, it can be
usefull to shift the x-axis, (for example to print a more readable file).
It can been done by the following method

.. code-block:: python

   my_clar.shift_x_axis(5e-2)
   print('\nMY SHIFTED INSTRUMENT\n{}'.format(my_clar))

We can see that all the x-related parameters (the main bore and the locations
of the holes) have been shifted by 5cm.
Let's come back to the previous situation:

.. code-block:: python

   my_clar.shift_x_axis(-5e-2)

Slicing
-------

The method ``extract(start, stop)`` allows cutting a part of an instrument (to remove a
connecting sleeve for example).
The slicing can be done at the exact junction of two parts:

.. code-block:: python

   mouth_piece = my_clar.extract(-np.inf, 0)
   print('*'*30 + '\n' + '*'*30)
   print('\nA PART\n{}'.format(mouth_piece))
   barrel = my_clar.extract(0, 7e-2)

Or anywhere on a part:

.. code-block:: python

   upper_joint = my_clar.extract(7e-2, 22.5e-2)
   lower_joint = my_clar.extract(22.5e-2, 65e-2)
   bell = my_clar.extract(65e-2, np.inf)
   mouth_piece.plot_InstrumentGeometry(figure=fig, color='r', linestyle='--')
   barrel.plot_InstrumentGeometry(figure=fig, color='b', linestyle=':')
   upper_joint.plot_InstrumentGeometry(figure=fig, color='g', linestyle='--')
   lower_joint.plot_InstrumentGeometry(figure=fig, color='y', linestyle='--')
   bell.plot_InstrumentGeometry(figure=fig, color='r', linestyle='--')
   plt.xlim((-60, 750))

Concatenation
-------------


lets now reassemble our clarinet but with a new shorter cylindrical barel.
The concatenation is simply obtained by summing the ``InstrumentGeometry`` in
the right order. It can be done with "+" or by using "sum" with a list of
``InstrumentGeometry``.

..important::
  It is not necessary to shift the x-axis before suming. It is performed
  automaticelly to fit the begining of an element to the end of the previous one.

.. code-block:: python

   new_barrel = InstrumentGeometry([[0, 3e-2, 8e-3, 8e-3, 'cone']])	
   new_clar = mouth_piece + new_barrel + upper_joint + lower_joint + bell
   list_instru = [mouth_piece, new_barrel, upper_joint, lower_joint, bell]
   new_clar_bis = sum(list_instru)
   new_clar.plot_InstrumentGeometry(color='k')
   plt.suptitle('My new clarinet')
   plt.xlim((-60, 750))
