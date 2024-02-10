
Ex. 7: Use Different Units
==========================

This example illustrates how to use different units (meter, millimeter) or use
dimater instead of radius in geometric data.

This example uses the `InstrumentGeometry <../modules/openwind.instrument_geometry>`_ class.

Imports
-------

.. code-block:: python

   This example illustrates how to use different units (meter, millimeter) or use
   dimater instead of radius in geometric data.
   import matplotlib.pyplot as plt
   from openwind import InstrumentGeometry

Using List
----------

The :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
has two options giving possibility to indicate the geometry with different conventions


* ``unit`` allowing to use meter or millimeter with keyword "m" or "mm" (default 'm')
* ``diameter`` allowing to use diameter instead of radius with keyword "True" (default 'False')

Let try with the instrument from 'Ex1_instrument_3.csv' rewritten here in mm and with diameter:

.. code-block:: python

   geom_mm_diam = [[0.0,   0.9,      17.4,    9.2,      'Circle',   -10],
                   [0.9,      1.4,    9.2,    4.8,      'Circle',     7],
                   [1.4,     10,      4.8,    6,          'Cone'],
                   [10,      30,      8.4,   10,          'Cone'],
                   [30,     100,     10,     10,        'Spline',    40,     70,     12,      8],
                   [100,    120,     10,     20,   'Exponential'],
                   [120,    140,     20,    100,        'Bessel',     0.8]]
   my_instru = InstrumentGeometry(geom_mm_diam, unit='mm', diameter=True)

You can observe that the radius of curvature of the "Circle" shape is converted in mm but stays a "Radius"
and the coefficient of the Bessel Horn is unchanged because it is without dimension.

We can check that the instrument has the correct profile and the correct length (given in meter)

.. code-block:: python

   my_instru.plot_InstrumentGeometry()
   print('Total length: %.3f [m]' %my_instru.get_main_bore_length())

The hole or valve list must use the same convention

.. code-block:: python

   valve_mm_diameter = [['label',  'variety',  'position', 'reconnection', 'length',   'diameter'],
                        ['piston', 'valve',    15,         16,             55,         5]]
   my_instru_with_valve = InstrumentGeometry(geom_mm_diam, valve_mm_diameter,
                                             unit='mm', diameter=True)
   my_instru_with_valve.plot_InstrumentGeometry()
   plt.show()

Using External File
-------------------

These options can also be used with external files. In that case they must be
indicated in head-lines starting with "!" as in files ``Ex7_Instru_mm_diam_*.txt``

.. code-block:: shell

  ! unit = mm
  ! diameter = True

Let's try:

.. code-block:: python

   my_instru_from_files = InstrumentGeometry('Ex7_Instru_mm_diam_MainBore.txt',
                                             'Ex7_Instru_mm_diam_Valves.txt')
   my_instru_from_files.plot_InstrumentGeometry()
   plt.show()

You can see a warning message indicating that there is here a conflict between the given option
(the default option: ``unit='m', diameter=False``\ ) and the one indicated in the file. The one
of the files are chosen. You can avoid the warning by indicating the correct option

.. code-block:: python

   my_instru_from_files_wo_warning = InstrumentGeometry('Ex7_Instru_mm_diam_MainBore.txt',
                                                        'Ex7_Instru_mm_diam_Valves.txt',
                                                        unit='mm', diameter=True)

Without head-lines, the unit given as keyword is used (by default, meter and radius)

Each file having its own options, it is possible to use a file in mm for the
main bore and a file in meter for the holes or valves

.. code-block:: python

   my_instru_several_options = InstrumentGeometry('Ex7_Instru_mm_diam_MainBore.txt',
                                                  'Ex7_Instru_m_rad_Valves.txt')

Write Files
-----------

Once the :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>` created,
it is possible to write the file with any convention. In addition to previous options two other keywords are available


* ``digit`` : allows to fix the number of digit
* ``disp_optim`` (usefull only for optimization/inversion) : if false, writes only the value of the parameter and not the optim options (bounds, etc)

.. code-block:: python

   my_instru_with_valve.write_files('Test_mm_diam', digit=1, unit='mm', diameter=True, disp_optim=False)
   my_instru_with_valve.write_files('Test_m_rad', digit=4, unit='m', diameter=False, disp_optim=False)
