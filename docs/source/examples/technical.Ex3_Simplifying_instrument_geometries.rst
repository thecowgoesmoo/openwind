
Ex. 3 : Simplifying Instrument Geometries
=========================================

.. code-block:: python

   from openwind.technical import InstrumentGeometry
   import matplotlib.pyplot as plt

If you measured a real instrument, you have noted the radius of the bore at different points of the instrument. The more measurement points, the better, but this greatly increases the data points of the geometry and the computation of the acoustic behaviour.

This is why it is interesting to simplify a complicated instrument geometry.

.. code-block:: python


   # we load a complex instrument with a lot of segments
   complex_instr = InstrumentGeometry('Ex3_complex_instrument.txt')

   fig1 = plt.figure(1)  # create new figure
   complex_instr.plot_InstrumentGeometry(figure=fig1)  # plot the instrument
   plt.title('A complex and noisy measurement with a lot of segments')

We need to give OW a starting shape, that will be optimised to fit the more complicated geometry. 

.. code-block:: python

   # Here we try to simplify our instrument down to 8 linear segments, where the
   # lengths of the segments are fixed, and the radii are to be optimized.
   simplified_bore = [[0.0, 0.2, '~0.005', '~0.005', 'linear'],
                      [0.2, 0.4, '~0.005', '~0.005', 'linear'],
                      [0.4, 0.6, '~0.005', '~0.005', 'linear'],
                      [0.6, 0.8, '~0.005', '~0.005', 'linear'],
                      [0.8, 1.0, '~0.005', '~0.005', 'linear'],
                      [1.0, 1.2, '~0.005', '~0.005', 'linear'],
                      [1.2, 1.4, '~0.005', '~0.005', 'linear'],
                      [1.4, 1.5, '~0.005', '~0.005', 'linear']]

   simplified_instr = InstrumentGeometry(simplified_bore)


   # the AdjustInstrumentGeometry is instanciated from the two Instrument Geometries
   adjustment = AdjustInstrumentGeometry(simplified_instr, complex_instr)
   # the optimization process is carried out
   adjusted_instr = adjustment.optimize_geometry(iter_detailed=False, max_iter=100)

Linear segments work well for relatively simple instruments, but fail to represent round parts correctly. This is where the other shapes come in handy. This will work much better :

.. code-block:: python

   better_simpl_bore = [[0.0, 0.015, '~0.005', '~0.005', 'spline', 0.005, 0.01, '~0.005', '~0.005'],
                        [0.015, 0.1, '~0.005', '~0.005', 'linear'],
                        [0.1, 0.3, '~0.005', '~0.005', 'spline', 0.2, '~0.005'],
                        [0.3, 0.5, '~0.005', '~0.005', 'linear'],
                        [0.5, 1.0, '~0.005', '~0.005', 'linear'],
                        [1.0, 1.5, '~0.005', '~0.005', 'spline', 1.2, 1.3, '~0.005', '~0.005']]

   better_simpl_instr = InstrumentGeometry(better_simpl_bore)

   # the AdjustInstrumentGeometry is instanciated from the two Instrument Geometries
   better_adjust = AdjustInstrumentGeometry(better_simpl_instr, complex_instr)
   # the optimization process is carried out
   better_adjusted_instr = better_adjust.optimize_geometry(iter_detailed=False,
                                                      max_iter=100)

Do not forget to save your simplified instrument using :

.. code-block:: python

   better_adjusted_instr.write_files('simplified_instrument')
