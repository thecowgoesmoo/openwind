
Ex. 2: Complex Shape On Measurement
===================================

Inversion of a complex shape from real measurements.

This example uses the :py:class:`InverseFrequentialResponse <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse>` class.

Imports
-------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind.inversion import InverseFrequentialResponse
   from openwind import InstrumentGeometry, Player, InstrumentPhysics
   from openwind.impedance_tools import read_impedance, plot_impedance
   from openwind.inversion.display_inversion import plot_evolution_geometry
   plt.close('all')

In this example a more complex shape is treated: a spline, including 4 design
variables. It is an occasion to approach the different types of design variables.
The target comes from real measurement.

Global Options
--------------

.. code-block:: python

   frequencies = np.linspace(100, 1e3, 10)
   temperature = 20
   losses = True

Target From Measurement
-----------------------

the geometry of this cylinder is: 436mm of long and 2mm of inner radius

.. code-block:: python

   f_mes, Z_mes = read_impedance('Impedance_20degC_MeasureCyl_L436mm_R2mm.txt',
                                 df_filt=5)
   Z_target = np.interp(frequencies, f_mes, Z_mes)
   fig_imped = plt.figure()
   plot_impedance(f_mes, Z_mes, figure=fig_imped, label='Measure filtered')
   plot_impedance(frequencies, Z_target, figure=fig_imped, label='Target',
                  linestyle='None', marker='o')
   plt.xlim([np.min(frequencies), np.max(frequencies)])

Definition Of The Optimized Geometry
------------------------------------

The initial geometry is here much more complex: a spline with 4 points

.. code-block:: python

   inverse_geom = [[0, '0.3<~.4<0.5', 2e-3, '~4e-3', 'spline',
                    0.05, 0.15, '0<~5e-3<0.01', '0<~2e-3']]

The entrance radius, and the location of intermadiate points are fixed .
Here different kind of indication are given for each parameters:


* '0.3<~.4<0.5': the total length of the pipe is restrained to the range [0.3,0.5]. The initial value is set to 0.4
* '~4e-3': the right end radius is unconstrained (can be negative). The initial value is set to 4e-3
* '0<~5e-3<0.01': the radius of second point of the spline is restrained to the range [0,0.01]. The initial value is set to 5e-3.
* '0<~2e-3': the third radius is imposed to be positive. The initial value is set to 2e-3.

.. code-block:: python

   instru_geom = InstrumentGeometry(inverse_geom)

During the process, an attribute ``optim_param`` has been instanciated.
It contains all the information on the parameters included in the optimization

.. code-block:: python

   print(instru_geom.optim_params)

We can plot the initial bore profile

.. code-block:: python

   fig_geom = plt.figure()
   instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')
   fig_geom.legend()

Construction Of The Inverse Problem
-----------------------------------

.. code-block:: python

   lengthElem = 0.05
   order_optim = 6

Instanciation of the physical equation

.. code-block:: python

   instru_phy = InstrumentPhysics(instru_geom, temperature, Player(), losses)

Instanciation of the inverse problem

.. code-block:: python

   inverse = InverseFrequentialResponse(instru_phy, frequencies, Z_target,
                                        l_ele=lengthElem, order=order_optim)

We can now compare the impedances at the initial state

.. code-block:: python

   inverse.solve()
   inverse.plot_impedance(figure=fig_imped, label='Initial', marker='v',
                          linestyle='')

And Now The Optimization
------------------------

.. code-block:: python

   result = inverse.optimize_freq_model(iter_detailed=True)

The default optimization algorithm chosen is 'trf' for "Trust Region
Reflective" (from scipy) which is often the most efficient for constrained
problem.
plot the final impedance

.. code-block:: python

   inverse.plot_impedance(figure=fig_imped, label='Final', marker='^',
                          linestyle='')

and the final geometry

.. code-block:: python

   instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Final Geometry')
   fig_geom.legend()

We can also plot the evolution of the geometry

.. code-block:: python

   plot_evolution_geometry(inverse, result.x_evol, double_plot=False)
