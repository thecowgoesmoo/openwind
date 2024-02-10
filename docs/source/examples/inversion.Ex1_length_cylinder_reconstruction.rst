Ex. 1: Length Cylinder Reconstruction
=====================================

Introduction to the basic aspects of bore reconstruction with OpenWInD.

The aim of this example is to introduce the basic aspects of bore reconstruction with openWInD.

This example uses the :py:class:`InverseFrequentialResponse <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse>` class.

It also necessitates the classes:


* :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
* :py:class:`Player <openwind.technical.player.Player>`
* :py:class:`InstrumentPhysics <openwind.continuous.instrument_physics.InstrumentPhysics>`

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
The geometry is 0.5m cylinder with a radius of 2mm.

.. code-block:: python

   target_geom = [[0, 0.5, 2e-3, 2e-3, 'linear']]
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

Definition Of The Optimized Geometry
------------------------------------

Here we want to adjust only the pipe length: only this parameter is preceded by "~"

.. code-block:: python

   inverse_geom = [[0, '~0.3', 2e-3, 2e-3, 'linear']]

the initial length is set here to 0.3m

.. code-block:: python

   instru_geom = InstrumentGeometry(inverse_geom)

During the process, an attribute ``optim_param`` has been instanciated.
It contains all the information on the parameters included in the optimization

.. code-block:: python

   print(instru_geom.optim_params)

We can compare the two bore at the initial state

.. code-block:: python

   fig_geom = plt.figure()
   target_computation.plot_instrument_geometry(figure=fig_geom, label='Target')
   instru_geom.plot_InstrumentGeometry(figure=fig_geom, label='Initial Geometry')
   fig_geom.legend()

Construction Of The Inverse Problem
-----------------------------------

Instanciate a player with defaults

.. code-block:: python

   player = Player()

Instanciation of the physical equation

.. code-block:: python

   instru_phy = InstrumentPhysics(instru_geom, temperature, player, losses)

Instanciation of the inverse problem

.. code-block:: python

   inverse = InverseFrequentialResponse(instru_phy, frequencies, Ztarget)

We can now compare the impedances at the initial state

.. code-block:: python

   inverse.solve()
   fig_imped = plt.figure()
   target_computation.plot_impedance(figure=fig_imped, label='Target', marker='o',
                                     linestyle=':')
   inverse.plot_impedance(figure=fig_imped, label='Initial', marker='x',
                          linestyle=':')

Optimization Process
--------------------

the InverseFrequentialResponse has a method which computes the cost and
gradient for a given value of the design parameters

.. code-block:: python

   cost, grad = inverse.get_cost_grad_hessian([], grad_type='adjoint')[0:2]
   print('With current geometry: Cost={:.2e}; Gradient={:.2e}'.format(cost,
                                                                      grad[0]))

This method can be used with any optimization algorithm.
This is what it is done in the dedicated method:

.. code-block:: python

   result = inverse.optimize_freq_model(iter_detailed=True)

The default optimization algorithm chosen is 'lm' for "Levenberg-Marquart"
(from scipy) which is often the most efficient for unconstrained problem.

Plot The Result
---------------

.. code-block:: python

   print('The final length is {:.2f}m'.format(result.x[0]))
   print('The deviation w.r. to the target value is '
         '{:.2e}m'.format(np.abs(result.x[0] - 0.5)))

we add the final impedance to the curve:

.. code-block:: python

   inverse.plot_impedance(figure=fig_imped, label='Final', marker='+',
                          linestyle=':')

we add the final geometry

.. code-block:: python

   instru_geom.plot_InstrumentGeometry(figure=fig_geom, linestyle=':', color='k',
                                       label='Final Geometry')
   fig_geom.legend()

plot the evolution of the length

.. code-block:: python

   plt.figure()
   plt.plot(np.arange(0, result.nit), np.array(result.x_evol))
   plt.xlabel('Iterations')
   plt.ylabel('Length (m)')
