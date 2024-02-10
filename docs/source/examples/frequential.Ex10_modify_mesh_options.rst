
Ex. 10: modify mesh options
===========================

How to fix the spatial discretization options (the mesh).

This example uses the classes:

* :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>` class.
* :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
* :py:class:`Player <openwind.technical.player.Player>`
* :py:class:`InstrumentPhysics <openwind.continuous.instrument_physics.InstrumentPhysics>`
* :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import (ImpedanceComputation, InstrumentGeometry, Player,
                         InstrumentPhysics, FrequentialSolver)
   from openwind.discretization import Mesh

   fs = np.arange(20, 2000, 1)
   geom_filename = 'Geom_trumpet.txt'

Chose your discretization characteristics
-----------------------------------------

Chosen fine discretization:


#. choose a length for the finite elements
   .. code-block:: python

      length_FEM = 0.1

#. choose an order for the finite elements
   .. code-block:: python

      order_FEM = 10

#. Find file 'Geom_trumpet.txt' describing the bore, and compute its impedance with specified length and order for the finite elements
   .. code-block:: python

      result = ImpedanceComputation(fs, geom_filename, l_ele = length_FEM, order = order_FEM)

#. Plot the discretization information
   .. code-block:: python

      result.discretization_infos()

#. Plot the impedance
   .. code-block:: python

      fig = plt.figure()
      result.plot_impedance(figure=fig, label=f"given fine discretization, nb dof = {result.get_nb_dof()}")

Chosen coarse discretization

.. code-block:: python

   length_FEM = 0.1
   order_FEM = 2
   result = ImpedanceComputation(fs, geom_filename, l_ele = length_FEM, order = order_FEM)
   result.discretization_infos()
   result.plot_impedance(figure=fig, label=f"given coarse discretization, nb dof = {result.get_nb_dof()}")

Default options
---------------

default is an adaptative mesh that provides a reasonable solution with a  low computational cost

.. code-block:: python

   result_adapt = ImpedanceComputation(fs, geom_filename)
   result_adapt.discretization_infos()
   result_adapt.plot_impedance(figure=fig, label=f"adaptive discretization, nb dof = {result_adapt.get_nb_dof()}")

Modify the minimal order for automatic mesh
-------------------------------------------


Load and process the instrument geometrical file

.. code-block:: python

   instr_geom = InstrumentGeometry(geom_filename)

Create a player using the default value : unitary flow for impedance computation

.. code-block:: python

   player = Player()

Choose the physics of the instrument from its geometry. Default models are chosen when they are not specified.  Here ``losses = True`` means that Zwikker-Koster model is solved.

.. code-block:: python

   instr_physics = InstrumentPhysics(instr_geom, temperature=25, player = player, losses=True)
   Mesh.ORDER_MIN = 4

Perform the discretization of the pipes and put all parts together ready to be solved.

.. code-block:: python

   freq_model = FrequentialSolver(instr_physics, fs)
   # Solve the linear system underlying the impedance computation.
   freq_model.solve()
   freq_model.discretization_infos()
   freq_model.plot_impedance(figure=fig, label=f"adaptive discretization orders > 4, nb dof = {freq_model.n_tot}")


   freq_model = FrequentialSolver(instr_physics, fs, order=2)
   freq_model.solve()
   freq_model.discretization_infos()
   freq_model.plot_impedance(figure=fig, label=f"adaptive discretization orders > 4, nb dof = {freq_model.n_tot}")
