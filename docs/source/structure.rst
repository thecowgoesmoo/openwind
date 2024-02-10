Code structure
##############

This section will present how Openwind is currently working. After a short
presentation of the global data workflow, the main modules of
Openwind and the input format that can be used with Openwind,
we are going to present each module and their submodules, plus the main classes
to get an idea of how modules are interconnected.

We made the choice not to use a UML diagram as it requires some knowledge of the UML format to be understood, so we tried to make it as understandable as possible.

..
  _(altough it is available `here <https://openwind.gitlabpages.inria.fr/web/uml.html>`_)


.. contents:: :local:
    :depth: 3


1. Global workflow
==================

.. image :: https://files.inria.fr/openwind/pictures/data-flow.png
  :width: 1000
  :align: center

So the very basic data flow in Openwind is :

.. code-block:: shell

            Raw Data (text files)
                    |
                    |
                    v
            Sketch (DesignShapes)
                    |
                    |
                    v
  Graph (Pipes, Junctions, Radiations, Excitators)
                    |
                    |
                    v
        Pressure & Flow Computation
         (Frequential & Temporal)

.. note::

    You don't need to do all those steps manually to make some computation, we
    have implemented this complete workflow in two scripts :
    :py:mod:`impedance_computation <openwind.impedance_computation>` and
    :py:mod:`temporal_simulation <openwind.temporal_simulation>`

This is the very high level overview, now let's dig a bit more into the code

.. _input_format:

2. Input Format
===============

Before we get started with the structure of Openwind, we are going to detail
quickly the input formats that can be used with openwind.

.. warning::

  Inputs are plain text files. We have diplayed them in tables here to make them
  easier to understand. When writing inputs in text formats, the delimiter is a
  space. Have a look at the :doc:`examples </examples/technical.Ex1_importing_instrument_geometries_into_OW>` for more details


Geometry
--------

The geometry input formats are shown in the :doc:`Importing geometry into openwind example </examples/technical.Ex1_importing_instrument_geometries_into_OW>`


Holes
-----

The holes input format is shown in the :doc:`Handling side holes example </examples/technical.Ex2_Handling_side_holes>`


Fingering Chart
---------------

The fingering chart format is shown in the :doc:`Handling side holes example </examples/technical.Ex2_Handling_side_holes>`


3. Modules workflow
===================

.. _technical:

a. Technical & Design
---------------------

.. important::

  The :py:mod:`technical <openwind.technical>` module aims to parse the user's :ref:`input <input_format>` into Openwind objects.

It can be divided in to types of input:

  * the information on the geometry of the instrument necessary to any simulation
  * the information on how this instrument is played, especially important for temporal simulation

Instrument geometry
~~~~~~~~~~~~~~~~~~~

In Openwind the geometry is described by :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
that can be found in the :py:mod:`design <openwind.design>` module.

.. image :: https://files.inria.fr/openwind/pictures/design+technical-module.png
  :width: 1000
  :align: center


*The legend for this figure is in the* :ref:`appendix <legend>`

The :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
class reads the input files and parse them into a list of
:py:class:`DesignShape <openwind.design.design_shape.DesignShape>` to build a
*main bore*. The *holes* file is parsed into a
:py:class:`Hole <openwind.technical.instrument_geometry.Hole>` object, the *fingering chart* file
is parsed into a :py:class:`FingeringChart <openwind.technical.fingering_chart.FingeringChart>` object.

The parsing is done through a call to private methods of :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`,
like `_create_main_bore_shapes()`, `_create_holes_shapes()`, `_create_fingering_chart()`
(those methods are not included in the documentation yet as they are private,
but you can have a look at the code !)

Each geometric parameters is associated to a :py:class:`DesignParameter <openwind.design.design_parameter.DesignParameter>`
which can be fixed or variable (used in optimization process). The :py:class:`OptimizationParameters <openwind.design.design_shape.OptimizationParameters>` is used only in the :ref:`inversion <inversion>`
to modify the value of the :py:class:`DesignParameter <openwind.design.design_parameter.DesignParameter>` during the optimization process.



You can now have a look at the :ref:`examples to define a geometry <define_a_geometry>`
to see how the InstrumentGeometry parsing can be used.


.. _player:

Player
~~~~~~

.. important::

  The :py:class:`Player <openwind.technical.player.Player>` defines the action needed to virtually play the instrument. It is especially important for temporal computation

..
	The player gets a :py:mod:`control curves dictionnary <openwind.technical.default_excitator_parameters>`
	indicating the value and the temporal evolution of the control parameters, and a succession of notes + a transition duration to create a :py:class:`Score <openwind.technical.score.Score>`

.. image :: https://files.inria.fr/openwind/pictures/player-simple.png
  :width: 700
  :align: center

The input are here a set of caracteristics about the excitator mecanism and the musician control in a dictionnary (:py:mod:`default dictionnaries are avaible <openwind.technical.default_excitator_parameters>`).
Its indicate the excitator type (reed, flow) and the temporal evolution of the control parameters associated.

A list of note events can also be given, allowing :py:class:`Player <openwind.technical.player.Player>` to
instanciate a :py:class:`Score <openwind.technical.score.Score>` indicating which note must be played at each time.

To understand how the player is used, you can have a look at
:doc:`the player example </examples/technical.Ex5_Create_Player>`.

More details on the :py:class:`Player <openwind.technical.player.Player>` can be
found in its own docstring.


.. _continuous:

b. Continuous
-------------

.. important::

  This module gives a physical meaning from the :ref:`technical<technical>` information.
  It associates coefficients to each element of the instrument corresponding to continuous equations (in space and time).


This is probably the most complex module of Openwind, as it is here that the
instrument (:py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>` + :py:class:`Player<openwind.technical.player.Player>` ) is transformed
into a graph (that we have called the :py:class:`Netlist <openwind.continuous.netlist.Netlist>`)
of components. Those components are linked to physical models that will be used for
:ref:`Frequential <frequential_computation>` and :ref:`Temporal <temporal_computation>`
computations.

The components can be divided in:

* :py:class:`Pipes <openwind.continuous.pipe.Pipe>` in which waves propagate
* :py:class:`NetlistConnector <openwind.continuous.netlist.NetlistConnector>` defining the boundary condition of the pipes. It can be:

  - :py:class:`Excitators <openwind.continuous.excitator.Excitator>`
  - :py:class:`Junctions <openwind.continuous.junction.PhysicalJunction>`
  - :py:class:`Radiation Models <openwind.continuous.physical_radiation.PhysicalRadiation>`

Graphically, it goes like this:

.. image :: https://files.inria.fr/openwind/pictures/graph.png
  :width: 1000
  :align: center


Pipes and connectors are linked together by a
:py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>` that is an object which
knows to which component it is linked to. For a main bore with one side-hole, it gives the following graph:


.. image :: https://files.inria.fr/openwind/pictures/components.png
  :width: 1000
  :align: center

Now let's have a more detailed look into the continuous module :

.. image :: https://files.inria.fr/openwind/pictures/continuous-module.png
  :width: 1000
  :align: center


Every component of the graph have the methods to compute the coefficients of the equations modelling them. The :py:class:`Pipe <openwind.continuous.pipe.Pipe>` is associtated to wave propagation equations and is defined by:

* a :py:class:`DesignShape <openwind.design.design_shape.DesignShape>` defining its geometry
* :py:class:`physical quantities <openwind.continuous.physics.Physics>` giving the value of the air  density, sound celerity etc.
* a :py:class:`Scaling <openwind.continuous.scaling.Scaling>` object, giving a set of values used to normalized the coefficients and avoid numerical issues
* a :py:class:`losses model <openwind.continuous.thermoviscous_models.ThermoviscousModel>` specifying how the thermoviscous losses are modeled


The other components do also have their own physical models. We are not going to
detail all of them as it would be a little bit long here, but we are just going
to take one connector (a :py:class:`Simple Junction <openwind.continuous.junction.SimpleJunction>`)
to see how it is defined.

The SimpleJunction is inherited from the
:py:class:`PhysicalJunction <openwind.continuous.junction.PhysicalJunction>`, itself
an inheritance of the :py:class:`NetlistConnector <openwind.continuous.netlist.NetlistConnector>`.
It has thus :

* a label
* a :py:class:`Scaling <openwind.continuous.scaling.Scaling>`

The :py:class:`Excitator <openwind.continuous.excitator.Excitator>` is also an inheritance of the :py:class:`NetlistConnector <openwind.continuous.netlist.NetlistConnector>`
which can be either a :py:class:`Flow<openwind.continuous.excitator.Flow>` or a :py:class:`Reed1dof<openwind.continuous.excitator.Reed1dof>`. Following the :ref:`Player <player>` information,
it is able to give the value of the coefficients of the equations modeling the excitator.

To finish on this module, let's have a look at the
:py:class:`InstrumentPhysics <openwind.continuous.instrument_physics.InstrumentPhysics>`
class. It is the *conductor* for the continuous module. It will create the whole
graph (or Netlist) given the :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`,
the temperature, the :py:class:`Player <openwind.technical.player.Player>`, the
:py:class:`ThermoviscousModel <openwind.continuous.thermoviscous_models.ThermoviscousModel>` and
:py:class:`PhysicalRadiation <openwind.continuous.physical_radiation.PhysicalRadiation>` and a bunch of
other optional parameters.

Once the Netlist is defined, we are going to descretize its components and then
we will be able to make the impedance computation.


c. Discretization
-----------------

.. important::

	This module perform the spatial discretization of the element in which the waves propagate.

.. image :: https://files.inria.fr/openwind/pictures/discretization-module.png
  :width: 400
  :align: center

In the :ref:`continuous <continuous>` module, only the :py:class:`Pipe<openwind.continuous.pipe.Pipe>` components are associated to equation which must be spatially discretized.

In openWInD, this discretization is performed thanks to the finite element method. The :py:class:`DiscretizedPipe <openwind.discretization.discretized_pipe.DiscretizedPipe>` therefore associates to a given
:py:class:`Pipe <openwind.continuous.pipe.Pipe>` a :py:class:`Mesh <openwind.discretization.mesh.Mesh>` composed of :py:class:`Elements <openwind.discretization.element.Element>`.
The computation of the nodes locations and base functions follows the Gauss-Lobato quadrature which is performed by the :py:class:`GLQuad <openwind.discretization.glQuad.GLQuad>` class.


.. seealso::

  More details on the discretization process can be found on the `corresponding publication <https://hal.archives-ouvertes.fr/hal-01963674>`_.


.. warning::

	This module is never directly used. The discretization is performed implicitely by the :ref:`frequential<frequential_computation>` or the :ref:`temporal<temporal_computation>` modules.

.. _frequential_computation:

d. Frequential
--------------


.. important::

  The :py:mod:`frequential <openwind.frequential>` module aims to

  * convert the graph (:py:class:`Netlist <openwind.continuous.netlist.Netlist>`) into a frequential graph of :py:class:`FrequentialComponent <openwind.frequential.frequential_component.FrequentialComponent>`
  * make the frequential computation to get the impedance of the instrument

Let's have a look at this module:

.. image :: https://files.inria.fr/openwind/pictures/frequential-module.png
  :width: 1000
  :align: center

This figure shows that the frequential graph is almost the same as the Netlist in
the `continuous` module. Also, one can see that the *conductor* here is the
:py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`
that knows the Netlist, the frequential array, and has the :py:meth:`solve() <openwind.frequential.frequential_solver.FrequentialSolver.solve()>` that computes
the impedance.

There are two other modules, the :py:mod:`frequential_pressure_condition <openwind.frequential.frequential_pressure_conditionr>`
and the :py:mod:`frequential_interpolation <openwind.frequential.frequential_interpolation>`, that are used
respectively for using Dirichlet boundary condition on the pressure unknown, and to
compute frequential interpolation in the solver.


You can now have a look at the :ref:`compute frequential response <compute_frequential_response>`
to see how the this module can be used.

.. _temporal_computation:

e. Temporal
-----------

.. important::

  The :py:mod:`temporal <openwind.temporal>` module aims to

  * convert the graph (:py:class:`Netlist <openwind.continuous.netlist.Netlist>`) into a temporal graph of :py:class:`TemporalComponent <openwind.frequential.tcomponent.TemporalComponent>`
  * make the temporal computation to get the pressure & flow of the instrument for a given array of time

Let's have a look at this module:

.. image :: https://files.inria.fr/openwind/pictures/temporal-module.png
  :width: 1000
  :align: center

This figure shows that the temporal graph is almost the same as the Netlist in
the `continuous` module. Also, one can see that the *conductor* here is the
:py:class:`TemporalSolver <openwind.frequential.temporal_solver.TemporalSolver>`
that knows the Netlist.

Each :py:class:`TemporalComponent <openwind.frequential.tcomponent.TemporalComponent>` has a `one_step()` method
that is used by the `temporal solver` to compute the pressure & flow for each
time step.

You can now have a look at the :ref:`perform temporal simulation <perform_temporal_simulation>` to see how the this module can be used.


.. _inversion:

f. Inversion
------------

.. important::

	The :py:mod:`inversion <openwind.inversion>` module aims to solve an inverse porblem in the frequential domain: find the geometry corresponding to some acoustics characteristics


.. image :: https://files.inria.fr/openwind/pictures/inversion-module.png
  :width: 1000
  :align: center

The :py:class:`InverseFrequentialResponse <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse>` class can be seen as a simple extension of the :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>` class.
From the solution of the direct problem, the cost function is computed as the squared norm of an observable (some examples are given in :py:mod:`observation<opewind.inversion.observation>`). :py:class:`InverseFrequentialResponse <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse>`
is also able to compute the gradient of this cost function with respect to some design variable specify in :py:class:`OptimizationParameters<openwind.design.desing_parameter.OptimizationParameters>`,
allowing him to perform an optimization by using its method :py:meth:`optimize_freq_model() <openwind.inversion.inverse_frequential_response.InverseFrequentialResponse.optimize_freq_model>`.

Appendix
========

All modules
-----------

You can see the same data flow as in the introduction but with a more complexe overview now

.. image :: https://files.inria.fr/openwind/pictures/all-modules.png
  :width: 1000
  :align: center


.. _legend:

Legend
------

.. image :: https://files.inria.fr/openwind/pictures/legend.png
  :width: 1000
  :align: center
