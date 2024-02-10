
Ex. 6: Chose losses model
=========================

How to chose the model used for the thermo-viscous losses. The different available models and their implementations are described in :


#. Alexis Thibault, Juliette Chabassier, "Viscothermal models for wind musical instruments", RR-9356, Inria Bordeaux Sud-Ouest (2020) `hal-02917351 <https://hal.inria.fr/hal-02917351>`_

This example uses the :py:class:`ImpedanceComputation <openwind.impedance_computation.ImpedanceComputation>` class.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import ImpedanceComputation

   fs = np.arange(20, 2000, 1)
   geom = 'Oboe_instrument.txt'
   holes = 'Oboe_holes.txt'

By default the losses are set to True

.. code-block:: python

   result = ImpedanceComputation(fs, geom, holes)

You can apply a variable temperature by defining a function all physical coefficients follow the temperature profile. In this case, the temperature variation is along the main axes. In the holes the temperature is uniform and equals the one in the main bore at their location.

.. code-block:: python

   total_length = result.get_instrument_geometry().get_main_bore_length()
   def grad_temp(x):
       T0 = 37
       T1 = 21
       return 37 + x*(T1 - T0)/total_length

Losses
------

The losses coefficient also follow the temperature profile.
``Losses`` can be a boolean, or in ``{'bessel', 'wl', 'keefe', 'minifkeefe','diffrepr', 'diffrepr+'}``\ : whether/how to take into account viscothermal losses. Default is True.


 * ``True`` and ``bessel`` : Zwikker-Kosten model
 * ``bessel_new`` : Zwikker-Koster model with modified loss coefficients to account for conicity. See [2].
 * ``wl`` : Webster-Lokshin model
 * ``keefe`` and ``minikeefe`` : approximation of Zwikker-Kosten for high Stokes number
 * ``diffrepr`` : diffusive representation of Zwikker-Kosten.
 * ``diffrepr+``\ : use diffusive representation with explicit additional variables (see [1])

.. code-block:: python

   losses_cats = [False,'bessel','wl','keefe','minikeefe','diffrepr','diffrepr+']
   markers = ['x','o','s','+','^','v','d']

   results= dict()
   fig = plt.figure()
   for marker, losses in zip(markers, losses_cats):
       result = ImpedanceComputation(fs, geom, holes, temperature=grad_temp,
                                       radiation_category=rad,
                                       losses=losses)
       label = f'losses={losses}'
       result.plot_impedance(figure=fig, label=label, marker=marker, markevery=200)
       results[label] = result
