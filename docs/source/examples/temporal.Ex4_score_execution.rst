
Ex. 4: Specify and modify a score
=================================

How to simulate opening and closing tone holes according to a specified score.

This example uses low level classes to run the temporal simulations as
:py:class:`Score <openwind.technical.score.Score>` , :py:class:`TemporalSolver <openwind.temporal.temporal_solver.TemporalSolver>`
and :py:class:`ExecuteScore <openwind.temporal.execute_score.ExecuteScore>` classes.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from openwind import (InstrumentGeometry, Player, InstrumentPhysics,
                         TemporalSolver)
   from openwind.technical import Score
   from openwind.temporal import ExecuteScore, RecordingDevice

a simple instrument with one hole ...

.. code-block:: python

   geom = [[0, 0.5, 2e-3, 10e-3, 'linear']]
   hole = [['label', 'position', 'radius', 'chimney'],
           ['hole1', .25, 3e-3, 5e-3]]

 ... and 2 fingerings
.. code-block:: python

   fingerings = [['label', 'note1', 'note2'],
                 ['hole1', 'o', 'x']]
   instrument = InstrumentGeometry(geom, hole, fingerings)

instanciate an instrument with a clarinet "mouthpiece"

.. code-block:: python

   player = Player('CLARINET')
   instrument_physics = InstrumentPhysics(instrument, 20, player, False)
   temporalsolver = TemporalSolver(instrument_physics, l_ele=0.01,
                                      order=4)

Run simulation!
---------------

The player is updated with the empty score ``no_note_events``

.. code-block:: python

   no_note_events = []
   player.update_score(no_note_events)

Run a temporal simulation with a duration (here .1)

.. code-block:: python

   temporalsolver.run_simulation(.1)

Run simulation and record output signals:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

We create a list of notes and their time of beginning

.. code-block:: python

   note_events = [('note1', .02), ('note2', .03), ('note1', .04)]

The player is updated with the new score ``note_events``

.. code-block:: python

   player.update_score(note_events, 1e-3)
   player.plot_controls(time)

the output will be stored in a Recording device

.. code-block:: python

   rec = RecordingDevice(record_energy=False)

run the simulation with a duration (here .1) and a callback class

.. code-block:: python

   temporalsolver.run_simulation(.1, callback=rec.callback)
   rec.stop_recording()

plot the output value of pressure at the bell

.. code-block:: python

   output_bell = rec.values['bell_radiation_pressure']
   plt.figure()
   plt.plot(output_bell)

If you do a mistake
^^^^^^^^^^^^^^^^^^^

An error message prompts when the asked notes are not in the fingering chart

.. code-block:: python

   strange_note = [('Do', .02), ('Re', .03), ('E', .04)]
   player.update_score(strange_note, 1e-3)
   temporalsolver.run_simulation(.1)

Low level instanciation
-----------------------

:py:class:`ExecuteScore <openwind.temporal.execute_score.ExecuteScore>` makes the link between a score (list of notes) and and instrument and its fingering

.. code-block:: python

   score_execution = ExecuteScore(instrument.fingering_chart,
                                  temporalsolver.t_components)

Set a score based on this empty list of notes

.. code-block:: python

   no_note_score = Score(no_note_events)

``set_score`` allows to modify the score with a series of notes

.. code-block:: python

   score_execution.set_score(no_note_score)

``set_fingering`` takes a time t (here 10) and sets the correct fingering
according to the given notes series

.. code-block:: python

   score_execution.set_fingering(10)

We create a new score with notes. the second parameter is the transition duration between notes (here 1e-3)

.. code-block:: python

   with_note_score = Score(note_events, 1e-3)

display the score along time

.. code-block:: python

   time = np.linspace(0,0.1,1000)
   with_note_score.plot_score(time)

change the score of the ``score_execution`` instance

.. code-block:: python

   score_execution.set_score(with_note_score)
   score_execution.set_fingering(1.5)
