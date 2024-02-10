Ex. 5 : Create a Player
============================

How to create a Player and change its values

This example uses the class:
:class:`Player <openwind.technical.player.Player>`


About Player
------------

In order to perform a temporal computation in Openwind, you will have to
create a player.

A player consists in a couple Excitator / Score

Excitators are initiated from excitator parameters dictionnaries which you can
find in openwind/technical/default_excitator_parameters. The default one is
UNITARY_FLOW, which looks like this:

.. code-block:: python

	UNITARY_FLOW = {
	    "excitator_type":"Flow",
	    "input_flow":1
	}


You can see that the excitator is a Flow, and it's value is constant and equal
to one.

It is important to understand that you can choose between several kinds of
excitators (Flow and Reed1dof at the moment), and according to the excitator type
you will have different excitator parameters. Parameters are time dependant
functions (or curves) or constant values. For a Flow, you will only have
"input_flow" as a parameter. For a Reed1dof, you will have "opening",
"mass", "section", "pulsation", "dissip", "width", "mouth_pressure", "model",
"contact_pulsation" and "contact_exponent". You can have a glimpse at the code
here: openwind/continuous/excitator.py


A Score is defined by a list of note_events and a transition_duration.
- note_events are tuples with the note name and the starting time of the note
- transition_duration is a float which give the duration between two notes


Create Player
--------------------

*First the imports*

.. code-block:: python

	import numpy as np
	import matplotlib.pyplot as plt
	from openwind import Player
	from openwind.technical.temporal_curves import constant_with_initial_ramp


Starting by creating a default player

.. code-block:: python

	player = Player()

player's Excitator is a "UNITARY_FLOW", and it's Score is empty


Player Modification
-------------------

You can now change the value of the input_flow of the score

.. code-block:: python

	player.update_curve("input_flow",2*np.pi*3700)

Or you can create a custom_flow dictionnary and update the player with it

.. code-block:: python

	custom_flow = {
	    "excitator_type":"Flow",
	    "input_flow": constant_with_initial_ramp(2000, 2e-2)
	}
	player.update_curves(custom_flow)


pay attention to the "s" at the end of update_curves, it is not the same method as above

You can check the new value of your input_flow for t =[-5,5] :

.. code-block:: python

	time_interval = np.linspace(-5,5,1000)
	player.plot_one_control("input_flow",time_interval)


Of course, you can update your player with all excitator dictionnaries
that are stored in default_excitator_parameters:

.. code-block:: python

	player.set_defaults("IMPULSE_400us")


**IMPORTANT NOTE**: if your player was created with a Flow excitator, you can not change it to another type of excitator. This is forbidden to prevent misusage of the code. If you want to use a Reed1dof instead of a Flow, you must create a new Player

Let's say we want to have a player that plays Oboe:

.. code-block:: python

	oboe_player = Player("OBOE")


oboe_player is using this Excitator :

.. code-block:: python

	OBOE = {
	   "excitator_type" : "Reed1dof",
	   "opening" : 8.9e-5,
	   "mass" : 7.1e-4,
	   "section" : 4.5e-5,
	   "pulsation" : 2*np.pi*600,
	   "dissip" : 0.4*2*np.pi*600,
	   "width" : 9e-3,
	   "mouth_pressure" : constant_with_initial_ramp(12000, 2e-2),
	   "model" : "inwards",
	   "contact_pulsation": 316,
	   "contact_exponent": 4
	 }


Let's say you want to change the mouth pressure value, once again :

.. code-block:: python

	oboe_player.update_curve("mouth_pressure",
                         	 constant_with_initial_ramp(13000, 2e-2))


You can plot all controls for the oboe_player :

.. code-block:: python

	oboe_player.plot_controls(time_interval)


At some point, if you got lost with your player, you can check which default dictionnaries availables

.. code-block:: python

	oboe_player.print_defaults()
	oboe_player.set_defaults("WOODWIND_REED")


Score Modification
------------------

If you want to modify your Score, first create a new note_events list :

.. code-block:: python

	note_events = [('note1', .02), ('note2', .03), ('note1', .04)]


Then you can change the transition_duration:

.. code-block:: python

	transition_duration = 1e-3


And finaly update your player's score:

.. code-block:: python

	oboe_player.update_score(note_events, transition_duration)
