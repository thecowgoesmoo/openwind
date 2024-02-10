#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright (C) 2019-2023, INRIA
#
# This file is part of Openwind.
#
# Openwind is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Openwind is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Openwind.  If not, see <https://www.gnu.org/licenses/>.
#
# For more informations about authors, see the CONTRIBUTORS file

"""
Module for the Player class
"""

import matplotlib.pyplot as plt
import numpy as np
import warnings

from openwind.technical import default_excitator_parameters as EXCITATOR_DEFAULTS
from openwind.technical import Score

from openwind.continuous import Excitator

AVAILABLE_DEFAULTS = [p for p in EXCITATOR_DEFAULTS.__dict__.keys() if
                      p[0] != "_" and p not in
                      ["np","constant_with_initial_ramp", "dirac_flow",
                       "triangle"]]

AVAILABLE_EXC_TYPES = [clss.__name__.lower() for clss in Excitator.__subclasses__()]

class Player:
    """
    The control parameters of the instrument.

    This object must contain:

    - a type of excitator
    - the corresponding series of control curves
    - a list of note_events
    - the transition duration

    Parameters
    ----------
    dict_key: dict or str, optional
        This is a dictionnary or  the name of the default dictionnary that will
        be given to Player. Default dictionnary can be found in
        :py:mod:`Default Excitator Parameters <openwind.technical.default_excitator_parameters>`
        Player's constructor will parse the default dictionnary entries into
        attributes. The default is 'UNITARY_FLOW'.
    note_events : List[(string, float)], optional
        List of note events: tuples with the note name and the starting
        time of the note. The default is an emtpy list.
    transition_duration: float, optional
        The transition duration between notes. The default is 0.02

    Attributes
    ----------
    excitator_type : str
        Determines if your Player input is a reed, a flute or a flow
    score : :py:class:`Score<openwind.technical.score.Score>`
        The note which must be played at each instant
    control_parameters : dict
        Associate to each control parameters its value and eventually its
        evolution with respect to time.

        - If it is a flow the only keys is "input_flow": the value of the flows
        - If it is a reed the keys must be ["opening", "mass", "section", \
        "pulsation", "dissip", "width", "mouth_pressure", "model", \
        "contact_pulsation", "contact_exponent"]
        - the Flute is not yet implemented

        .. seealso::
            :py:mod:`excitator<openwind.continuous.excitator>`
                for more information on these control parameters


    """

    def __init__(self, dict_key='UNITARY_FLOW', note_events=[],
                 transition_duration=.02):
        self.excitator_type = None
        self.control_parameters = dict()
        if type(dict_key) == str:
            self.set_defaults(dict_key)
        else:
            self.check_control_parameters(dict_key)
            self.update_curves(dict_key)
        self.score = Score(note_events, transition_duration)

    def set_defaults(self, dict_key):
        """
        Method which updates the player attributes values with a default dict
        stored in
        :py:class:`DefaultExcitatorParameters \
        <openwind.technical.default_excitator_parameters>`

        Parameters
        ----------
        dict_key: str
            The key to the parameters dictionnary to load in Player

        """
        # Check that user input exists
        if dict_key not in AVAILABLE_DEFAULTS:
            raise AttributeError('Unknown default excitator. Chose between'
                                 ' {}'.format(AVAILABLE_DEFAULTS))
        else:
            control_param = getattr(EXCITATOR_DEFAULTS, dict_key)
            self.check_control_parameters(control_param)
            self.update_curves(control_param)


    def set_excitator_type(self, new_exc_type):
        if self.excitator_type is None:
            if new_exc_type.lower() in AVAILABLE_EXC_TYPES:
                self.excitator_type = new_exc_type
                self._exc_class = [clss for clss in Excitator.__subclasses__()
                                   if clss.__name__.lower()==new_exc_type.lower()][0]
        elif self.excitator_type != new_exc_type:
            raise ValueError("ERROR: you are trying to change the excitator type "
                             "from '%s' to '%s', but this is forbidden. "
                             "You must create a new Player instance and add it to "
                             "your InstrumentPhysics instance instead"
                             %(self.excitator_type, new_exc_type))
        else:
            pass

    def check_control_parameters(self, control_parameters):
        if "excitator_type" in control_parameters:
            self.set_excitator_type(control_parameters["excitator_type"])
        if not self.excitator_type:
            raise ValueError('Please indicate an "excitator_type."')
        self._exc_class.check_needed_params(control_parameters)

    def update_score(self, note_events, transition_duration=None):
        """
        Update the score

        Parameters
        ----------
        note_events : List[(string, float)]
            List of note events: tuples with the note name and the starting
            time of the note
        transition_duration : float, optional
            The new transition duration between notes, if none, do not change
            the value. The default is None.
        """
        self.score.set_note_events(note_events)
        if transition_duration:
            self.score.set_transition_duration(transition_duration)

    def get_score(self):
        """
        Return the score

        Returns
        -------
        openwind.technical.score.Score
            The :py:class:`Score <openwind.technical.score.Score>` associated
            with the current player
        """
        return self.score

    def update_curve(self, label, new_curve):
        """
        Update a curve given by a label with a new curve

        Parameters
        ----------
        label: str
            It corresponds to one of the key of the Excitator Parameter
            dictionnary
        new_curve: time dependand function
            The new function we want to use as a curve for the attribute
            given by label


        """
        if label == 'excitator_type':
            self.set_excitator_type(new_curve)
        else:
            if label not in self._exc_class.POSSIBLE_PARAMS:
                raise ValueError(f'The parameter "{label}" is not recognized by '
                                 f'"{self.excitator_type}" excitator. Please chose in'
                                 f'between: {self._exc_class.POSSIBLE_PARAMS}.')
            self.control_parameters[label] = new_curve

    def update_curves(self, dict_):
        """
        Update several curves at once from a given dictionnary

        Parameters
        ----------
        dict_: dict
            A dictionnary of curves. Can be either one from \
            :py:class:`Default Excitator Parameters <openwind.technical.default_excitator_parameters>` \
            or one defined by the user

        """
        for label, new_curve in dict_.items():
            self.update_curve(label, new_curve)

    @classmethod
    def print_defaults(cls):
        """
        Class method that will print the available curves dictionnary from
        :py:class:`Default Excitator Parameters <openwind.technical.default_excitator_parameters>`
        """
        #pdb.set_trace()
        print("Available default parameters are %s" %AVAILABLE_DEFAULTS)
        # print("\n Advanced usage : you can add your own default dictionnary to"
        #       " openwind/technical/parameters.py\n")

    def plot_one_control(self, label, time, ax=None):
        """
        Plot one control curve

        Parameters
        ----------
        label : string
            The name of the control plotted.
        time : np.array
            The time axis.
        ax : matplotlib.axes, optional
            The axes on which plot the curve. The default is None.
        """
        _curve = self.control_parameters[label]
        if callable(_curve):
            ys = _curve(time)
        else:
            ys = _curve * np.ones_like(time)
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
        ax.plot(time, ys)
        ax.set_xlabel('Time')
        ax.set_ylabel(label)

    def plot_controls(self, time):
        """
        Plot all the controle curves of the player.

        Parameters
        ----------
        time : np.array
            The time axis.
        """
        title = self.excitator_type
        if title.lower() == 'reed1dof':
            title += ', ' + self.control_parameters['model']
        curves = [p for p in self.control_parameters.keys() - {"model"}]
        n_row = len(curves) + 1
        fig = plt.figure()
        ax = [fig.add_subplot(n_row, 1, 1)]
        self.score.plot_score(time, ax[0])
        for k, label in enumerate(curves):
            ax.append(fig.add_subplot(n_row, 1, k+2, sharex=ax[0]))
            self.plot_one_control(label, time, ax[k+1])
        fig.suptitle(title)

    def __repr__(self):
        msg = 'dict_key={'
        msg += "'excitator_type': {}".format(self.excitator_type)
        excitator_labels = [p for p in self.control_parameters.keys()]
        for label in excitator_labels:
            msg += ", '{}': {}".format(label, self.control_parameters[label])
        msg += "}"
        return "<openwind.Player({}, {})>".format(msg, repr(self.score))

    def __str__(self):
        msg = "Player:\n"
        msg += "\texcitator type: {}\n".format(self.excitator_type)
        msg += "\tcontrol parameters:\n"
        excitator_labels = [p for p in self.control_parameters.keys()]
        for label in excitator_labels:
            msg += "\t\t{}: {}\n".format(label, self.control_parameters[label])
        msg += str(self.score)
        return msg

    def display(self):
        print(repr(self))
