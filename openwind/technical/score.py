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

import numpy as np
import matplotlib.pyplot as plt


def transition(t, t_0, t_1):
    """
    Transition function between 0 and 1

    Parameters
    ----------
    t : np.array
        The time axis
    t_0 : float
        The instant at which starts the transition (return 0)
    t_1 : float
        The instant at which ends the transition (return 1)

    Return
    ------
    array
        The ponderation (within 0 and 1) function at each instant
    """
    return (t - t_0)/(t_1 - t_0)


class Score:
    """
    Score is defined by note events (itself being a note + time when it is
    played) and a transition_duration

    Parameters
    ----------
    note_events : List[Tuple(String, Float)]
        the list of note events (note_name, time) describing the score to play
    transition_duration : float (optional)
        the caracteristic time to change fingering from one note to another.
        The default value is 0.02 seconde.
    """

    def __init__(self, note_events=[], transition_duration=0.02):
        self._transition_duration = transition_duration
        # Sort events by time
        note_events = sorted(note_events, key=(lambda evt: evt[1]))
        self._notes = [evt[0] for evt in note_events]
        self._times = np.array([evt[1] for evt in note_events])
        self.__set_enveloppes()

    def __repr__(self):
        note_events = [(n, t) for n, t in zip(self._notes, self._times)]
        return ("<openwind.technical.Score(note_events={},"
                " transition_duration={})>".format(note_events,
                                                  self._transition_duration))

    def __str__(self):
        note_events = [(n, t) for n, t in zip(self._notes, self._times)]
        return ("Score:\n\tNote Events: {}\n\t"
                "Transition Duration: {}".format(note_events,
                                                 self._transition_duration))


    def is_score(self):
        """
        Are they notes or not?

        Returns
        -------
        bool

        """
        return len(self._notes) > 0

    def get_all_notes(self):
        """
        Get the names of all the notes called during the score.

        Returns
        -------
        List[String]
            List of the note names.

        """
        return self.__all_notes

    def set_note_events(self, note_events):
        """
        Modify the score.

        Parameters
        ----------
        note_events : List[Tuple(String, Float)]
            the list of note events (note_name, time) describing the new score

        """
        note_events = sorted(note_events, key=(lambda evt: evt[1]))
        self._notes = [evt[0] for evt in note_events]
        self._times = np.array([evt[1] for evt in note_events])
        self.__set_enveloppes()

    def set_transition_duration(self, transition_duration):
        """
        Modify the transition duration between notes.

        Parameters
        ----------
        transition_duration : float (optional)
            the caracteristic time to change fingering from one to another
        """
        self._transition_duration = transition_duration
        self.__set_enveloppes()

    def __activation_note(self, occurences):
        """
        The activation function of a given notes.

        Parameters
        ----------
        occurences : list[(float, float)]
            List of occurences of the note specified by tuple with starting and
            ending time

        Returns
        -------
        callable
            The activation envelop of the note with respect to time.

        """

        def env(t):
            enveloppe = np.zeros_like(t)
            for occurence in occurences:
                t_start, t_end = occurence
                t_endtrans = t_end - self._transition_duration
                t_starttrans = max(0, t_start - self._transition_duration)
                enveloppe += np.minimum(1, transition(t, t_starttrans, t_start),
                                        out=np.zeros_like(t),
                                        where=np.logical_and(t > t_starttrans,
                                                             t <= t_end))
                enveloppe -= (transition(t, t_endtrans, t_end)
                              * np.logical_and(t >= t_endtrans, t <= t_end))
            return enveloppe
        return env

    def __set_enveloppes(self):
        # set the first time to zero and add a last time at infinity
        self.__all_notes = list()
        self.__all_env = list()
        start_times = np.append(self._times, 1e10)
        start_times[0] = 1e-10
        t_events = list()
        for k, note in enumerate(self._notes):
            if note in self.__all_notes:
                index = self.__all_notes.index(note)
                t_events[index].append((start_times[k], start_times[k+1]))
            else:
                self.__all_notes.append(note)
                t_events.append([(start_times[k], start_times[k+1])])
        for event in t_events:
            self.__all_env.append(self.__activation_note(event))

    def plot_score(self, time, ax=None):
        """
        Plot the score trough the activation of all the notes at each time.

        Parameters
        ----------
        time : float, np.array
            The time vector at which plot the notes activations.
        """
        if not ax:
            fig = plt.figure()
            ax = fig.add_subplot()
        for k, note in enumerate(self.__all_notes):
            ax.plot(time, self.__all_env[k](time), label=note)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Notes Activations')

    def get_notes_at_time(self, time):
        """
        Get the note(s) activated at the considered instant.

        Two notes can be returned if the considered time is during a
        transition

        Parameters
        ----------
        time : float
            The instant at which the note is demanded.

        Returns
        -------
        list[(note_name, proportion)]
            A list of tuples containing the name of the notes (a string) and
            their proportion of activation (a float between 0 and 1).
            The sum of proportion must be 1.

        """
        # i is such that _times[i-1] < t <= _times[i]
        i = np.searchsorted(self._times, time)

        # Before first note and after beginning of the last note,
        # maintain the fingering
        if i == 0:
            return [(self._notes[0], 1)]
        if i >= len(self._times):
            return [(self._notes[-1], 1)]

        end_time = self._times[i]
        beginning_of_transition = end_time - self._transition_duration
        if time < beginning_of_transition:
            return [(self._notes[i-1], 1)]

        proportion = ((time - beginning_of_transition)
                      / (end_time - beginning_of_transition))
        return [(self._notes[i-1], 1-proportion), (self._notes[i], proportion)]

    def get_notes_at_time_deprecated(self, time):
        """
        Get the note(s) activated at the considered instant.

        Two notes can be returned if the considered time is during a
        transition.

        This method is much more longer than `get_notes_at_time` but could be
        used to computed activation "a priori" for several instants.

        Parameters
        ----------
        time : float
            The instant at which the note is demanded.

        Returns
        -------
        list[(note_name, proportion)]
            A list of tuples containing the name of the notes (a string) and
            their proportion of activation (a float between 0 and 1).
            The sum of proportion must be 1.

        """
        # this is very long! 50 times longer than the previous method
        proportions = list()
        for env in self.__all_env:
            proportions.append(env(time))
        return [(self.__all_notes[k], prop) for k, prop in
                enumerate(proportions) if prop > 0]
