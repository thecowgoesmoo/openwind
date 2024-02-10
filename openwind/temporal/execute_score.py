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
Deal the temporal evolution of the fingering.
"""

from openwind.technical import Score


class ExecuteScore:
    """
    Deal the temporal evolution of the radiation condition following a
    fingering chart and a score.

    Parameters
    ----------
    fingering_chart : :py:class:`FingeringChart <openwind.technical.fingering_chart.FingeringChart>`
        The Fingering Chart associated to the played instrument
    t_components : list of :py:class:`TemporalComponent <openwind.temporal.tcomponent.TemporalComponent>`
        The temporal components which can be modified by the fingerings, like
        the radiation of the holes.

    See Also
    --------
    :py:class:`FingeringChart <openwind.technical.fingering_chart.FingeringChart>`
        For more information on the fingering charts.
    :py:class:`Score <openwind.technical.score.Score>`
        For more information on the score.

    """

    def __init__(self, fingering_chart, t_components):
        self.fingering_chart = fingering_chart
        self.t_components = t_components
        self._score = Score()

    def __check_notes(self):
        """
        Check if the score's note names correspond to the chart's ones

        Raises
        ------
        ValueError
            An error if they not correspond.

        """
        fing_notes = self.fingering_chart.all_notes()
        score_notes = self._score.get_all_notes()
        if len(score_notes) > 0 \
           and not any([note in fing_notes for note in score_notes]):
            raise ValueError('The notes of the score must correspond to the '
                             'ones of the fingering chart:\n'
                             'Score:{} \nChart:{}'.format(score_notes,
                                                          fing_notes))

    def set_score(self, score):
        """
        Set or update the score.

        If ``None`` is given, a default score (without note names and therefore
        without change of fingering) is instanciated.

        If an actual score is given (with note names etc), check the
        correspondance between the note names in the score and the ones of the
        fingering chart .

        Parameters
        ----------
        score : :py:class:`Score <openwind.technical.score.Score>` or None
            The new score. If None, set a default score.
        """
        if score is None:
            self._score = Score()
        else:
            self._score = score

        if self._score.is_score():
            self.__check_notes()

    def __get_fingering(self, t):
        """
        Return the fingering at the given instant following the score.

        Parameters
        ----------
        t : float
            The instant at which is read the score.

        Returns
        -------
        :py:class:`Fingering <openwind.technical.fingering_chart.Fingering>`
            The fingering at this instant (eventually a mix between two
            fingerings).

        """
        notes = self._score.get_notes_at_time(t)
        if len(notes) == 1:
            return self.fingering_chart.fingering_of(notes[0][0])
        elif len(notes) == 2:
            proportion = notes[1][1]
            prev_note = self.fingering_chart.fingering_of(notes[0][0])
            next_note = self.fingering_chart.fingering_of(notes[1][0])
            return prev_note.mix(next_note, proportion)
        else:
            raise ValueError('Three notes are played together, it is '
                             'impossible to mix: {}'.format(notes))

    def set_fingering(self, t):
        """
        Set the right fingering at given time following the score.

        Parameters
        ----------
        t : float
            The instant at which is read the score.
        """
        if self._score.is_score():
            fingering = self.__get_fingering(t)
            fingering.apply_to(self.t_components)
