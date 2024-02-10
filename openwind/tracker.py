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

"""User-friendly progress display during a long simulation."""

import time
import datetime

class SimulationTracker:
    """Display computation time data about a simulation as it is ongoing.

    Call update() at the end of every loop to get nice info.

    Parameters
    ----------
    n_steps : int
        The total number of steps of the simulation.
    first_estimate : float, optional
        The time after which to give a first estimate of the remaining time.
    min_delay : float, optional
        The minimal amount of time to wait before printing again.
    display_enabled : bool, optional
        Whether to display the info automatically. Default is True.
    start_now : bool, optional
        Whether to start the clock immediately. Default is True.
        If False, you need to call start().

    """

    def __init__(self, n_steps, first_estimate=1.0, min_delay=5.0,
                 display_enabled=True, start_now=True):

        self._n_steps = n_steps
        self._steps_done = 0
        self._last_percentage_displayed = -1

        self._delay_before_first_estimate = first_estimate
        self._min_delay = min_delay

        self._is_display_enabled = display_enabled
        if start_now:
            self.start()

    def set_n_steps(self, n_steps):
        """
        Modify the total number of steps of the simulation.

        Parameters
        ----------
        n_steps : int
            The new total number of steps of the simulation.
        """
        self._n_steps = n_steps

    def start(self):
        """
        Start and/or initialize the clock.

        """
        self._steps_done = 0
        self._t0 = time.perf_counter()
        self._next_estimate = self._t0 + self._delay_before_first_estimate

        # Display a welcome message as we start
        if self._is_display_enabled:
            self._display_welcome()


    def enable_display(self, enable):
        """
        Modify whether to display the info automatically

        Parameters
        ----------
        display_enabled : bool
            the new option
        """
        self._is_display_enabled = enable

    def update(self):
        """Track the simulation at each step, and display if necessary."""
        self._steps_done += 1
        if self._should_display():
            self.display()

    def _should_display(self):
        if not self._is_display_enabled:
            return False

        # Be sure to display when finished
        if self._steps_done == self._n_steps:
            return True

        # Otherwise never display too soon
        if time.perf_counter() < self._next_estimate:
            return False

        # and check that we never displayed the current percentage
        return self.get_percentage() > self._last_percentage_displayed

    def get_percentage(self):
        """
        Compute the current percentage of progression of the simulation.

        Returns
        -------
        int
        """
        return int(self._steps_done * 100 / self._n_steps)

    def get_running_time(self):
        """
        How long has the simulation run for?

        Returns
        -------
        float

        """
        return time.perf_counter() - self._t0

    def estimate_remaining_time(self):
        """
        Compute an estimate of the remaining time of the simulation,
        assuming that every step takes a similar amount of time.

        Returns
        -------
        float
        """
        return self.get_running_time() * (self._n_steps - self._steps_done) / (self._steps_done)

    @staticmethod
    def _format_time(time_):
        """Format a duration in seconds into a pretty string.

        Puts time_ in ..h..m..s format, omitting hours and/or minutes when
        unnecessary.
        """
        time_ = int(time_)
        if time_ < 60:
            return '{}s'.format(time_)
        if time_ < 3600:
            minutes, seconds = divmod(time_, 60)
            return '{}m{}s'.format(minutes, seconds)

        hours = time_//3600
        minutes, seconds = divmod(time_%3600, 60)
        return '{}h{}m{}s'.format(hours, minutes, seconds)

    def display(self):
        """
        Display the percentage and remaining time of the simulation.

        Automatically called by update().
        """
        if self._steps_done < self._n_steps:
            self._display_running()
        else:
            self._display_finished()

    def _display_running(self):
        percentage = self.get_percentage()
        estimate = self.estimate_remaining_time()
        est_string = SimulationTracker._format_time(int(estimate))

        print("{}% : {} remaining".format(percentage, est_string))

        self._last_percentage_displayed = percentage
        self._next_estimate = time.perf_counter() + self._min_delay

    def _display_welcome(self):
        print("\nStarting simulation! (Current time is {})".format(self._now()))

    @staticmethod
    def _now():
        return datetime.datetime.now().strftime("%H:%M:%S (%d/%m/%Y)")

    def _display_finished(self):
        run_duration = SimulationTracker._format_time(self.get_running_time())
        print("100%! Simulation ran for {}, and stopped at {}.".format(run_duration, self._now()))
