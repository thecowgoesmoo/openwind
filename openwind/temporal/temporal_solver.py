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
Module for TemporalSolver class.
"""

from openwind.continuous import (ThermoviscousLossless,
                                 ThermoviscousDiffusiveRepresentation,
                                 ParametricRoughness,
                                 RadiationPade,
                                 PhysicalRadiation,
                                 JunctionTjoint, JunctionSimple, JunctionSwitch,
                                 JunctionDiscontinuity,
                                 Reed1dof, Reed1dof_Scaled, Flow, Flute,
                                 RadiationPerfectlyOpen,
                                 Tonehole,
                                 )
from openwind.temporal import (TemporalPipe, TemporalRadiation,
                               TemporalJunction, TemporalLossyPipe,
                               TemporalSimpleJunction,
                               TemporalJunctionSwitch,
                               TemporalJunctionDiscontinuity,
                               TemporalFlowCondition,
                               TemporalPressureCondition,
                               TemporalRoughPipe,
                               TemporalTonehole,
                               TemporalReed1dofScaled, TemporalReed1dof,
			       TemporalFlute,
                               )
from openwind.temporal import ExecuteScore
from openwind.tracker import SimulationTracker
import numpy as np


class TemporalSolver:
    """Prepare an instrument for simulation in time domain.

    This class is responsible for converting a continuous instrument model
    (:py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`)
    to objects that can run the numerical schemes, and for running said scheme.

    Parameters
    ----------
    instru_physics : :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`
        Description of the instrument.
    cfl_alpha: float, optional
        Coefficient used to guarantee the respect of the CFL condition.
        `dt` will be set to `cfl_alpha * dt_max` or slightly lower. It must be
        in ]0,1]. Default is 0.9.
    theta_scheme_parameter : float, optional
        Coefficient used in the excitator scheme. See
        :py:class:`TemporalReed1dof <openwind.temporal.treed1dof.TemporalReed1dof>`.
        Default is 0.25.
    contact_quadratization_cst : float, optional
        Constant used in the quadratization of the contact force. See
        :py:class:`TemporalReed1dof <openwind.temporal.treed1dof.TemporalReed1dof>`.
        Default 1
    **discr_params : keywords
        Discretization parameters. See :py:class:`Mesh<openwind.discretization.mesh>`.

    """

    def __init__(self, instru_physics,
                 cfl_alpha=0.9,
                 theta_scheme_parameter=0.25,
                 interp_grid=None,
                 disable_compute_dt=False,
                 contact_quadratization_cst=1,
                 **discr_params):
        assert 0 < cfl_alpha <= 1, "cfl_alpha must be between 0 and 1"

        self.scaling = instru_physics.scaling
        self.instru_physics = instru_physics
        self.discr_params = discr_params
        self.theta_scheme_parameter = theta_scheme_parameter
        self.contact_quadratization_cst = contact_quadratization_cst
        self._dt = None
        self.__convert_temporal_components()

        if not disable_compute_dt:
            self.__compute_dt(cfl_alpha)
        else:
            self._dt = np.inf # Must be changed afterwards!

        self.use_interp = False
        if interp_grid:
            self.use_interp = True
            self.__enable_interpolation(interp_grid)

        fingering_chart = instru_physics.instrument_geometry.fingering_chart
        self._execute_score = ExecuteScore(fingering_chart,  self.t_components)
        self._current_time = 0.0
        self.energy_check = None

        self.tracker = None

    def __repr__(self):
        return ("<openwind.temporal.TemporalSolver:("
                #"\nTemporalSolver(instr_physics, cfl_alpha, theta_scheme_parameter, **discr_params)\n\n" +
                "\n{},".format(repr(self.instru_physics)) +
                "\ndt={},".format(self.get_dt()) +
                "\ntheta_scheme_parameter={},".format(repr(self.theta_scheme_parameter)) +
                # "\ndisc_params={},".format(repr(self.discr_params)) +
                "\nmesh_info:{{{}pipes, elements/pipe:{}, "
                "order/element:{}}}\n)>".format(len(self.t_pipes),
                                                self.get_elements_mesh(),
                                                self.get_orders_mesh()))

    def __str__(self):
        return ("TemporalSolver:\n" + "="*20 +
                "\nInstrument Physics:\n{}\n".format(self.instru_physics.netlist) +"-"*20 +
                "\n{}\n".format(self.instru_physics.player) +"-"*20 +
                "\nTemperature: {}°C\n".format(self.instru_physics.temperature) +"="*20 +
                "\nTheta scheme parameter: {}\n".format(self.theta_scheme_parameter) + "="*20 +
                "\nTime step: {}\n".format(self.get_dt()) + "="*20 +
                "\n" + self.__get_mesh_info())

    def __convert_temporal_components(self):
        netlist = self.instru_physics.netlist
        self.t_pipes, self.t_connectors = \
            netlist.convert_with_structure(self._convert_pipe,
                                           self._convert_connector)
        self.t_components = self.t_connectors + self.t_pipes


    def _convert_pipe(self, pipe):
        """
        Construct an appropriate instance of (a subclass of) TemporalPipe.

        Parameters
        ----------
        pipe : openwind.continuous.Pipe
            Continuous model of the pipe.
        **discr_params : keyword arguments
            Discretization parameters, passed to the TPipe initializer.

        Returns
        -------
        openwind.temporal.TemporalPipe.

        """
        losses = pipe.get_losses()
        # print(losses)
        if isinstance(losses, ThermoviscousDiffusiveRepresentation):
            return TemporalLossyPipe(pipe, t_solver=self, **self.discr_params)
        if isinstance(losses, ParametricRoughness):
            return TemporalRoughPipe(pipe, t_solver=self, **self.discr_params)
        if isinstance(losses, ThermoviscousLossless):
            return TemporalPipe(pipe, t_solver=self, **self.discr_params)
        raise ValueError("Temporal computation only supports "
                          "losses = {False, 'diffrepr'}.")


    def _convert_connector(self, connector, ends):
        """
        Construct the appropriate temporal version of a connector.

        Parameters
        ----------
        connector : :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`
            Continuous model for radiation, junction, or source.
        ends : List[TPipe.End]
            The list of all `TPipe.End`s this connector connects to.

        Returns
        -------
        openwind.temporal.TemporalComponent.

        """

        if isinstance(connector, PhysicalRadiation):
            radiation_model = connector
            if isinstance(radiation_model, RadiationPerfectlyOpen):
                return TemporalPressureCondition(connector, ends,
                                                 t_solver=self)
            elif isinstance(radiation_model, RadiationPade):
                return TemporalRadiation(connector, ends,
                                         t_solver=self)
            else:
                raise ValueError("Radiation models usable in temporal are"
                                 " RadiationPerfectlyOpen or"
                                 " RadiationPade.")

        if isinstance(connector, JunctionTjoint):
            return TemporalJunction(connector, ends, t_solver=self)

        if isinstance(connector, JunctionSimple):
            return TemporalSimpleJunction(connector, ends,
                                          t_solver=self)
        if isinstance(connector, JunctionDiscontinuity):
            return TemporalJunctionDiscontinuity(connector, ends,
                                                 t_solver=self)
        if isinstance(connector, JunctionSwitch):
            return TemporalJunctionSwitch(connector, ends, t_solver=self)

        if isinstance(connector, Reed1dof) or isinstance(connector, Reed1dof_Scaled):
            # return TemporalReed1dof(connector, ends, t_solver=self,
            #                     theta=self.theta_scheme_parameter)
            return TemporalReed1dofScaled(connector, ends, t_solver=self,
                                          theta=self.theta_scheme_parameter,
                                          contact_quadratization_cst = self.contact_quadratization_cst)

        if isinstance(connector, Flow):
            return TemporalFlowCondition(connector, ends, t_solver=self)
        if isinstance(connector, Flute):
            return TemporalFlute(connector, ends, t_solver=self)
        if isinstance(connector, Tonehole):
            return TemporalTonehole(connector, ends, t_solver=self, **self.discr_params)
        raise ValueError("Could not convert %s" % str(connector))



    def __compute_dt(self, cfl_alpha):
        self.cfl_of_components = [(t_comp.label, t_comp.get_maximal_dt())
                                  for t_comp in self.t_components]
        self.cfl_of_components = sorted(self.cfl_of_components,
                                        key=lambda x: x[1])
        _, cfl = self.cfl_of_components[0]
        self._set_dt(cfl_alpha * cfl)

    def _set_dt(self, dt):
        """Change the value of time step dt."""
        self._dt = dt
        for t_pipe in self.t_pipes:
            t_pipe.set_dt(self._dt)
        for t_connector in self.t_connectors:
            t_connector.set_dt(self._dt)

    def __enable_interpolation(self, interp_grid):
        if interp_grid == "original":
            # Select all pipes including toneholes
            self.t_pipes_sorted = self.t_pipes
            for t_pipe in self.t_pipes:
                t_pipe.set_interp("original")
        elif np.isscalar(interp_grid):
            self.__enable_interpolation_uniform(interp_grid)
        else:
            raise ValueError("'interp_grid' must be 'original' or a scalar.")

    def __enable_interpolation_uniform(self, interp_grid):
        """Enables the interpolation using a regularly spaced grid of step 'interp_grid'."""
        # Calculate the positions of the points along the length of the instrument
        # (assumes the instrument starts at x=0)
        total_length = self.instru_physics.instrument_geometry.get_main_bore_length()
        self.x_interp = interp_grid * np.arange(0, np.ceil(total_length / interp_grid))
        # Sort the t_pipes by increasing starting x position
        t_pipes_starts = [(tp, tp.pipe.get_endpoints_position_value()[0])
                          for tp in self.t_pipes if tp.pipe.on_main_bore]
        t_pipes_xstart_sorted = sorted(t_pipes_starts, key=lambda x: x[1])
        self.t_pipes_sorted = [t_pipe for (t_pipe, xmin) in t_pipes_xstart_sorted]
        x_bounds = [v for (tp,v) in t_pipes_starts]
        ind_pipes = np.searchsorted(x_bounds, self.x_interp) - 1
        ind_pipes[ind_pipes == -1] = 0
        for k, t_pipe in enumerate(self.t_pipes_sorted):
            # Construct local x array
            x_interp_k = self.x_interp[ind_pipes == k]
            x_interp_local = t_pipe.pipe.get_local_x(x_interp_k)
            t_pipe.set_interp(x_interp_local)

    def get_dt(self):
        """
        Returns
        -------
        float
            Time step duration.
        """
        return self._dt

    def reset(self):
        """Reset the simulation."""
        for t_component in self.t_components:
            t_component.reset_variables()
        self._current_time = 0.0

    def get_current_time(self):
        """
        Returns
        -------
        float
            Current in-simulation physical time in seconds.
        """
        return self._current_time

    def get_current_PVgradP_interp(self):
        """Concatenate the interpolated pressure on the main bore."""
        flow_scale = self.scaling.get_scaling_flow()
        press_scale = self.scaling.get_scaling_pressure()

        P_interp_loc = [t_pipe.get_P_interp()*press_scale for k, t_pipe in enumerate(self.t_pipes_sorted)]
        V_interp_loc = [t_pipe.get_V_interp()*flow_scale for k, t_pipe in enumerate(self.t_pipes_sorted)]
        gradP_interp_loc = [t_pipe.get_gradP_interp()*press_scale for k, t_pipe in enumerate(self.t_pipes_sorted)]
        return np.concatenate(P_interp_loc), np.concatenate(V_interp_loc), np.concatenate(gradP_interp_loc)


    def one_step(self, check_scheme=False):
        """Perform one time step of the numerical scheme.

        See also
        --------
        :py:meth:`run_simulation()<TemporalSolver.run_simulation>`
        """
        # We consider that during the update, current time is (n+1/2)*dt.
        self._current_time += self._dt/2 * self.scaling.get_time()

        self._execute_score.set_fingering(self._current_time)

        # Update connectors first, and pipes afterwards
        for t_connector in self.t_connectors:
            t_connector.one_step()
        for t_pipe in self.t_pipes:
            t_pipe.one_step(check_scheme)

        self._current_time += self._dt/2 * self.scaling.get_time()

    def energy(self):
        """Calculate total numerical energy stored in instrument.

        Returns
        -------
        float
            The total energy stored
        """
        return sum(t_comp.energy() for t_comp in self.t_components)

    def dissipated_last_step(self):
        """
        Calculate total numerical energy dissipated during last step.

        Returns
        -------
        float
            The total dissipated energy
        """
        return sum(t_comp.dissipated_last_step() for t_comp
                   in self.t_components)

    def run_simulation(self, duration,
                       callback=None,
                       enable_tracker_display=True,
                       energy_check=False,
                       n_steps=None):
        """Run the simulation for a given duration.

        Calculates the number of steps needed, changes dt accordingly,
        and runs the simulation.

        .. warning::
            May reset the variables of the t_components.

        Parameters
        ----------
        duration : float
            Duration of the simulation.
            Final time should be `duration` up to numerical error.
        callback : callable, optional
            A function to call after each step, taking this TemporalSolver
            as an argument.
        enable_tracker_display : bool, optional
            Whether to enable printing information on percentage of progression
            and remaining time. Default is `True`. See
            :py:class:`SimulationTracker<openwind.tracker.SimulationTracker>`.
        energy_check : bool, optional
            Whether to check that the scheme is energy-consistent. More costly.
            See :py:class:`EnergyCheck<EnergyCheck>`. Default is False
        n_steps : int, optional
            If given, forces the simulation to run in exactly `n_steps` steps.
            Fails if that contradicts stability. Default is None.

        See also
        --------
        :py:meth:`run_simulation_steps`:
            To run for a given number of steps instead.
        """

        n_steps_needed = int(np.ceil(duration / self._dt / self.scaling.get_time()))
        if n_steps is not None:
            print("Custom number of steps:", n_steps)
            if n_steps < n_steps_needed:
                # print("WARNING: not enough steps for CFL. Changing to", n_steps_needed)
                print(f"WARNING: {n_steps} is not enough steps for CFL. You should change it to {n_steps_needed} to ensure stability.")
                # n_steps = n_steps_needed
        else:
            n_steps = n_steps_needed
        new_dt = duration / n_steps / self.scaling.get_time()
        self._set_dt(new_dt)  # Change dt so that the simulation lasts exactly `duration`.
        self.run_simulation_steps(n_steps, callback,
                                  enable_tracker_display, energy_check)

    def run_simulation_steps(self, n_steps,
                             callback=None,
                             enable_tracker_display=True,
                             energy_check=False):
        """Run simulation for a given number of steps.

        Does not change dt.

        Parameters
        ----------
        n_steps : int
            The number of steps
        callback : callable, optional
            A function to call after each step, taking this TemporalSolver
            as an argument.
        enable_tracker_display : bool, optional
            Whether to enable printing information on percentage of progression
            and remaining time. Default is `True`. See
            :py:class:`SimulationTracker<openwind.tracker.SimulationTracker>`.
        energy_check : bool, optional
            Whether to check that the scheme is energy-consistent. More costly.
            See :py:class:`EnergyCheck<EnergyCheck>`. Default is False

        See also
        --------
        run_simulation:
            To run for a given duration instead.
        """
        self._execute_score.set_score(self.instru_physics.player.get_score())
        self.instru_physics._update_player()
        self.tracker = SimulationTracker(n_steps, display_enabled=enable_tracker_display)
        if energy_check:
            self.energy_check = EnergyCheck(self)


        #print(self.t_components['bore0'].PV)
        self.n_steps = n_steps
        print("n_steps =", n_steps)
        for cur_step in range(n_steps):
            self.cur_step = cur_step

            self.one_step(check_scheme=True)
            #print(self.t_components['bore0'].PV)


            # All the functions to call after an iteration
            if callback:
                callback(self)
            self.tracker.update()
            if energy_check:
                self.energy_check()

        if self.energy_check:
            self.energy_check.finish()

    def get_lengths_pipes(self):
        """
        Returns
        -------
        list of float
            The length of each pipe (in meter)
        """
        return [t_pipe.pipe.get_length() for t_pipe in self.t_pipes]

    def get_orders_mesh(self):
        """
        Returns
        -------
        list of list of int
            The order of each elements of each pipe
        """
        return [t_pipe.mesh.get_orders().tolist() for t_pipe in self.t_pipes]

    def get_elements_mesh(self):
        """
        Returns
        -------
        list of int
            The number of elements on each pipe.
        """
        return [len(x) for x in self.get_orders_mesh()]

    def __get_mesh_info(self):
        msg = "Mesh info:"
        # msg += '\n\t{:d} degrees of freedom'.format(self.n_tot)
        msg += "\n\tpipes type: {}".format([t for t in self.t_pipes])
        lengths = self.get_lengths_pipes()
        msg += "\n\t{:d} pipes of length: {}".format(len(lengths), lengths)

        # Orders contains one sub-list for each pipe.
        orders = self.get_orders_mesh()
        elem_per_pipe = self.get_elements_mesh()
        msg += ('\n\t{} elements distributed '
                'as: {}'.format(sum(elem_per_pipe), elem_per_pipe))
        msg += '\n\tOrders on each element: {}'.format(orders)
        return msg

    def discretization_infos(self):
        """
        Information of the total mesh used to solve the problem.

        See Also
        --------
        :py:class:`Mesh <openwind.discretization.mesh.Mesh>`

        Returns
        -------
        str
        """
        print(self.__get_mesh_info())



class EnergyCheck:
    """Check the global energy balance of the scheme, by computing energy at every time step.

    Used in run_simulation()

    Examples
    --------
    >>> t_solver.run_simulation(duration=0.1, energy_check=True)

    .. warning::
        Does not take energy sources into account (yet). Will fail if
        a component is a source of energy (for instance a nonzero
        :py:class:`TemporalFlowCondition<openwind.temporal.tflow_condition.TemporalFlowCondition>`).

    Parameters
    ----------
    t_solver : :py:class:`TemporalSolver<TemporalSolver>`
        The associated solver
    """

    def __init__(self, t_solver):
        self.t_solver = t_solver
        self.init_energy = self.prev_energy = self.energy = self.t_solver.energy()
        self.dissipated_total = 0
        print("Initial energy:", self.init_energy)
        self.call_count = 0
        self.max_err = 0
        self.energy_errs = []

    def __call__(self):
        self.call_count += 1
        self.prev_energy = self.energy
        self.energy = self.t_solver.energy()
        dissip_last_step = self.t_solver.dissipated_last_step()
        self.dissipated_total += dissip_last_step

        residual = (self.energy - self.prev_energy + dissip_last_step) \
            / (self.t_solver.get_dt() * self.energy)

        err = abs(residual)
        self.energy_errs.append(residual)

        if err > self.max_err:
            self.max_err = err
        if err > 1:         # When huge error, stop the program
            raise Exception(f"Energy balance failed (very badly)! {err}")

    def finish(self):
        """Finalize the energy check at the end of the simulation.

        Raises
        ------
        Exception
            If the energy balance was not verified, but the error was
            not large enough to raise an Exception earlier.
        """
        print("EnergyCheck was called {} times.".format(self.call_count))
        print("Final values:")
        print("t = {:.3e} ; energy = {} ; dissipated_total = {} ; energy+dissip = {} ; maximal error on energy balance = {}".format(self.t_solver.get_current_time(),
              self.energy, self.dissipated_total, self.energy+self.dissipated_total, self.max_err))
        if self.max_err > 1e-8:  # If there was error, say it in the end
            raise Exception("Energy balance failed! (but not too badly)")
