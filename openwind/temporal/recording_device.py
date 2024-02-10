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

"""Record selected data during time-domain simulation."""

from collections import defaultdict
from datetime import datetime

import numpy as np

import h5py



def format_hdf5_filename(filename):
    if filename == None:
        date_str = f'{datetime.now():%Y%m%d-%H%M%S}'
        return f"tmp_{date_str}.hdf5"
    return filename

class RecordingDevice:
    """
    Record simulation data throughout the instrument.

    Only the physical quantities at the "exit" of the instrument (entrance,
    holes, bell, ...) can be recorded and reused directly.

    Interpolated data "P_interp", "V_interp" and "gradP_interp" is written
    to disk into a HDF5 file.

    A typical use is:

    .. code-block:: python

        rec = simulate(...)  # Returns a RecordingDevice
        bell_pressure = rec.values['bell_radiation_pressure']

    See Also
    --------
    :py:meth:`simulate()<openwind.temporal_simulation.simulate>`
        A method involving :py:class:`RecordingDevice`
    :py:class:`TemporalComponentExit<openwind.temporal.tcomponent.TemporalComponentExit>`
        The component which can be recorded

    Parameters
    ----------
    record_energy: bool
        If True, record also the energy at the exit. Defaults to False.

    Attributes
    ----------
    ts: list
        The instants at which the recording is done
    values: dict
        A dictionnary associating the recording value to each radiating end.
        The keys are the label of the radiating object and an indication on
        the quantity, for example: `'bell_radiation_pressure'` for the bell
    dt: float
        The time step of the recorded temporal simulation.
    t_solver: :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
        The temporal solver recorded.

    """

    def __init__(self, record_energy=False, hdf5_file=None):
        self.hdf5_file = format_hdf5_filename(hdf5_file)
        self.ts = []
        self.values = defaultdict(list)
        self.dsets = dict()

        # self.values = defaultdict(list)
        self.dt = None
        self.record_energy = record_energy
        self._stopped = False

    def callback(self, t_solver):
        """
        The method performing the recording along the simulation.

        It must be given as an option when the method
        :py:meth:`TemporalSolver.run_simulation()\
        <openwind.temporal.temporal_solver.TemporalSolver.run_simulation>`
        is used.

        Example
        -------
        .. code-block:: python

            t_solver = TemporalSolver(instru_physics)
            rec = RecordingDevice()
            t_solver.run_simulation(duration, callback=rec.callback)
            rec.stop_recording()

        Parameters
        ----------
        t_solver: :py:class:`TemporalSolver<openwind.temporal.temporal_solver.TemporalSolver>`
            The temporal solver from which we want to record the values.

        """
        assert not self._stopped
        if not self.dt:
            # First iteration of the callback: we must create all the datasets.
            self.t_solver = t_solver
            self.dt = t_solver.get_dt()
            if t_solver.use_interp:
                self._create_hdf5_file()

        # Recording saves values from time t = dt*(n-1/2)
        self.ts.append(t_solver.get_current_time() - self.dt/2)
        # self.f['ts'][n] = t_solver.get_current_time() - self.dt/2
        for t_comp in t_solver.t_components:
            for name, value in t_comp.get_values_to_record().items():
                # self.f[t_comp.label + "_" + name][n,...] = value
                self.values[t_comp.label + "_" + name].append(value)
            if self.record_energy:
                self._do_record_energies_of(t_comp)

        if t_solver.use_interp:
            P_interp, V_interp, gradP_interp = t_solver.get_current_PVgradP_interp()
            # self.dsets["P_interp"][n,...] = P_interp
            # self.dsets["P_interp"].write_direct(P_interp, dest_sel=np.s_[n,...])
            # self.dsets["V_interp"][n,...] = V_interp
            # self.dsets["V_interp"].write_direct(V_interp, dest_sel=np.s_[n,...])
            # self.dsets["gradP_interp"][n,...] = gradP_interp
            # self.dsets["gradP_interp"].write_direct(gradP_interp, dest_sel=np.s_[n,...])
            self.dsets["P_interp"].write(P_interp)
            self.dsets["V_interp"].write(V_interp)
            self.dsets["gradP_interp"].write(gradP_interp)


    def _create_hdf5_file(self):
        """Go through all the components to find the data to record,
        and create datasets for each.

        This method calculates the data at the first time step
        but does not store it, it is only to know its shape.
        """
        t_solver = self.t_solver
        # scalar dataset for the time step
        # self.f.create_dataset('dt',data=self.dt)
        # # dataset for the time grid
        # self.f.create_dataset('ts',shape=(t_solver.n_steps,), dtype=np.float64)
        # # dataset for every value to record from the components
        # for t_comp in t_solver.t_components:
        #     for name, value in t_comp.get_values_to_record().items():
        #         self._create_dataset_for(t_comp.label + "_" + name, value)
        #     if self.record_energy:
        #         # Create datasets for every energy-related method
        #         for attr in dir(t_comp):
        #             if attr.startswith('energy') and callable(getattr(t_comp, attr)):
        #                 value = getattr(t_comp, attr)()
        #                 short_name = attr.replace('energy', 'E')
        #                 self._create_dataset_for(t_comp.label+"_"+short_name, value)
        #         # Create dataset for the dissipation
        #         q = t_comp.dissipated_last_step()
        #         full_name = t_comp.label+"_Q"
        #         self._create_dataset_for(full_name, q)

        # Create datasets for interpolated pressure, flow and gradient of pressure
        if t_solver.use_interp:
            print(f"Opening file '{self.hdf5_file}' in write mode.")
            self.f = h5py.File(self.hdf5_file, mode="w")

            P_interp, V_interp, gradP_interp = t_solver.get_current_PVgradP_interp()
            self._create_dataset_for("P_interp", P_interp)
            self._create_dataset_for("V_interp", V_interp)
            self._create_dataset_for("gradP_interp", gradP_interp)


    def _create_dataset_for(self, name, value):
        """Create a dataset for a time-varying value, designated by the given name.

        Assumes that "value" is either a scalar or an array,
        # and that its shape does not change during the simulation,
        # so that the first axis of the dataset is of size (n_steps).
        """
        # print(f"_create_dataset_for({name}, {value})")
        dset = self.f.create_dataset(name,
                              shape=(self.t_solver.n_steps,) + np.shape(value),
                              dtype=np.array(value).dtype,
                              chunks=(1,) + np.shape(value))
        self.dsets[name] = BufferedDataset(dset)

    def _do_record_energies_of(self, t_comp):
        n = self.t_solver.cur_step
        # Find all methods starting with 'energy'
        for attr in dir(t_comp):
            if attr.startswith('energy') and callable(getattr(t_comp, attr)):
                value = getattr(t_comp, attr)()
                short_name = attr.replace('energy', 'E')
                # self.f[t_comp.label+"_"+short_name][n,:] = value
                self.values[t_comp.label+"_"+short_name].append(value)
        q = t_comp.dissipated_last_step()
        # self.f[t_comp.label+"_Q"][n,:] = q
        self.values[t_comp.label+"_Q"].append(q)

    def stop_recording(self):
        """
        Notify the device that the simulation is over.

        Converts all recorded values to numpy arrays, so that we can
        manipulate them easily.
        """
        self._stopped = True
        self.ts = np.array(self.ts)
        for key in self.values:
            self.values[key] = np.array(self.values[key])

        # Flush all our buffered datasets
        for bdset in self.dsets.values():
            bdset.flush()

    def __repr__(self):
        return (
            f"<openwind.temporal.RecordingDevice ({'running' if not self._stopped else 'stopped'}, "
            +f"t={self.ts[-1] if len(self.ts) > 0 else 0:.3e}); "
            +f"values.keys()={list(self.values.keys())}; "
            +(f"f.keys()={list(self.f.keys())}" if hasattr(self,'f') else "")
            +">")



class BufferedDataset:
    """Create a buffer to write less frequently to the dataset."""

    def __init__(self, dset, buf_size=1000):
        self.dset = dset
        self.buffer = []
        self.buf_size = buf_size
        self.start_pos = 0 # index of the start of the buffer in the dataset

    def write(self, data):
        """Add a new data entry"""
        self.buffer.append(data)
        if len(self.buffer) >= self.buf_size:
            self.flush()

    def flush(self):
        """Write the buffered data to the dataset"""
        if len(self.buffer) > 0:
            data = np.array(self.buffer)
            # print(data.shape)
            # print(data)
            end_pos = self.start_pos + data.shape[0]
            self.dset[self.start_pos:end_pos,...] = data
            self.start_pos = end_pos
            self.buffer = []
