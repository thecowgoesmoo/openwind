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
Created on Tue May 30 17:21:49 2023

@author: alexis
"""

import unittest
import numpy as np
from os import system

from openwind.technical import InstrumentGeometry, Player
from openwind.continuous import InstrumentPhysics
from openwind.temporal import TemporalSolver, RecordingDevice

geom = [[0,   0.123, 2e-3, 10e-3, 'linear'],
        [0.123, 0.2, 2e-3, 10e-3, 'linear']]
instrument = InstrumentGeometry(geom)
player = Player('IMPULSE_400us')

class TestTemporalInterp(unittest.TestCase):

    def test_temporal_interp(self):

        instrument_physics = InstrumentPhysics(instrument, 20, player, False, nondim=False)
        temporalsolver = TemporalSolver(instrument_physics, l_ele=0.01, order=4, interp_grid=0.0162)

        for t_pipe in temporalsolver.t_pipes:
            assert t_pipe.use_interp

        rec = RecordingDevice(record_energy=False, hdf5_file='test_interp.hdf5')
        temporalsolver.run_simulation(0.01, callback=rec.callback)
        rec.stop_recording()
        print(rec)

        p0_nmh = rec.values['source_pressure']
        p0_n = rec.f['P_interp'][:,0]
        mu_p0_nph = (p0_n[:-1] + p0_n[1:])/2
        assert np.max(abs(p0_nmh[1:] - mu_p0_nph)) < 1e-10

        assert rec.f['P_interp'].shape == (len(rec.ts), len(temporalsolver.x_interp))

        system('rm *.hdf5') # clean created files



    def test_temp_inter_nondim(self):
        instrument_physics = InstrumentPhysics(instrument, 20, player, False, nondim=False)
        temporalsolver = TemporalSolver(instrument_physics, l_ele=0.01, order=4, interp_grid=0.0162)
        rec_dim = RecordingDevice(record_energy=False, hdf5_file='dim_test_interp.hdf5')
        temporalsolver.run_simulation(0.005, callback=rec_dim.callback)
        rec_dim.stop_recording()
        p0_dim = rec_dim.f['P_interp'][:,0]

        instrument_physics_nondim = InstrumentPhysics(instrument, 20, player, False, nondim=True)
        temporalsolver_nondim = TemporalSolver(instrument_physics_nondim, l_ele=0.01, order=4, interp_grid=0.0162)
        rec_nondim = RecordingDevice(record_energy=False, hdf5_file='NONdim_test_interp.hdf5')
        temporalsolver_nondim.run_simulation(0.005, callback=rec_nondim.callback)
        rec_nondim.stop_recording()
        p0_nondim = rec_nondim.f['P_interp'][:,0]

        err = np.linalg.norm(p0_nondim - p0_dim)/np.linalg.norm(p0_dim)
        self.assertLess(err, 1e-12, 'The interpolation is different for dim/non-dim computation.')
        # assert np.max(abs(p0_nmh_nondim - p0_nmh_dim)) < 1e-10

        system('rm *.hdf5') # clean created files

if __name__ == "__main__":
    unittest.main()
