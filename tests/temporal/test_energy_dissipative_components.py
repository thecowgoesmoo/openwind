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
import unittest

from openwind.technical import InstrumentGeometry, Player
from openwind.continuous import InstrumentPhysics
from openwind.temporal import TemporalSolver, RecordingDevice


class TestEnergyDissipativeComponent(unittest.TestCase):

    def check_dissipative_instrument(self, instru_geom, player=Player("ZERO_FLOW"),
                                     losses=False, radiation='closed', nondim=False):
        instru_phy = InstrumentPhysics(instru_geom, 25, player, losses=losses,
                                       radiation_category=radiation, nondim=nondim)
        t_solver = TemporalSolver(instru_phy, l_ele=1e-1, order=2)

        # Put some energy in the pipes.
        for tpipe in t_solver.t_pipes:
            np.random.seed(0) # Seed every time because order may be unpredictable
            tpipe.add_pressure(np.random.random(tpipe.nH1)*1e3)

        t_solver.one_step()

        n_steps = int(np.ceil(0.01 / t_solver.get_dt()))
        t_solver.run_simulation_steps(n_steps,
                                      energy_check=True,
                                      enable_tracker_display=False)

        self.assertGreater(t_solver.energy_check.dissipated_total, 1e-8,
                           msg='Instrument does not dissipates energy and should!')

    def check_active_instrument(self, instru_geom, player, nondim=False):
        instru_phy = InstrumentPhysics(instru_geom, 25, player, losses=False,
                                       radiation_category='closed', nondim=nondim)
        t_solver = TemporalSolver(instru_phy, l_ele=1e-1, order=2)

        t_solver.one_step()

        n_steps = int(np.ceil(0.01 / t_solver.get_dt()))
        t_solver.run_simulation_steps(n_steps,
                                      energy_check=True,
                                      enable_tracker_display=False)

        self.assertLess(t_solver.energy_check.dissipated_total, -1e-9,
                        msg='Instrument does not create energy and should!')


    def test_pipe_lossy(self):
        print('\n' + '-'*20 + '\nPipe')
        shape = [[0  , 1e-3],
                 [0.2, 1e-3]]
        mm = InstrumentGeometry(shape)
        self.check_dissipative_instrument(mm, losses='diffrepr')

    def test_pipe_rough(self):
        print('\n' + '-'*20 + '\nPipe (rough)')
        shape = [[0  , 1e-3],
                 [0.2, 1e-3]]
        mm = InstrumentGeometry(shape)
        self.check_dissipative_instrument(mm, losses='parametric_roughness 1.0 100e-6 4')

    def test_pipe_rough_junction(self):
        print('\n' + '-'*20 + '\nPipe (rough)')
        shape = [[0  , 0.2, 1e-3, 1e-3, 'linear'],
                 [0.2, 0.4, 2e-3, 2e-3, 'linear']]
        mm = InstrumentGeometry(shape)
        self.check_dissipative_instrument(mm, losses='parametric_roughness 1.0 100e-6 8')

    def test_T_junction(self):
        print('\n' + '-'*20 + '\nT_junction')
        shape = [[0.0, 8e-3],
                 [0.2, 8e-3]]
        holes = [[0.15, 0.03, 3e-3]]
        mm = InstrumentGeometry(shape, holes)
        self.check_dissipative_instrument(mm, losses='diffrepr')

    def test_T_junction_nondim(self):
        print('\n' + '-'*20 + '\nT_junction')
        shape = [[0.0, 8e-3],
                 [0.2, 8e-3]]
        holes = [[0.15, 0.03, 3e-3]]
        mm = InstrumentGeometry(shape, holes)
        self.check_dissipative_instrument(mm, losses='diffrepr', nondim=True)

    def test_radiation(self):
        print('\n' + '-'*20 + '\nRadiation')
        shape = [[0  , 1e-3],
                 [0.2, 1e-3]]
        mm = InstrumentGeometry(shape)
        self.check_dissipative_instrument(mm, radiation='unflanged')

    def test_fing(self):
        print('\n' + '-'*20 + '\nChange Fingering')
        geom = [[0, 0.005], [0.1, 0.005], [0.7, 0.005]]
        holes = [['label', 'position', 'radius', 'chimney', 'reconnection'],
                 ['h1', 0.1, 3e-3, 0.1, 0.15]]
        fing_chart = [['label', 'A', 'B'],
                      ['h1', 'o', 'x']]
        mm = InstrumentGeometry(geom, holes, fing_chart)
        player = Player("ZERO_FLOW", [('B', 0), ('A', 8e-3)], transition_duration=5e-3)
        self.check_dissipative_instrument(mm, player=player, radiation='unflanged')

    def test_inward_reed(self):
        print('\n' + '-'*20 + '\nInward Reed')
        shape = [[0.0, 8e-3], [0.2, 8e-3]]
        reed_properties = {
            "opening":4e-4,
            "mass":3.376e-6,
            "section":14.6e-5,
            "pulsation":23250,
            "dissip":3000,
            "width":1e-2,
            "contact_pulsation":0
            }

        mm = InstrumentGeometry(shape)
        player = Player("WOODWIND_REED")
        player.update_curves(reed_properties)
        self.check_active_instrument(mm, player=player)

    def test_outward_reed(self):
        print('\n' + '-'*20 + '\nOutward Reed')
        shape = [[0.0, 8e-3], [0.2, 8e-3]]
        reed_properties = {"contact_pulsation":0}
        mm = InstrumentGeometry(shape)
        player = Player("LIPS")
        player.update_curves(reed_properties)
        self.check_active_instrument(mm, player=player)

    def test_outward_with_contact(self):
        print('\n' + '-'*20 + '\nOutward Reed with contact')
        player = Player('LIPS')
        player.update_curve('contact_pulsation',300)
        shape = [[0.0, 5e-3],
                 [.6, 5e-3]]
        mm = InstrumentGeometry(shape)
        # print(player)
        self.check_active_instrument(mm, player=player)

    def test_outward_reed_nondim(self):
        print('\n' + '-'*20 + '\nOutward Reed nondim')
        shape = [[0.0, 8e-3], [0.2, 8e-3]]
        reed_properties = {"contact_pulsation":0}
        mm = InstrumentGeometry(shape)
        player = Player("LIPS")
        player.update_curves(reed_properties)
        self.check_active_instrument(mm, player=player, nondim=True)

    def test_flow(self):
        print('\n' + '-'*20 + '\nImpulse flow')
        shape = [[0.0, 8e-3], [0.2, 8e-3]]
        mm = InstrumentGeometry(shape)
        player = Player("IMPULSE_400us")
        self.check_active_instrument(mm, player=player, nondim=True)

    def test_flow_nondim(self):
        print('\n' + '-'*20 + '\nImpulse flow nondim')
        shape = [[0.0, 8e-3], [0.2, 8e-3]]
        mm = InstrumentGeometry(shape)
        player = Player("IMPULSE_400us")
        self.check_active_instrument(mm, player=player, nondim=True)

if __name__ == "__main__":
    unittest.main()
    # suite = unittest.TestSuite()
    # suite.addTest(TestEnergyDissipativeComponent("test_inward_reed"))
    # runner = unittest.TextTestRunner()
    # runner.run(suite)
