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
Create an animation based on the results of a simulation.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from openwind.temporal.utils import export_mono


def linlin(x, x1, x2, y1, y2):
    # Linear scaling
    return y1 + (y2-y1) * (x-x1)/(x2-x1)


def pv_to_rgb(p, v, pmax, vmax):
    # Convert pressure and flow to an RGB color
    p255 = int(linlin(p, -pmax, pmax, 0, 255))
    v255 = int(linlin(v, -vmax, vmax, 0, 255))
    # Use p for the red channel and v for the green channel
    assert 0 <= p255 and p255 < 256 and 0 <= v255 and v255 < 256
    return f"#{p255:02x}{v255:02x}00"

def ensure_directory_exists(directory_name):
    # Get the current working directory
    current_directory = os.getcwd()
    # Create the full path to the subdirectory
    subdirectory_path = os.path.join(current_directory, directory_name)
    # Check if the subdirectory already exists
    if os.path.exists(subdirectory_path):
        # Check if it's a directory
        if os.path.isdir(subdirectory_path):
            pass # The subdirectory '{directory_name}' already exists, no problem
        else:
            raise FileExistsError(f"'{directory_name}' already exists as a file.")
    else:
        # Create the subdirectory if it doesn't exist
        os.mkdir(subdirectory_path)
        print(f"Created subdirectory '{directory_name}'.")



class SimuAnimation:

    def __init__(self, instrument_geometry, rec, name="simu",
                 signal_key='bell_radiation_pressure'):
        self.ig = instrument_geometry
        self.rec = rec  # simulation result
        self.name = name
        self.signal_key = signal_key
        ensure_directory_exists(name)

    def save_geom(self):
        # Plot the geometry of the instrument
        fig = plt.figure(figsize=(5, 2))
        self.ig.plot_InstrumentGeometry(figure=fig, double_plot=False)
        plt.tight_layout()
        plt.minorticks_on()
        plt.grid(True, 'minor', alpha=0.3)
        plt.grid(True, 'major')
        # Save the figure to a .png image
        plt.savefig(f"{self.name}/geom.png", dpi=300)

    def save_signal_plot(self):
        signal = self.rec.values[self.signal_key]
        plt.figure(figsize=(4, 2.6))
        plt.plot(self.rec.ts, signal, linewidth=0.7)
        plt.xlabel("Time $t$ (s)")
        plt.ylabel("Pressure (Pa)")
        plt.minorticks_on()
        plt.grid(True, 'minor', alpha=0.3)
        plt.grid(True, 'major')
        plt.tight_layout()
        # Save the figure to a .png image
        plt.savefig(f"{self.name}/signal_plot.png", dpi=300)

    def save_sound(self):
        # Extract the signal recorded at the bell
        signal = self.rec.values[self.signal_key]
        # Write it to an audio file
        export_mono(f'{self.name}/signal.wav', signal, self.rec.ts)

    def save_frames(self, fps=60):
        self.fps = fps
        rec = self.rec
        duration = rec.ts[-1]
        self.nframes = int(duration * fps)

        self.calc_pmax_vmax()
        for k in range(self.nframes):
            print(f"Exporting frame {k}/{self.nframes}...")

            # Let the axes fill the whole figure
            fig = plt.figure(k+100, figsize=(5, 2))
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_aspect('equal', adjustable='datalim')

            PP, VV = self._get_PP_VV(k)
            pipe_data = self._get_pipe_data(PP, VV)
            # Display the data corresponding to each pipe using colored polygons
            self.plot_pipe_data(pipe_data)

            plt.savefig(f"{self.name}/{k}.png", dpi=300)
            plt.close(fig)

    def _find_hole(self, label):
        for hole in self.ig.holes:
            if hole.label == label:
                return hole
        raise ValueError(f"Label not found: {label}")

    def _get_PP_VV(self, k):
        # k : frame number (int between 0 and nframes-1)
        t = k/self.fps  # Time of the frame
        # Determine the data index corresponding to this frame
        idx = np.searchsorted(self.rec.ts, t)
        # Fetch the pressure and flow values for each pipe
        PP = self.rec.f['P_interp'][idx, :]
        VV = self.rec.f['V_interp'][idx, :]
        return PP, VV

    def calc_pmax_vmax(self):
        pmax = 0
        vmax = 0
        for k in range(self.nframes):
            PP, VV = self._get_PP_VV(k)
            pmax = max(pmax, max(abs(PP)))
            vmax = max(vmax, max(abs(VV)))
        self.pmax = pmax
        self.vmax = vmax

    def _get_pipe_data(self, PP, VV):
        """Obtain lists of pipe position and pressure/flow data from the objects
        and the current slice of the pressure/flow data.

        The lists contain one entry per pipe.
        """
        pos = 0  # Current position in the concatenated array
        xx_pipes = []
        pp_pipes = []
        vv_pipes = []
        rr_pipes = []
        x0_holes = []
        for t_pipe in self.rec.t_solver.t_pipes:
            pipe = t_pipe.pipe
            nL2 = t_pipe.nL2
            xL2 = t_pipe.mesh.get_xL2()
            x0, _ = pipe.get_endpoints_position_value()
            length = pipe.get_length()
            pp_pipes.append(PP[pos:pos+nL2])
            vv_pipes.append(VV[pos:pos+nL2])
            rr_pipes.append(pipe.get_radius_at(xL2))

            if not pipe.on_main_bore:
                # For toneholes, xx contains the y values
                # Determine the actual position of the hole
                # (otherwise x0 is 0.0 for holes!)
                hole = self._find_hole(pipe.label)
                x0 = hole.position.get_value()
                # Determine the offset from the tube radius
                r0 = self.ig.get_main_bore_radius_at(x0)
                xx_pipes.append(r0 + xL2 * length)
                x0_holes.append(x0)
            else:
                xx_pipes.append(x0 + xL2 * length)
                x0_holes.append(None)
            pos += nL2
        return (xx_pipes, pp_pipes, vv_pipes, rr_pipes, x0_holes)

    def plot_pipe_data(self, pipe_data):
        """Plot filled polygons from pipe data"""
        for (xx, pp, vv, rr, x0) in zip(*pipe_data):
            for i in range(len(xx)-1):
                # Plot a filled polygon
                if x0 is None:
                    # Main bore
                    xplt = [xx[i], xx[i+1], xx[i+1], xx[i]]
                    yplt = [-rr[i], -rr[i+1], rr[i+1], rr[i]]
                else:
                    xplt = [x0-rr[i], x0-rr[i+1], x0+rr[i+1], x0+rr[i]]
                    yplt = [xx[i], xx[i+1], xx[i+1], xx[i]]
                col = pv_to_rgb(pp[i], vv[i], self.pmax, self.vmax)
                plt.fill(xplt, yplt, color=col)
