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
mettre une petite info sur ce que fait ce module ici
"""

import numpy as np
from scipy.sparse import lil_matrix


class FrequentialInterpolation:
    """
    Construct the matrices to interpolate the acoustic fields on the instrument

    Parameters
    ----------
    freq_fem : :py:class:`FrequentialSolver<openwind.frequential.FrequentialSolver>`
        The frequential graph from which perform the interpolation

    pipes_label : str or list of str, optional
            The label of the pipes on which interpolate. The default is "main_bore"
            and correspond to an interpolation on all the pipes of the main bore.

    interp_grid : {float, array(float), 'original', 'radiation'}, optional
            Determine the point on which are computed the interpolated data.
            you can give either

            - a list of points on which to interpolate
            - a float which is the step of the interpolation grid
            - 'original'{Default}, if you want to keep the GaussLobato grid
            - 'radiation' if you want to observe acoustic flow at the radiation opening

    Attributes
    ----------

    x_interp : list(float)
        The x-value of the interpolation points

    interp_mat_L2: scipy.parse.lil_matrix
        The matrix to interpolate L2 variable

    interp_mat_H1: scipy.parse.lil_matrix
        The matrix to interpolate H1 variable

    diff_interp_mat_L2: scipy.parse.lil_matrix
        The matrix to interpolate the gradient of the L2 variable

    diff_interp_mat_H1: scipy.parse.lil_matrix
        The matrix to interpolate the gradient of the H1 variable

    """

    def __init__(self, freq_fem, pipes_label='main_bore', interp_grid='original'):
        self.convention = 'VH1'
        self.n_dof = freq_fem.n_tot

        self.__pipe_identification(freq_fem, pipes_label)

        if isinstance(interp_grid, str) and interp_grid == "radiation":
            self.__build_interpolation_matrix_radiation(freq_fem)
        elif isinstance(interp_grid, str) and interp_grid == "original":
            self.__construct_grid_dof()
            self.__build_interpolation_matrix()
            self.__build_diff_interpolation_matrix()
        else:
            self.__construct_x_grid(interp_grid)
            self.__build_interpolation_matrix()
            self.__build_diff_interpolation_matrix()

    def __pipe_identification(self, freq_fem, pipes_label):
        self.f_pipes = []
        if pipes_label == 'main_bore':
            for f_pipe in freq_fem.f_pipes:
                if hasattr(f_pipe.pipe,'on_main_bore') and f_pipe.pipe.on_main_bore:
                    self.f_pipes.append(f_pipe)
        else:
            for f_pipe in freq_fem.f_pipes:
                if f_pipe.pipe.label in pipes_label:
                    self.f_pipes.append(f_pipe)


    def __construct_x_grid(self, interp_grid):
        if len(self.f_pipes) <= 1:
            total_length = self.f_pipes[0].pipe.get_length()
        else:
            x_max = []
            for f_pipe in self.f_pipes:
                _, xmax = f_pipe.pipe.get_endpoints_position_value()
                x_max.append(xmax)
            total_length = max(x_max)

        if isinstance(interp_grid, float): # if interp is one number it is the step size
            self.x_interp = np.arange(0, total_length, interp_grid)
            if np.isclose(self.x_interp[-1], total_length):
                self.x_interp[-1]= total_length
            else:
                self.x_interp = np.append(self.x_interp, total_length)

        elif len(interp_grid) >= 1:  # if not, it is the coordinate of each points
            self.x_interp = np.array(interp_grid)
            if ((np.max(self.x_interp) > total_length) or
               (np.min(self.x_interp) < 0)):
                self.x_interp = self.x_interp[(self.x_interp >= 0) &
                                              (self.x_interp <= total_length)]
                print(('WARNING: some interpolation points outside the '
                       'designated pipes have been excluded.'))

    def __build_interpolation_matrix_radiation(self, freq_fem):
        from scipy.sparse import lil_matrix
        rad_component = [comp for comp in freq_fem.f_connectors
                         if type(comp).__name__ == 'FrequentialRadiation']
        pipes_end = [comp.freq_end for comp in rad_component]

        freq_pipes = [freq_end.f_pipe for freq_end in pipes_end]

        self.x_interp = np.arange(len(freq_pipes))

        self.interp_mat_L2 = lil_matrix((len(rad_component), self.n_dof))
        self.interp_mat_H1 = lil_matrix((len(rad_component), self.n_dof))

        self.diff_interp_mat_L2 = lil_matrix((len(self.x_interp), self.n_dof))
        self.diff_interp_mat_H1 = lil_matrix((len(self.x_interp), self.n_dof))

        for k, f_pipe in enumerate(freq_pipes):
            x_rad_local = np.array([pipes_end[k].pos.x])
            self.interp_mat_L2[k, :] = f_pipe.place_interp_matrix(x_rad_local)
            self.interp_mat_H1[k, :] = f_pipe.place_interp_matrix(x_rad_local,
                                                               variable='H1')
            # self.diff_interp_mat_L2[k, :] = f_pipe.place_interp_matrix_grad(x_rad_local)
            # self.diff_interp_mat_H1[k, :] = f_pipe.place_interp_matrix_grad(x_rad_local, variable='H1')

    def __construct_grid_dof(self):
        if len(self.f_pipes) == 1:
            f_pipe = self.f_pipes[0]
            self.x_interp = f_pipe.mesh.get_xH1()*f_pipe.pipe.get_length()
        else:
            x_grid = []
            for f_pipe in self.f_pipes:
                xmin, xmax = f_pipe.pipe.get_endpoints_position_value()
                x_grid = np.append(x_grid, f_pipe.mesh.get_xH1()*(xmax - xmin)
                                   + xmin)
            self.x_interp = x_grid[np.append(True, np.diff(x_grid) != 0)]

    def __get_local_x(self, x_local, f_pipe):
        if len(self.f_pipes) == 1:
            return x_local/self.f_pipes[0].pipe.get_length() # WHY????
        else:
            return f_pipe.pipe.get_local_x(x_local)

    def __localize_xinterp_on_pipes(self):
        x_bounds = []
        sorted_func = lambda f_pipe: f_pipe.pipe.get_endpoints_position_value()[0]
        f_pipe_sorted = sorted(self.f_pipes, key=sorted_func)
        for f_pipe in f_pipe_sorted:
            xmin, xmax = f_pipe.pipe.get_endpoints_position_value()
            x_bounds.append(xmin)
        x_bounds.append(xmax)
        ind_pipes = np.searchsorted(x_bounds, self.x_interp) - 1
        ind_pipes[ind_pipes == -1] = 0
        return ind_pipes, f_pipe_sorted

    def __build_interpolation_matrix(self):
        self.interp_mat_L2 = lil_matrix((len(self.x_interp), self.n_dof))
        self.interp_mat_H1 = lil_matrix((len(self.x_interp), self.n_dof))
        ind_pipes, f_pipe_sorted = self.__localize_xinterp_on_pipes()
        for k, f_pipe in enumerate(f_pipe_sorted):
            ind_local = np.nonzero(ind_pipes == k)[0]
            x_local = self.x_interp[ind_pipes == k]
            x_interp_local = self.__get_local_x(x_local, f_pipe)
            self.interp_mat_L2[ind_local, :] = f_pipe.place_interp_matrix(x_interp_local)
            self.interp_mat_H1[ind_local, :] = f_pipe.place_interp_matrix(x_interp_local, variable='H1')

    def interpolate_L2(self, Uh):
        """
        Interpolate the L2 variable from Uh vector

        Parameters
        ----------
        Uh : array of float
            The vector with the value L2 then H1 variable at the dof.

        Returns
        -------
        array of float
            The vector with the interpolated value of the L2 variable.

        """
        return self.interp_mat_L2.dot(Uh)

    def interpolate_H1(self, Uh):
        """
        Interpolate the H1 variable from Uh vector

        Parameters
        ----------
        Uh : array of float
            The vector with the value L2 then H1 variable at the dof.

        Returns
        -------
        array of float
            The vector with the interpolated value of the H1 variable.

        """
        return self.interp_mat_H1.dot(Uh)

    def interpolate_gradH1(self, Uh):
        """
        Interpolate the gradient of the H1 variable from Uh vector

        Parameters
        ----------
        Uh : array of float
            The vector with the value L2 then H1 variable at the dof.

        Returns
        -------
        array of float
            The vector with the interpolated value of the gradient of the H1
            variable.

        """
        return self.diff_interp_mat_H1.dot(Uh)

#%%
    def __get_diff_local_x(self, x_local, f_pipe, diff_index):
        if len(self.f_pipes) == 1:
            xmin = 0
            dxmin = 0
            xmax = self.f_pipes[0].pipe.get_length()
            dxmax = self.f_pipes[0].pipe.get_diff_length(diff_index)
        else:
            xmin, xmax = f_pipe.pipe.get_endpoints_position_value()
            dxmin, dxmax = f_pipe.pipe.get_diff_endpoints_position(diff_index)
        return ((x_local - xmax)*dxmin - (x_local - xmin)*dxmax)/(xmax - xmin)**2

    def __build_diff_interpolation_matrix(self):
        self.diff_interp_mat_L2 = lil_matrix((len(self.x_interp), self.n_dof))
        self.diff_interp_mat_H1 = lil_matrix((len(self.x_interp), self.n_dof))
        ind_pipes, f_pipe_sorted = self.__localize_xinterp_on_pipes()
        for k, f_pipe in enumerate(f_pipe_sorted):
            ind_local = np.nonzero(ind_pipes == k)[0]
            x_local = self.x_interp[ind_pipes == k]
            x_interp_local = self.__get_local_x(x_local, f_pipe)
            self.diff_interp_mat_L2[ind_local, :] = f_pipe.place_interp_matrix_grad(x_interp_local)
            self.diff_interp_mat_H1[ind_local, :] = f_pipe.place_interp_matrix_grad(x_interp_local, variable='H1')

    def diff_interpolate_L2(self, Uh, diff_index):
        """
        Interpolate the derivative of the L2 variable wr to one design variable

        Parameters
        ----------
        Uh : array of float
            The vector with the value L2 then H1 variable at the dof.
        diff_index : int
            The index of the design parameter to which differentiate, such as
            stocked in :py:class:`OptimizationParameters\
            <openwind.design.design_parameter.OptimizationParameters>`

        Returns
        -------
        array
            The vector with the interpolated value of the gradient of the L2
            variable.

        """

        diff_interp_L2 = lil_matrix((len(self.x_interp), self.n_dof))
        # diff_interp_L2 += self.diff_interp_mat_L2
        ind_pipes, f_pipe_sorted = self.__localize_xinterp_on_pipes()
        for k, f_pipe in enumerate(f_pipe_sorted):
            ind_local = np.nonzero(ind_pipes == k)[0]
            x_local = self.x_interp[ind_pipes == k]
            d_xinterp = self.__get_diff_local_x(x_local, f_pipe, diff_index)
            diff_interp_L2[ind_local, :] = self.diff_interp_mat_L2[ind_local, :].multiply(d_xinterp[:, np.newaxis])
        return diff_interp_L2.dot(Uh)

    def diff_interpolate_H1(self, Uh, diff_index):
        """
        Interpolate the derivative of the H1 variable wr to one design variable

        Parameters
        ----------
        Uh : array of float
            The vector with the value L2 then H1 variable at the dof.
        diff_index : int
            The index of the design parameter to which differentiate, such as
            stocked in :py:class:`OptimizationParameters\
            <openwind.design.design_parameter.OptimizationParameters>`

        Returns
        -------
        array
            The vector with the interpolated value of the gradient of the H1
            variable.

        """
        diff_interp_H1 = lil_matrix((len(self.x_interp), self.n_dof))
        # diff_interp_H1 += self.diff_interp_mat_H1
        ind_pipes, f_pipe_sorted = self.__localize_xinterp_on_pipes()
        for k, f_pipe in enumerate(f_pipe_sorted):
            ind_local = np.nonzero(ind_pipes == k)[0]
            x_local = self.x_interp[ind_pipes == k]
            d_xinterp = self.__get_diff_local_x(x_local, f_pipe, diff_index)
            diff_interp_H1[ind_local, :] = self.diff_interp_mat_H1[ind_local, :].multiply(d_xinterp[:, np.newaxis])
        return diff_interp_H1.dot(Uh)
