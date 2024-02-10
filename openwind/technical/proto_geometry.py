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
import warnings


def ProtoGeometry(start, end, sections, N_subsections, types):
    """
        Easy creation of a proto-geometry to be optimized.

        Sections are fixed elements of geometry, i.e. with fixed positions.
        Subsections are floating elements of geometry, i.e. with variable
        positions.
        All radii are variable.
        Additional parameters (e.g., alpha parameter for Bessel types) are
        variable as well.

        Parameters
        ----------
        start : int, float
            Starting point of the geometry (fixed position).

        end : int, float
            End point of the geometry (fixed position).

        sections : list of integers or list of floats
            Fixed position of different sections.

        N_subsections : list of integers
            Number of floating subsections for each section.

        types : list of strings or lists
            Types of floating subsections.
            Length must match the number of sections, each element of the list
            corresponds to a section.
            If an element is a string, all subsections in corresponding section
            will be of the type descibed by that string ;
            If an element is a list, its length must match the number of
            subsection in the corresponding section.

            Possible types are : 'linear', 'circle', 'exponential', 'bessel',
            'spline' and 'splineX', where X is an integer and describes the
            number of knots in the spline.
            'spline' is equivalent to 'spline3' and describes a spline with a
            starting point/knot, an ending point/knot, and a knot in the
            middle (3 knots in total)




        Returns
        -------
        List describing the main bore of an instrument geometry.

    """

    sec_pos = []

    for ii in range(len(N_subsections)):
        if isinstance(types[ii], list) and len(types[ii]) == N_subsections[ii]:  # type is list, length matches
            pass
        elif isinstance(types[ii], str):
            types[ii] = [types[ii]] * N_subsections[ii]  # type is string
        elif len(types[ii]) == 1:
            types[ii] = types[ii] * N_subsections[ii]  # type is list of length 1
        else:
            warnings.warn('types not understood, proceeding with all linear')
            types[ii] = N_subsections[ii] * ['linear']

        sec_pos.append(np.linspace([start, *sections, end][ii],
                                   [start, *sections, end][ii+1],
                                   N_subsections[ii]+1))


    proto_geom = list()

    for ii in range(len(sec_pos)):
        for jj in range(N_subsections[ii]):

            if types[ii][jj] == 'linear':
                shape_params = ['~0.005', '~0.005', 'linear']  # [r1, r2]
            elif types[ii][jj] == 'circle':
                shape_params = ['~0.005', '~0.005', 'circle', '~-0.02']  # [y1, y2, R]
            elif types[ii][jj] == 'exponential' or types[ii][jj] == 'exp':
                shape_params = ['~0.005', '~0.005', 'exponential']  # [r1, r2]
            elif types[ii][jj] == 'bessel':  # alpha will be variable
                shape_params = ['~0.03', '~0.06', 'bessel', '~1']  # [r1, r2, alpha]
            elif types[ii][jj] == 'spline':  # 3-point spline : [r1, r2, k1, k2]
                shape_params = ['~0.005', '~0.005', 'spline',
                                (str(sec_pos[ii][jj]) +
                                 '<~' + str(np.mean([sec_pos[ii][jj], sec_pos[ii][jj+1]])) +
                                 '<' + str(sec_pos[ii][jj+1])),
                                '~0.005']
            elif types[ii][jj][:6] == 'spline':
                try:
                    int(types[ii][jj][6:])
                except ValueError:
                    warnings.warn('splineX - X not a number -- proceeding with' +
                                  'default spline3')
                    types[ii][jj] = 'spline3'
                shape_params = ['~0.005', '~0.005', 'spline']  # [r1, r2]
                shape_params.extend([str(kk)
                                     for kk in
                                     np.linspace(sec_pos[ii][jj],
                                                 sec_pos[ii][jj+1],
                                                 int(types[ii][jj][6:]))[1:-1]])
                shape_params.extend(['~0.005' for kk in
                                    np.linspace(sec_pos[ii][jj],
                                                 sec_pos[ii][jj+1],
                                                 int(types[ii][jj][6:]))[1:-1]])


            if N_subsections[ii] == 1:
                proto_geom.append([sec_pos[ii][jj],
                                    sec_pos[ii][jj+1],
                                    *shape_params])
            elif types[ii][jj][:6] == 'spline':
                proto_geom.append([sec_pos[ii][jj],
                                    sec_pos[ii][jj+1],
                                    *shape_params])
            elif jj == 0:
                proto_geom.append([sec_pos[ii][jj],
                                    '~' + str(sec_pos[ii][jj+1]),
                                    *shape_params])
            elif jj == N_subsections[ii]-1:
                proto_geom.append(['~' + str(sec_pos[ii][jj]),
                                    sec_pos[ii][jj+1],
                                    *shape_params])

            else:
                proto_geom.append(['~' + str(sec_pos[ii][jj]),
                                    '~' + str(sec_pos[ii][jj+1]),
                                    *shape_params])


    return proto_geom
