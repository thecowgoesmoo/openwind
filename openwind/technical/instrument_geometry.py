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
Parse instrument geometry and fingering into OpenWind classes.
"""

import warnings
import numpy as np
import matplotlib.pyplot as plt

import openwind
from openwind.design import (FixedParameter, OptimizationParameters,
                             VariableParameter, DesignParameter,
                             VariableHolePosition, VariableHoleRadius)
from openwind.design import Spline, Cone, Bessel, Circle, Exponential
from openwind.technical import FingeringChart
from openwind.technical.fingering_chart import tabulate
from openwind.technical import parser


class Hole:
    """
    Hole with its shape and location on the main bore.

    Attributes
    ----------
    shape : :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
        The shape of the chimney.
    position : :py:class:`DesignParameter \
    <openwind.design.design_parameter.DesignParameter>`
        The position of the Hole on the main bore.
    label : string
        The name of the hole.
    """

    def __init__(self, shape, position, label):
        self.shape = shape
        self.position = position
        self.label = label

    def __repr__(self):
        return '<Hole([{}], {}, {})>'.format(self.shape, self.position, self.label)

class BrassValve:
    """
    A valve: the shape of the deviation tube and its location on the main bore.

    Attributes
    ----------
    shape : :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
        The shape of the deviation tube.
    position : :py:class:`DesignParameter \
    <openwind.design.design_parameter.DesignParameter>`
        The position of the valve on the main bore.
    label : string
        The name of the valve.
    reconnection_position : :py:class:`DesignParameter \
    <openwind.design.design_parameter.DesignParameter>`
        The position on the mainbore at which the reconnection of the deviation pipe
        is connected.
    """

    def __init__(self, shape, position, label, reconnection_position):
        self.shape = shape
        self.position = position
        self.label = label
        if reconnection_position.get_value() <= position.get_value():
            raise ValueError("{}: The reconnection of the deviation pipe must be "
                             "placed afterwards its beginning "
                             "({}>{})".format(label, reconnection_position,
                                            position))
        self.reconnection_position = reconnection_position

    def __repr__(self):
        return '<BrassValve([{}], {}, {}, {})>'.format(self.shape, self.position,
                                                       self.label,
                                                       self.reconnection_position)

class InstrumentGeometry:
    """
    Parse instrument geometry and fingering into OpenWind classes.

    Create an instrument with the bore shape, the holes and the valves given
    by the geometry described in the files.

    Parameters
    ----------
    main_bore : str or list
        filename or list of data respecting the file format with the
        main bore geometry
    holes_or_vales : str or list, optional
        filename or list of data respecting the file format, with the
        holes and/or valves geometries. The default is None corresponding to
        an instrument without hole or valve.
    fingering_chart : str or list, optional
        filename or list of data respecting the file format, indicating the
        fingering chart in accordance with the holes and/or valves. The default
        is None corresponding to no fingering (everything open)
    unit: str {'m', 'mm'}, optional
        The unit (meter or millimeter). Default is 'm' (meter)
    diameter: boolean, optional
        If True assume that diameter are given instead of radius. The default is False.
    allow_long_instrument : boolean, optional
        if true, it is possible to simulate long instrument, otherwise an error is
        raised to avoid long computation due to unit mistake (m/mm). Default is False


    **Structure of the files**

    In all of the files, data are separated by whitespace.
    Comments start with a `#` and extend to the end of the line.
    Blank lines are ignored.

    Instead of giving a filename, it is possible (and equivalent) to give
    the list of already-split lines, for instance:

        >>> InstrumentGeometry([[0.0, 2e-3], [0.2, 0.01]])
        <openwind.InstrumentGeometry(1 main bore parts,...)>

    *Header: format options*

    In all files you can indicate in header lines starting with "!",
    the unit (meter or millimeter) and if the file contains radius or diameter.

    .. code-block:: shell

        ! unit = mm
        ! diameter = True

    The available options are:

    - unit : {m, meter, mm, millimeter}
    - diameter :{True, False}

    This header has the priority over the instanciation options (if they are
    different, the options of the files are used)

    *Main bore geometry file*

    The instrument bore is assumed to be aligned along one axis `x`.
    Abscissae `x` are absolute, with `x = 0` corresponding to the
    entrance of the instrument, and increasing `x` along its length.

    Each line describes one section of the instrument,
    which can be of one of several basic shapes.
    Measurements are given in meters.
    The line must be of the form, either:

    - ``x1 x2 r1 r2 type [param...]``, where:

        - ``x1, x2`` are the beginning and end abscissae of the part,
        - ``r1, r2`` are the radii at the beginning and end,
        - ``type`` is one of ``{'linear', 'spline', 'circle', 'exponential', 'bessel'}``
        - ``[param ...]`` are the parameters of the shape, if necessary:

            - radius of the 'circle'
            - ``alpha`` parameter of the 'bessel' function
            - ``x_i... r_i...`` internal points of the 'spline'
    - ``x r``: in which case the radius is assumed to evolve linearly from the\
        last specified point.


    .. code-block:: shell

        # This is my favorite instrument
        # x1    x2     r1      r2     type         param
          0     0.1    0.02    0.03   linear
          0.1   1.2    0.03    0.015  circle       -10   # slightly fat curve
          1.2   2.6    0.015   0.02   exponential
        # x     r
          2.7   2.1e-2   # the bell is smaller than usual



    *Holes and valves file*

    The holes and valves (pistons) dimensions are deal similarly.
    First line of the file contains column titles,
    the following lines contain the data.

    Possible column names are :

    - 'label' (optional): the hole/piston label
    - 'variety' (optional): if it is a hole or a valve (if not indicated, hole is assumed)
    - 'position' (mandatory): the location of the hole/valve on the main bore
    - 'length' (mandatory): the length of the chimney or the deviation pipe
    - 'radius' (mandatory): the radius of the chimney or deviation pipe. Currently only cylindrical pipe are supported
    - 'type' (optional): the shape of the pipe. Currently the only supported 'type' is 'linear'.
    - 'reconnection' (mandatory for valve only): The location of the reconnection of the deviation pipe on the main bore

    If 'label' is provided, each hole will be labeled by the given name.

    .. code-block:: shell

        label       variety position radius  length  reconnection
        #--------------------------------------------------------
        g_hole      hole    0.1      1e-3    0.021   /
        b_flat_hole hole    0.23     4.2e-3  2e-3    /
        piston1     valve   0.31     2.5e-3  0.1    0.32
        ...

    Alternate (old) format for holes : ``x l r type``.
    This format is deprecated and may give inconsistent hole labels.

    *Fingering chart file*

    First line of the file must be: ``label [note_name...]``.
    The following lines are: ``hole_name ['x' or 'o' ...]`` specifying for \
        each note, whether this hole is open ('o') or closed ('x').

    .. code-block:: shell

        label        C1    do    re    fa    open    closed    fork
        #----------------------------------------------------------
        g_hole       x     x     x     x     o       x         o
        b_flat_hole  o     o     o     o     o       x         x
        ...


    **Type of design parameters for the inversion**

    For the inversion, it is necessary to indicate which parameters are fixed,
    and which parameters is variable and can be modified by the algorithm. This
    is done directly in the MainBore and Holes files by adding information on
    the numerical values.

    Several types of variable parameters can be used:

    - '0.01': The numerical value only is a fixed parameters. \
        A :class:`FixedParameter` object is instantiated.
    - '~0.01': The '~' indicates a variable parameters (unlimited range). \
        A :class:`VariableParameter` object is instantiated without bound.
    - '1e-3<~0.01': The range of the variable parameter is limited by a low \
        bound. A :class:`VariableParameter` object is instantiated with lower \
        bound only.
    - '1e-3<~0.01<1e-1': The range of the variable parameter is limited by low\
        and upper bounds. A :class:`VariableParameter` object is instantiated \
        with lower and upper bounds.
    - '~0.01%': Only valid for hole position and hole radius! \
        The parameter is defined relatively to the main bore shape on which \
        the hole is placed. A :py:class:`VariableHolePosition \
        <openwind.design.design_parameter.VariableHolePosition>` or \
        :py:class:`VariableHoleRadius \
        <openwind.design.design_parameter.VariableHoleRadius>` object is instantiated.
        If bounds are given, non-linear constraints are instancied.

    See Also
    --------
    :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` :
        Mother class of design parameters
    :py:class:`DesignShape <openwind.design.design_shape.DesignShape>` :
        Mother class of design shapes


    Attributes
    ----------
    main_bore_shapes: list[ :py:class:`DesignShape \
    <openwind.design.design_shape.DesignShape>` ]
        The description of the main bore
    holes: list[ :py:class:`Hole \
    <openwind.technical.instrument_geometry.Hole>` ]
        The description of the tone holes
    valves: list[ :py:class:`BrassValve\
    <openwind.technical.instrument_geometry.BrassValve>` ]
        The description of the valves and the associated deviation pipe
    fingering_chart: :py:class:`FingeringChart \
    <openwind.technical.fingering_chart.FingeringChart>`
        The fingerings of the various notes.
    optim_params: :py:class:`OptimizationParameters \
    <openwind.design.design_parameter.OptimizationParameters>`
        Used only for inverse problem. Organized the variable parameters which
        are modified during the optimization process.

    """

    def __init__(self, main_bore, holes_valves=list(), fingering_chart=list(),
                 unit='m', diameter=False, allow_long_instrument=False):
        self.optim_params = OptimizationParameters()

        bore_list, hole_list, fing_list, bore_opt, hole_opt = parser.from_files_to_lists(main_bore, holes_valves, fingering_chart)

        self.main_bore_shapes = list()
        bore_opt = self.check_options(bore_opt, unit, diameter, file_type='Main Bore')
        self._create_main_bore_shapes(bore_list, bore_opt)
        if not allow_long_instrument:
            self._check_length()

        self.holes = list()
        self.valves = list()
        hole_opt = self.check_options(hole_opt, unit, diameter, file_type='Holes/Valves')
        self._create_holes_shapes(hole_list, hole_opt)

        self.set_fingering_chart(fing_list)


    def __repr__(self):
        return '<openwind.InstrumentGeometry({} main bore parts, {} holes, ' \
               '{} valves, {})>'.format(len(self.main_bore_shapes),
                                        len(self.holes), len(self.valves),
                                        repr(self.fingering_chart))

    def __str__(self, opt_param=dict()):
        return(('*'*10 + 'Main Bore' + '*'*10 + '\n{} \n'
                + '*'*10 + 'Side Components' + '*'*10 + '\n{} \n\n'
                + '*'*10 + 'Fingering Chart' + '*'*10 + '\n{}\n')
               .format(self.print_main_bore_shape(**opt_param), self.print_side_components(**opt_param),
                       self.fingering_chart))

    def _check_length(self):
        Ltot = self.get_main_bore_length()
        if Ltot>3:
            warnings.warn(f'Your instrument is especially long: {Ltot} meters.'
                          ' Maybe you forgot to specify the unit with the keyword: unit="mm".\n')
        if Ltot>25:
            raise ValueError(f'Your instrument is too long: {Ltot} meters!'
                             ' Maybe you forgot to specify the unit with the keyword: unit="mm".\n')

# %% Constraints

    def constrain_parts_length(self, Lmin=0, Lmax=np.inf):
        """
        Constrain the length of each main bore part to be in the range [Lmin, Lmax]

        Only used for optimization

        .. warning::
            If one shape is a spline, the distances between the nodes have the
            same constraint (Dmin = Lmin/Nnodes)

        Parameters
        ----------
        Lmin : float, optional
            The minimal length of each part (in m). The default is 0.
        Lmax : float, optional
            The maximal length of each part (in m). The default is np.inf.

        """
        for shape in self.main_bore_shapes:
            shape.create_length_constraint(Lmin, Lmax)
            if isinstance(shape, Spline):
                Nnodes = len(shape.X) - 1
                shape.create_nodes_distance_constraints(Lmin/Nnodes, Lmax)

    def constrain_2_holes_distance(self, label_hole1, label_hole2, Lmin=0, Lmax= np.inf, edges=False):
        """
        Add a constraint on the distance between the location of 2 holes.

        The two holes are not necessarily adjacent.

        Parameters
        ----------
        label_hole1 : string
            The label of the first hole.
        label_hole2 : string
            The label of the second hole.
        Lmin : float, optional
            The minimal distance. The default is 0.
        Lmax : float, optional
            The maximal distance. The default is np.inf.
        edges : boolean, optional
            If true the constraint is set to the hole edges, if false, the hole centers.
            The default is False.

        Raises
        ------
        ValueError
            Raises error if the labels do not correspond to actual holes
        """
        holes_labels = self.get_hole_labels()
        index1 = [i for i,label in enumerate(holes_labels) if label==label_hole1]
        if len(index1)<1:
            raise ValueError(f'The hole "{label_hole1}" does not exist: please chose in : {holes_labels}')
        index2 = [i for i,label in enumerate(holes_labels) if label==label_hole2]
        if len(index2)<1:
            raise ValueError(f'The hole "{label_hole2}" does not exist: please chose in : {holes_labels}')
        if edges:
            self._constrain_hole_edges_distance([self.holes[k] for k in [index1[0], index2[0]]], Lmin, Lmax)
        else:
            self._constrain_hole_centers_distance([self.holes[k] for k in [index1[0], index2[0]]], Lmin, Lmax)

    def constrain_all_holes_distance(self, Lmin=0, Lmax=np.inf, edges=False):
        """
        Add constraints on the distance between adjacent holes.

        For N holes, add N-1 constraints with the same bounds.
        If you want to set different bounds, please use the methode:
        :py:meth:`constrain_2_holes_distance() <openwind.technical.instrument_geometry.InstrumentGeometry.constrain_2_holes_distance>`

        Parameters
        ----------
        Lmin : float, optional
            The minimal distance. The default is 0.
        Lmax : float, optional
            The maximal distance. The default is np.inf.
        edges : boolean, optional
            If true the constraint is set to the hole edges, if false, the hole centers.
            The default is False.

        """
        if edges:
            self._constrain_hole_edges_distance(self.holes, Lmin, Lmax)
        else:
            self._constrain_hole_centers_distance(self.holes, Lmin, Lmax)

    def _constrain_hole_centers_distance(self, holes, Lmin, Lmax):
        holes.sort(key=lambda x: x.position.get_value())
        for k in range(len(holes)-1):
            fun = lambda p=k: holes[p+1].position.get_value() - holes[p].position.get_value()
            jac = lambda p=k: holes[p+1].position.get_jacobian() - holes[p].position.get_jacobian()
            self.optim_params.add_nonlinear_constraint(fun, jac, Lmin, Lmax, label=f'Centers Distance ({holes[k].label}, {holes[k+1].label})')

    def _constrain_hole_edges_distance(self, holes, Lmin, Lmax):
        holes.sort(key=lambda x: x.position.get_value())
        radii = [h.shape.get_endpoints_radius()[0] for h in holes]
        for k in range(len(holes)-1):
            fun = lambda p=k: holes[p+1].position.get_value() - holes[p].position.get_value() - radii[p+1].get_value() - radii[p].get_value()
            jac = lambda p=k: holes[p+1].position.get_jacobian() - holes[p].position.get_jacobian() - radii[p+1].get_jacobian() - radii[p].get_jacobian()
            self.optim_params.add_nonlinear_constraint(fun, jac, Lmin, Lmax, label=f'Edges Distance ({holes[k].label}, {holes[k+1].label})')

    # %% creation of lists of shape and fingering
    @staticmethod
    def check_options(geom_options, unit, diameter, file_type=''):
        if 'unit' in geom_options and DesignParameter.UNIT_DICT[geom_options['unit']] != DesignParameter.UNIT_DICT[unit]:
            warnings.warn(file_type + ': data interpreted with option (from file): unit=%s'%geom_options['unit'], stacklevel=2)
        elif 'unit' not in geom_options:
            geom_options['unit'] = unit
        if 'diameter' in geom_options and DesignParameter.DIAMETER_DICT[geom_options['diameter']] != DesignParameter.DIAMETER_DICT[str(diameter)]:
            warnings.warn(file_type + ': data interpreted with option (from file): diameter=%s'%geom_options['diameter'], stacklevel=2)
        elif 'diameter' not in geom_options:
            geom_options['diameter'] = diameter
        return geom_options

    @staticmethod
    def label_remove_space_sharp(label):
        return label.replace(' ','_').replace('#','_sharp').replace('__','_')

    def _create_main_bore_shapes(self, main_bore_list, geom_options):
        """
        Construct the "DesignShapes" of the main bore from the list of shape data.

        The possible format of the data are speciefied in the main docstring.
        The data are interpreted by the method ```_parse_geometry```.

        Parameters
        ----------
        main_bore_list : list
            list of data for each "shape".
        geom_options : dict
            the dictionnary of options for geometric interpretation (unit etc)

        Attributes
        ----------
        main_bore_shapes : list of [ :py:class:`DesignShape \ <openwind.design.design_shape.DesignShape>` ]
            list of DesignShape

        """
        Xlast = None
        Rlast = None
        for k, raw_part in enumerate(main_bore_list):
            label = 'bore' + str(k)
            X, R, shape_type, Geom_param = self._parse_geometry(raw_part, Xlast,
                                                                Rlast, label, **geom_options)

            if Xlast is not None and \
                min([x.get_value() for x in X]) < Xlast.get_value():
                raise ValueError(f"{label}: Some abscissae x are going backwards, "
                                 "there must be a mistake in the instrument"
                                 " file.")
            if min([r.get_value() for r in R]) < 0:
                raise ValueError(f"{label}: Radius must be positive.")

            if shape_type is not None:
                new_shape = self._build_shape(X, R, shape_type, Geom_param, label)
                self.main_bore_shapes.append(new_shape)
            Xlast = X[-1]
            Rlast = R[-1]
        #     if (k==0):
        #         self.Rinput = R[0].get_value()
        # self.ltot = self.main_bore_shapes[-1].get_position_from_xnorm(1)

    def add_side_components(self, holes_valves, unit='m', diameter=False):
        """
        Add holes or valves to the geometry from list of filename

        Parameters
        ----------
        holes_or_vales : str or list, optional
            filename or list of data respecting the file format, with the
            holes and/or valves geometries. The default is None corresponding to
            an instrument without hole or valve.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If True assume that diameter are given instead of radius. The default is False.

        """
        hole_list, hole_opt = parser.interpret_data(holes_valves)
        hole_opt = self.check_options(hole_opt, unit, diameter, file_type='Holes/Valves')
        self._create_holes_shapes(hole_list, hole_opt)

    def _create_holes_shapes(self, holes_list, geom_options):
        """
        Construct the DesignShape of each hole from the data.

        The possible format of the data are speciefied in the main docstring.
        The data are interpreted by the method ```_parse_holes_new_format``` or
        ```_parse_holes_old_format```.

        .. warning::
            The current version accept only one DesignShape by hole

        Parameters
        ----------
        data : list
            list of data for each hole or valve.

        geom_options : dict
            the dictionnary of options for geometric interpretation (unit etc)

        """
        try:
            if len(holes_list) == 0:  # No holes
                return
            if str(holes_list[0][0]).isalpha():
                self._parse_holes_new_format(holes_list, **geom_options)
            else:
                self._parse_holes_old_format(holes_list, **geom_options)
        except (ValueError, IndexError) as err:
            msg = ("\nImpossible to read the holes/valves data. See documentation of InstrumentGeometry.")
            raise ValueError(str(err) + msg).with_traceback(err.__traceback__)


    def set_fingering_chart(self, data):
        """
        Set the :py:attr:`fingering_chart<openwind.technical.InstrumentGeometry.fingering_chart>`

        Parameters
        ----------
        data : list or string or :py:class:`FingeringChart \
        <openwind.technical.fingering_chart.FingeringChart>` or None
            `data` is either direct data, or a filename containing the data.
            If data is None, the default fingering with all holes open
            is assumed
        """
        if type(data) == FingeringChart:
            self.fingering_chart = data
        else:
            self.fingering_chart = self._create_fingering_chart(data)


    def _create_fingering_chart(self, data):
        """
        Construct the :py:class:`FingeringChart \
        <openwind.technical.fingering_chart.FingeringChart>` from the data.

        Parameters
        ----------
        data : list
            The list with the fingering data for each hole.
            If the list is empty, the default fingering with all holes open
            is assumed

        Returns
        -------
        fingering_chart: :py:class:`FingeringChart \
        <openwind.technical.fingering_chart.FingeringChart>`
            The FingeringChart object containing the information on the
            fingerings.

        """
        labels_side_components = self.get_hole_labels() + self.get_valve_labels()

        if len(data)==0:
            return FingeringChart()
        # First line should be 'label'
        # followed by the names of all the notes
        column_titles, remaining_lines = data[0], data[1:]
        if column_titles[0] != 'label':
            raise ValueError('The 1st line, 1st column of Fing. Chart file must be: "label"')
        note_names = [parser.clean_label(l) for l in column_titles[1:]]
        number_side_comp = len(remaining_lines)
        chart = -np.ones((number_side_comp, len(note_names))) # Initialize at -1
        chart_comp_labels = []
        for comp_i, line in enumerate(remaining_lines):
            comp_label = parser.clean_label(line[0])
            if comp_label not in labels_side_components + ['bell', 'entrance']:
                raise ValueError("Side component '{}' was not defined in holes file. Chose between: {}"
                                 .format(comp_label, labels_side_components + ['bell', 'entrance']))
            chart_comp_labels.append(comp_label)

            for note_j, s in enumerate(line[1:]):
                factor = parser.parse_opening_factor(s)
                chart[comp_i, note_j] = factor

        # Check if fingering chart is valid
        if np.any(chart == -1):
            raise ValueError("Invalid fingering chart: missing data.")
        # Check if all side components from holes file have been defined in fingering chart
        missing = [comp for comp in labels_side_components
                   if comp not in chart_comp_labels]
        if missing:
            warnings.warn(('Side components {} missing from fingering chart.\n'
                           'They will be assumed to remain open.').format(missing))

        return FingeringChart(note_names, chart_comp_labels, chart,
                              other_side_comp=missing)



    # %% Creating DesignParameters / DesignShapes

    def __estim_value(self, param, **geom_options):
        """
        Get the numerical value associated to a parameters.

        The numerical value `x` is extracted from `param`, following
        the different format possible (see the class docstring):
            - a float: `x`
            - the string: `'~x'`
            - the string: `'x_min<~x'`
            - the string: `'x_min<~x<x_max'`
            - the string: `'~x%'`

        Parameters
        ----------
        param : float or string
            Float or string containing the parameters value.

        Returns
        -------
        float
            The numerical value of the parameter.

        """
        coef = DesignParameter.get_writing_coef(**geom_options)
        value = parser.interpret_parameter_data(param)[0]
        return value/coef

    def __designparameter(self, param, label, **geom_options):
        """
        Create the DesignParameter associated to the format specified in the
        input parameter (except for holes parameters).

        The subclass of :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
        instanciated depend of the format of `param`. The different format
        possible are (see the class docstring):

        - a float:      `x`                 => FixedParameter
        - the string:   `'~x'`              => VariableParameter without bounds
        - the string:   `'x_min<~x'`        => VariableParameter lower bounds
        - the string:   `'x_min<~x<x_max'` => VariableParameter lower and upper bounds

        Parameters
        ----------
        param : float or string
            Float or string containing the parameters value.
        label : string
            the label of the parameter

        Returns
        -------
        :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
            The DesignParameter object associated to the parameter

        """
        value, variable, relative, bounds = parser.interpret_parameter_data(param)
        if relative:
            raise ValueError(f'{label} = "{param}": only hole radius and position can be defined relatively (wit "%")')
        if variable:
            return VariableParameter(value, self.optim_params, label, bounds,
                                     **geom_options)
        else:
            return FixedParameter(value, label, **geom_options)

    def __localize_hole(self, x, **geom_options):
        """
        Localize the hole on the main bore.

        Estimate the main bore shape on which is placed the hole, from its
        position

        Parameters
        ----------
        x : string, float
            The hole position data.

        Returns
        -------
        :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
            The design shape of the main bore part on which is placed the hole.

        """
        hole_position = self.__estim_value(x, **geom_options)
        main_bore_bound = [shape.get_position_from_xnorm(0)
                           for shape in self.main_bore_shapes]
        main_bore_bound.append(self.main_bore_shapes[-1]
                               .get_position_from_xnorm(1))
        if hole_position > np.max(main_bore_bound):
            raise ValueError('One hole is placed outside the main bore!')
        index_main_bore_shape = np.max([0, np.searchsorted(main_bore_bound,
                                                           hole_position) - 1])
        return self.main_bore_shapes[index_main_bore_shape]

    def __hole_position_designparameter(self, x, main_bore_shape, hole_label, **geom_options):
        """
        Create the adequate DesignParameter for a hole position.

        The subclass of :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
        instanciated depend of the format of `param`. The different format
        possible are (see the class docstring):

        - if '~x%' => VariableHolePosition
        - else : it is treat like other parameters.

        Parameters
        ----------
        x : string, float
            The data corresponding to the hole positon.
        main_bore_shape : openwind.design.design_shape.DesignShape
            The design shape of the main bore part on which is placed the hole.
        hole_label : string
            The hole label.

        Returns
        -------
        :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
            The DesignParameter object associated to the parameter

        """
        label = hole_label + '_position'
        value, variable, relative, bounds = parser.interpret_parameter_data(x)
        if not variable:
            return FixedParameter(value, label, **geom_options)
        elif not relative:
            return VariableParameter(value, self.optim_params, label, bounds,
                                     **geom_options)
        else:
            design_param = VariableHolePosition(value, self.optim_params,
                                                main_bore_shape, label,
                                                **geom_options)
            if any([np.isfinite(b) for b in bounds]):
                design_param.create_value_nonlin_cons(bounds[0], bounds[1])
            return design_param

    def __hole_radius_designparameter(self, r_data, main_bore_shape,
                                      hole_position, hole_label, **geom_options):
        """
        Create the adequate DesignParameter for a hole radius.

        The subclass of :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
        instanciated depend of the format of `r_data`. The different format
        possible are (see the class docstring):

        - if '~x%' => VariableHoleRadius
        - else : it is treat like other parameters.

        Parameters
        ----------
        r_data : string, float
            The data corresponding to the hole radius.
        main_bore_shape : openwind.design.design_shape.DesignShape
            The design shape of the main bore part on which is placed the hole.
        hole_position : openwind.design.design_parameter.DesignParameter
            The design parameter of the hole position
        hole_label : string
            The hole label.

        Returns
        -------
        :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`
            The DesignParameter object associated to the parameter

        """
        label = hole_label + '_radius'

        value, variable, relative, bounds = parser.interpret_parameter_data(r_data)
        if not variable:
            return FixedParameter(value, label, **geom_options)
        elif not relative:
            return VariableParameter(value, self.optim_params, label, bounds,
                                     **geom_options)
        else:
            design_param = VariableHoleRadius(value, self.optim_params,
                                                main_bore_shape, hole_position,
                                                label, **geom_options)
            if any([np.isfinite(b) for b in bounds]):
                design_param.create_value_nonlin_cons(bounds[0], bounds[1])
            return design_param

    def _build_shape(self, X, R, shape_type, Geom_param, label):
        """
        Construct a DesignShape from DesignParamters

        Parameters
        ----------
        X : list of [ openwind.design.design_parameter.DesignParameter ]
            The 2 ends position of the shape.
        R : list of [ openwind.design.design_parameter.DesignParameter ]
            The 2 ends radius of the shape.
        shape_type : string
            The type of the shape.
        Geom_param : list of [ openwind.design.design_parameter.DesignParameter ]
            The eventual supplementary parameters (necessary for Circle,
            Bessel and Spline). Empty list if not necessary


        Returns
        -------
        shape : openwind.design.design_shape.DesignShape
            The design shape of the considered tube.

        """
        shape_type = shape_type.lower()
        if (shape_type == 'linear' or shape_type == '' or shape_type == 'cone'
            or shape_type == 'cylinder'):
            shape = Cone(*(X + R), label=label)
        elif shape_type == 'exponential':
            shape = Exponential(*(X + R), label=label)
        elif shape_type == 'circle':
            shape = Circle(*(X + R + Geom_param), label=label)
        elif shape_type == 'bessel':
            shape = Bessel(*(X + R + Geom_param), label=label)
        elif shape_type == 'spline':
            shape = Spline(*(X + R), label=label)
        else:
            msg = ("The shape '" + shape_type + "' is unknown. Please " +
                   "chose between: 'cone'(default), 'spline', 'circle', " +
                   "'exponential' and 'bessel'")
            raise ValueError(msg)
        return shape


    # %% Parsing geometry data
    def _parse_geometry(self, raw_part, Xlast, Rlast, label, **geom_options):
        """
        Interpret a list of data to design the main bore shape.

        The list of data is treated differently following to format:
            - 'x r': two columns, a conical part from the last ending point
            is created
            - else: the list is treated independently to build a new shape

        Parameters
        ----------
        raw_part : list
            the list of the data.
        Xlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending position.
        Rlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending point radius.
        label : string
            The label of the shape.

        Returns
        -------
        X : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The 2 ends position of the shape.
        R : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The 2 ends radius of the shape.
        shape_type : string
            The type of the shape.
        Geom_param : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The eventual supplementary parameters (necessary for Circle,
            Bessel and Spline).

        """

        if len(raw_part) == 2:  # if the input contains only x and r
            return self._parse_x_r(raw_part, Xlast, Rlast, label, **geom_options)
        else:
            return self._parse_detailed_geometry(raw_part, Xlast, Rlast, label, **geom_options)

    def _parse_x_r(self, raw_part, Xlast, Rlast, label, **geom_options):
        """
        Treat the list of data in the forme: position radius

        In this format each couple of value (position, radius) is used to
        create a conical part from the last ending point position and radius.

        Parameters
        ----------
        raw_part : list
            List of two data.
        Xlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending position.
        Rlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending point radius..
        label : string
            The label of the shape.

        Returns
        -------
        X : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The 2 ends position of the shape.
        R : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The 2 ends radius of the shape.
        shape_type : None
            Here it is the default type: 'linear'.
        Geom_param : []
            For this type no supplementary parameter is needed.

        """
        pos_options = {k: geom_options[k] for k in set(geom_options.keys()) - set({'diameter'})}
        # X, R = [self.__designparameter(value) for value in raw_part]
        X = self.__designparameter(raw_part[0], label+'_pos_plus', **pos_options)
        R = self.__designparameter(raw_part[1], label+'_radius_plus', **geom_options)
        if Xlast is None:  # for the first line only create the design parameter
            return [X], [R], None, []
        elif X.get_value() == Xlast.get_value():  # discontinuity: change only the radius
            return [X], [R], None, []
        else:
            return [Xlast, X], [Rlast, R], 'linear', []

    def _parse_detailed_geometry(self, raw_part, Xlast, Rlast, label, **geom_options):
        """
        Treat the list of data associated to detailed geometry.

        Each list must contains at least four elements in this order:
            - the left end position
            - the right end postion
            - the left end radius
            - the right end radius
        It can also contains the type which is a string chosen between
        {linear, cone, cylinder, exponential, spline, bessel, circle}
        The default type is 'linear' (or 'cone')

        For some shape type other parameters are necessary, added after this
        five first elements.

        .. warning::
            The right end position must correspond to the left end position of
            the last tube (Xlast).

            If the right end radius is equal to the left end radius they are
            treated as a common design parameter, else, a discontinuity of
            section is created.


        Parameters
        ----------
        raw_part : list
            the list of the data (at list 4 elements: len(raw_part>=4)).
        Xlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending position. None for the first shape
        Rlast : :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` or None
            The last ending point radius. None for the first shape
        label : string
            The label of the shape.

        Returns
        -------
        X : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>`]
            The 2 ends position of the shape.
        R : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The 2 ends radius of the shape.
        shape_type : string
            The type of the shape.
        Geom_param : list of [ :py:class:`DesignParameter \
        <openwind.design.design_parameter.DesignParameter>` ]
            The eventual supplementary parameters (necessary for Circle,
            Bessel and Spline).

        """
        pos_options = {k: geom_options[k] for k in set(geom_options.keys()) - set({'diameter'})}
        if Xlast is None:  # For the first line, the first column is the begining of the main bore
            X = [self.__designparameter(raw_part[0], label + '_pos_minus', **pos_options)]
        elif np.isclose(self.__estim_value(raw_part[0], **pos_options), Xlast.get_value()):  # the parts of the main bore must be connected
            X = [Xlast]
        else:
            msg = (f'{label}: The main bore parts are not connected! '
                   'It is impossible to construct the instrument.')
            raise ValueError(msg)

        if (Rlast is None or not
            np.isclose(self.__estim_value(raw_part[2], **geom_options), Rlast.get_value())):  # begining or discontinuity
            R = [self.__designparameter(raw_part[2], label + '_radius_minus', **geom_options)]
        else:
            R = [Rlast]
        shape_type = raw_part[4].lower()
        if shape_type == 'spline':
            if len(raw_part) < 7:
                raise ValueError(f'{label}: "Spline" shape needs at least 2 supplementary parameters.')
            params = raw_part[5:]
            N = len(params)//2
            X.extend([self.__designparameter(value, label + '_spline_x' + str(k), **pos_options)
                      for k, value in enumerate(params[:N])])
            R.extend([self.__designparameter(value, label + '_spline_r' + str(k), **geom_options)
                      for k, value in enumerate(params[N:])])
        X.append(self.__designparameter(raw_part[1], label + '_pos_plus', **pos_options))
        R.append(self.__designparameter(raw_part[3], label + '_radius_plus', **geom_options))
        if shape_type == 'circle' or shape_type == 'bessel':
            if len(raw_part) != 6:
                raise ValueError(f'{label}: "Circle" and "Bessel" shapes need one supplementary parameter.')
            if shape_type == 'circle' : #radius of curvature does not use diameter option
            	Geom_param = [self.__designparameter(raw_part[5], label + '_param', **pos_options)]
            elif shape_type == 'bessel': # bessel coef has no dimension
            	Geom_param = [self.__designparameter(raw_part[5], label + '_param')]
        else:
            Geom_param = []
        return X, R, shape_type, Geom_param

#%% Parsing hole data
    def _parse_holes_old_format(self, raw_holes, **geom_options):
        """
        Treat the old hole file in format: `position chimney radius type`.

        In this file format each columns is supposed to contains in this order:
            - the position of the hole on the main bore
            - the chimney length
            - the radius of the hole
            - the shape type (only linear is accept)

        Parameters
        ----------
        raw_holes : List
            the list of the data list for each hole.

        """
        warnings.warn("`position chimney radius type` hole file format is deprecated. "
                      "Use the file format with column headers instead.",
                      DeprecationWarning)
        for i, raw_hole in enumerate(raw_holes):
            label = "hole{}".format(i+1)
            if len(raw_hole)>3:
                position, chimney, radius, type_ = raw_hole
                self._add_side_component(position, chimney, radius, None, type_, label, **geom_options)
            else:
                position, chimney, radius = raw_hole
                self._add_side_component(position, chimney, radius, None, label=label, **geom_options)

    def _parse_holes_new_format(self, raw_holes, **geom_options):
        """
        Treat the hole file with column headers.

        The first line of this file must be column headers which can be:
            - 'position' or 'x' : the position of the hole on the main pipe (NEEDED)
            - 'radius' or 'r' or 'diameter': the radius/diameter of the hole (NEEDED)
            - 'chimney' or 'l' : the chimney height (NEEDED)
            - 'label' : the label of the hole (NEEDED)
            - 'variety' : the type of side element: hole or valve (optional, default 'hole')
            - 'radius_out' or 'r_out' : the second radius for conical chimney or deviation pipe (optional)
            - 'reconnection' : (NEEDED for valve) the location on the main bore of the reconnection of the deviation pipe
            - 'type' : the shape type of the chimney tube (optional), only
            'linear' is accepted.

        .. warning::
            The header names does not indicate if it is radius or diameter, please
            use instead the option `diameter=True` of :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`,
            or the header of the file with the line `! diameter=True`

        Parameters
        ----------
        raw_holes : List of [List]
            the list of the data list for each hole.


        """
        unifiate_column_name = {'x': 'position', 'position': 'position',
                                'location': 'position',
                                'r': 'radius', 'radius': 'radius', 'diameter':'radius',
                                'l': 'length', 'chimney':'length',
                                'length':'length',
                                'type':'type_', 'label':'label',
				                'radius_out': 'radius_out','r_out': 'radius_out', 'diameter_out': 'radius_out',
                                'variety':'variety', 'reconnection':'reconnection'}
        column_names = list()
        for col in raw_holes[0]:
            column_names += [unifiate_column_name.get(col)]


        mandatory_columns = ['label', 'position', 'length', 'radius']
        if not all(col in column_names for col in mandatory_columns):
            raise ValueError("Hole file must contain columns {}."
                             "See documentation of openwind hole "
                             "files.".format(mandatory_columns))

        for i, line in enumerate(raw_holes[1:]):
            hole_data = dict(zip(column_names, line))
            self._add_side_component(**hole_data, **geom_options)

    def _add_side_component(self, position, length, radius, radius_out=None, type_='linear',
                          label=None, variety='hole', reconnection=None, **geom_options):
        if label is None:
            raise ValueError("Side component (hole or valve) needs a label")
        label = parser.clean_label(label)
        if label in self.get_hole_labels() + self.get_valve_labels():
            raise ValueError("Several side components (hole or valve) were defined with the same label.")

        if variety.lower() == 'hole':
            self._add_hole(position=position, chimney=length, radius=radius,
                          radius_out=radius_out, type_=type_, label=label, **geom_options)
        elif variety.lower() == 'valve':
            self._add_valve(position=position, length=length, radius=radius, radius_out=radius_out,
                            reconnection=reconnection, type_=type_, label=label,
                            **geom_options)
        else:
            raise ValueError('Unknown side component variety, chose between'
                             ' "hole" and "valve"')

    def _add_valve(self, position, length, radius, radius_out=None, reconnection=None, type_='linear',
                   label=None, **geom_options):
        if reconnection is None:
            raise ValueError('Valve needs its reconnection location on the main bore'
                             'in column "reconnection"')
        pos_options = {k: geom_options[k] for k in set(geom_options.keys()) - set({'diameter'})}
        main_bore_entry = self.__localize_hole(position, **pos_options)
        entry_position = self.__hole_position_designparameter(position,
                                                              main_bore_entry,
                                                              label, **pos_options)

        main_bore_reconnection = self.__localize_hole(reconnection, **pos_options)
        reconnection_position = self.__hole_position_designparameter(reconnection,
                                                                main_bore_reconnection,
                                                                label + '_reconnection',
                                                                **pos_options)

        shape_X = [self.__designparameter(0.0, label + '_dev_pipe_entry', **pos_options)]
        shape_X.append(self.__designparameter(length, label + '_length', **pos_options))

        r_entry = self.__hole_radius_designparameter(radius, main_bore_entry,
                                                     entry_position, label, **geom_options)

        if radius_out is None:
            r_reconnection = r_entry
        else:
             r_reconnection = self.__hole_radius_designparameter(radius_out,
								 main_bore_reconnection,
                                                       		 reconnection_position,
								 label + '_reconnection',
	                                                         **geom_options)

        new_shape = self._build_shape(shape_X, [r_entry, r_reconnection], 'linear', [], label + '_shape')
        self.valves.append(BrassValve(new_shape, entry_position, label, reconnection_position))


    def _add_hole(self, position, chimney, radius, radius_out=None,
		  type_='linear', label=None, **geom_options):
        """
        Create a hole from the data.

        Generate the design shape and the design parameters associated to the
        hole corresponding to the specified data.

        Parameters
        ----------
        position : string or float
            The data corresponding to the hole position.
        chimney : string or float
            The data corresponding to the chimney height.
        radius : string or float
            The data corresponding to the hole radius.
        radius_out: string or float, optional
            The data corresponding to the output hole radius. If none,
            cylindrical chimey is assumed. The default is None.
        type_ : string, optional
            The shape type of the chimney tube (only `linear` is accepted).
            The default is 'linear'.
        label : string, optional
            The hole label. Given automatically if None. The default is None.

        Attribute
        --------
        holes: list
            list of the Hole object.

        """
        if type_ not in ['linear', 'cone', 'cylinder']:
            raise ValueError("Only cylindrical holes are implemented (type='linear')")
        if label is None:
            raise ValueError("Hole needs a label")
        if label in self.get_hole_labels():
            raise ValueError("Several holes were defined with the same label.")
        pos_options = {k: geom_options[k] for k in set(geom_options.keys()) - set({'diameter'})}
        main_bore_shape = self.__localize_hole(position, **pos_options)
        # hole_position = self.__designparameter(x, label + '_position')
        hole_position = self.__hole_position_designparameter(position, main_bore_shape,
                                                             label, **pos_options)
        shape_X = [self.__designparameter(0.0, label + '_chimney_start', **pos_options)]
        shape_X.append(self.__designparameter(chimney, label + '_chimney', **pos_options))
        # r_param = self.__designparameter(r, label + '_radius')
        r_param = self.__hole_radius_designparameter(radius, main_bore_shape,
                                                     hole_position, label, **geom_options)
        if radius_out is None:
            rout_param = r_param
        else:
            rout_param = self.__hole_radius_designparameter(radius_out, main_bore_shape,
                                                     hole_position, label + '_out', **geom_options)
        shape_R = [r_param, rout_param]
        new_shape = self._build_shape(shape_X, shape_R, 'linear', [], label + '_shape')
        self.holes.append(Hole(new_shape, hole_position, label))

    # %% gets
    def get_hole_labels(self):
        """
        Returns
        -------
        list
            The labels of holes.

        """
        return [hole.label for hole in self.holes]

    def get_valve_labels(self):
        """
        Returns
        -------
        list
            The labels of the brass valves.
        """
        return [valve.label for valve in self.valves]


    def get_bore_list(self, all_fields=False,  digit=5, unit='m', diameter=False, disp_optim=True):
        """
        Return the list of data of the geometry.

        Parameters
        ----------
        all_fields: bool, optional
            Indicate all the fields for the side components even if they are not
            needed. Default: False
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optim: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.

        Returns
        -------
        list
            The list of the data of the main bore.
        list
            The list of data for the side components.

        """
        opt_param = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        bore_list = list()
        for shape in self.main_bore_shapes:
            bore_list.append(parser.parse_line(shape.__str__(**opt_param)))
        side_list = parser.parse_lines(self.print_side_components(all_fields=all_fields, **opt_param).splitlines())
        return bore_list, side_list

    def get_main_bore_length(self):
        """
        Return the total length of the main bore in meter.

        Returns
        -------
        float
            Total length of the main bore

        """
        mb_length = [shape.get_length() for shape in self.main_bore_shapes]
        return sum(mb_length)

    def get_main_bore_radius_at(self, position):
        """
        Return the radius of the main bore at given position in meter.

        Parameters
        ----------
        position : float or array of float
            Position at which is estimated the radius.

        Returns
        -------
        radius : array of float
            Radius at the given position.

        """
        radius = np.zeros_like(position, dtype=float)
        for shape in self.main_bore_shapes:
            x_norm = np.array(shape.get_xnorm_from_position(position))
            is_in = (x_norm >= 0) & (x_norm <= 1)
            if np.any(is_in):
                radius[is_in] = shape.get_radius_at(x_norm[is_in])
        return radius

    @staticmethod
    def _get_xr_shape(shape, dx_max=1e-3, Nx_min=10):

        if type(shape) is Cone:
            x = np.array([0,1])
        else:
            Nx = int(max(np.ceil(shape.get_length()/dx_max)+1, Nx_min))
            x = np.linspace(0, 1, Nx)
        radius = shape.get_radius_at(x)
        position = shape.get_position_from_xnorm(x)
        return position, radius

    def get_xr_main_bore(self, dx_max=1e-3, Nx_min=10):
        """
        Get 2 vectors: the position x and the radius of the main bore r

        For conical shape, only the 2 boundaries are given for other shape, the
        x values are distances to at most dx_max, and their is at least Nx_min
        points per shape.
        If the jonction between 2 shapes is continuous, the boundaries are not repeated.

        Parameters
        ----------
        dx_max : float, optional
            The max distance between 2 points in non-conical shapes. The default is 1e-3.
        Nx_min : int, optional
            The minimal number of point in non-conical shapes. The default is 10.

        Returns
        -------
        x : np.array of float
            The position vector.
        r : np.array of float
            The radius vector.

        """
        x_mb = list()
        r_mb = list()
        for shape in self.main_bore_shapes:
            x_s, r_s = InstrumentGeometry._get_xr_shape(shape, dx_max, Nx_min)
            x_mb.append(x_s)
            r_mb.append(r_s)
        x = np.concatenate(x_mb)
        r = np.concatenate(r_mb)
        # remove redondant point
        indices_ok = np.append(True,np.logical_or(np.diff(x)>1e-8, np.abs(np.diff(r)/r[1:])>1e-8))
        x = x[indices_ok]
        r = r[indices_ok]
        return x, r


    # %% Modify the geometry

    def shift_x_axis(self, offset):
        """
        Shift the x-axis by the specified offset.

        It shifts the value of all the design parameters related to the x-axis:

        - the bounds of the main bore shapes
        - the positions of the spline nodes
        - the holes' positions

        Parameters
        ----------
        offset : float
            The offset in meter (positive or negative).

        """

        shifted = list()

        bounds_tuple = [shape.get_endpoints_position() for shape in self.main_bore_shapes]
        spline_tuple = [shape.X for shape in self.main_bore_shapes if type(shape) is Spline]
        mb_pos = [pos for bound in bounds_tuple + spline_tuple for pos in bound]
        hole_pos = [hole.position for hole in self.holes]
        valve_pos = [valve.position for valve in self.valves]
        valve_reco = [valve.reconnection_position for valve in self.valves]

        position = hole_pos + mb_pos + valve_pos + valve_reco

        for param in position:
            if param not in shifted:
                if type(param) is FixedParameter:
                    param._value += offset
                else:
                    self.optim_params.values[param.index] += offset
                    self.optim_params.bounds[param.index] = tuple([b + offset for b in self.optim_params.bounds[param.index]])
                shifted.append(param)


    def __add__(self, other):

        """
        Concatenate two InstrumentGeometry.

        The x-axis of the second :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
        is shifted such as its begining correspond to the end of the first one.

        .. warning::
            The addition of InstrumentGeometry does not manage yet the
            Fingering chart!

        To avoid problem with the gestion of the parameters etc, the simpliest
        is to generate the list corresponding ot the new geomtry and instantiate
        a totally new InstrumentGeometry from it.

        Parameters
        -----------
        other : :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
            The second instrument wich must be added dowstream this one

        Returns
        --------
        :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
            The concatenation of the two instruments

        """
        # get the list of the actual geometry
        mb_self, side_self = self.get_bore_list(all_fields=True, digit=15)

        if type(other) is InstrumentGeometry:
            # shift the other geometry such its begin start at the end of self
            starting_other = other.main_bore_shapes[0].get_position_from_xnorm(0)
            offset = self.main_bore_shapes[-1].get_position_from_xnorm(1) - starting_other
            other.shift_x_axis(offset)

            # get the list of the other-shifted instrument
            mb_other, side_other = other.get_bore_list(all_fields=True, digit=15)
            # remove the second head-line
            side_other = side_other[1:]

            # restore the initial offset of the other instrument
            other.shift_x_axis(-offset)

            if len(self.fingering_chart.all_notes() + other.fingering_chart.all_notes()) != 0:
                warnings.warn('The addition of InstrumentGeometry does not manage '
                              'yet the Fingering chart!')
            fing_chart = list()
        elif other == 0: # the addition with 0 resinstanciate the same instrument (necessary to use "sum")
            mb_other = list()
            side_other = list()
            fing_chart = self.fingering_chart
        else:
            raise TypeError("can only concatenate 'InstrumentGeometry' (not '{}') to 'InstrumentGeometry'".format(type(other).__name__))

        # combine and instanciate a new instrument
        return InstrumentGeometry(mb_self + mb_other, side_self + side_other, fing_chart)

    def __radd__(self, other):
        if other == 0:
            return self.__add__(other)
        else:
            return other.__add__(self)

    def extract(self, start, stop=np.inf):
        """
        Extract a part of this instrument between two positions

        Get a new InstrumenteGeometry corresponding to this one, cut between
        the two indicated  position.

        .. warning::
            The excratction of InstrumentGeometry does not manage yet the
            Fingering chart!

        Parameters
        -----------
        start: float
            The position (in meter) of the "left" end of the slice.
            (You can indicate -np.Inf to keep the original end)
        stop: float
            The position (in meter) of the "rigth" end of the slice.
            (You can indicate +np.Inf to keep the original end)

        Returns
        --------
        :py:class:`InstrumentGeometry<openwind.technical.instrument_geometry.InstrumentGeometry>`
            The extracted instrument.

        """
        mb_self, side_self = self.get_bore_list(all_fields=True, digit=15)
        x0_mb = [shape.get_position_from_xnorm(0) for shape
                 in self.main_bore_shapes]
        x1_mb = [shape.get_position_from_xnorm(1) for shape
                 in self.main_bore_shapes]
        # get the shape which must be included in the new instrument
        extract_mb = [part for k, part in enumerate(mb_self) if x0_mb[k]<stop
                    and x1_mb[k]>start]

        # get the holes which must be included in the new isntrument
        if len(side_self)>0:
            x_hole = [hole.position.get_value() for hole in self.holes]
            x_valve = [valve.position.get_value() for valve in self.valves]
            x_reco =  [valve.reconnection_position.get_value() for valve in self.valves]
            if (any(np.logical_and(np.array(x_valve)<stop, np.array(x_reco)>stop))
                or any(np.logical_and(np.array(x_valve)<start, np.array(x_reco)>start))):
                raise ValueError('It is impossible to slice an instrument in between the 2 extremities of a valve.')
            x_pos = sorted(x_hole + x_valve)
            extract_side = [side_self[0]]

            extract_side += [side for k, side in enumerate(side_self[1:])
                           if x_pos[k]<stop and x_pos[k]>=start]
        else:
            extract_side = list()

        if len(self.fingering_chart.all_notes()) != 0:
            warnings.warn('The extracting of InstrumentGeometry does not manage '
                          'yet the Fingering chart!')
        # create a new instrument from the lists
        extract_geom = InstrumentGeometry(extract_mb, extract_side)

        # cut the first shape if needed
        geom_entrance = extract_geom.main_bore_shapes[0]
        if geom_entrance.get_position_from_xnorm(0) < start:
            x_stop = geom_entrance.get_position_from_xnorm(1)
            geom_entrance.cut_shape(start, x_stop)

        # cut the last shape if needed
        geom_end = extract_geom.main_bore_shapes[-1]
        if geom_end.get_position_from_xnorm(1) > stop:
            x_start = geom_end.get_position_from_xnorm(0)
            geom_end.cut_shape(x_start, stop)

        return extract_geom


    # %% Plot
    def plot_InstrumentGeometry(self, figure=None, note=None, double_plot=True, label='_', **kwargs):
        """
        Display the geometry and holes of the instrument.

        If a note name is given, also display the fingering of that note.

        Parameters
        ----------
        figure: matplotlib.figure.Figure, optional
            Which figure to use. By default opens a new figure.
        note: str, optional
            If a note name is given, closed holes are filled, whereas
            open holes are outlined.
            By default all holes are outlined.
        kwargs:
            Additional arguments are passed to the `plt.plot` function.

        """
        mmeter = 1e3

        if not figure:
            fig = plt.figure()
        else:
            fig = figure
        ax = fig.get_axes()

        if self.holes == []  and len(ax) < 2:
            double_plot = False

        if len(ax) < 2 and double_plot:
            ax = [fig.add_subplot(2, 1, 1)]
            ax.append(fig.add_subplot(2, 1, 2, sharex=ax[0]))
        elif len(ax) < 1:
            ax.append(fig.add_subplot(1, 1, 1))

        self._plot_shape(self.main_bore_shapes[0], ax, mmeter, shift_x=0,
                         shift_y=0, label=label, **kwargs)
        for shape in self.main_bore_shapes[1:]:
            self._plot_shape(shape, ax, mmeter, shift_x=0, shift_y=0, **kwargs)

        self._plot_holes(ax, mmeter, note, **kwargs)

        self._plot_valves(ax, mmeter, **kwargs)

        if label != '_':
            ax[0].legend()

        ax[-1].set_xlabel('Position (mm)')
        ax[0].set_ylabel('Radius (mm)')
        ax[0].axis('equal')
        if double_plot:
            ax[1].axis('equal')
            ax[1].set_ylabel('Radius (mm)')

    def _plot_shape(self, shape, ax, mmeter, shift_x=0, shift_y=0, **kwargs):
        x, r = InstrumentGeometry._get_xr_shape(shape)
        radius = np.append(r, np.nan)
        position = np.append(x, np.nan) + shift_x
        line= ax[0].plot(np.append(position, np.flip(position))*mmeter,
                         (np.append(radius, np.flip(-radius)) + shift_y)*mmeter,
                         **kwargs)
        if len(ax)>1:
            ax[1].plot(np.append(position, np.flip(position))*mmeter,
                       (np.append(radius, np.flip(-radius)) + shift_y)*mmeter, **kwargs)
        return line

    def _plot_holes(self, ax, mmeter, note, **kwargs):
        if note:
            fingering = self.fingering_chart.fingering_of(note)
            def plot_or_fill(axes, hole):
                if fingering.is_side_comp_open(hole.label):
                    return axes.plot
                else:
                    return axes.fill
        else:
            def plot_or_fill(axes, hole):
                return axes.plot

        x = np.linspace(0, 1, 10)
        for hole in self.holes:
            position = hole.position.get_value()
            radius = hole.shape.get_radius_at(x)
            chimney = hole.shape.get_position_from_xnorm(x)
            main_bore = self.__localize_hole(position)
            pos_norm = main_bore.get_xnorm_from_position(position)
            main_radius = main_bore.get_radius_at(pos_norm)
            x_plot = position + np.append(radius, np.flip(-radius))
            y_plot = main_radius + np.append(chimney, np.flip(chimney))
            hole_plot = plot_or_fill(ax[0], hole)(x_plot*mmeter, y_plot*mmeter, **kwargs)
            if type(hole_plot[0]).__name__ == 'Polygon':
                hole_plot[0].set_edgecolor(hole_plot[0].get_facecolor())

            if len(ax)>1:
                theta = np.linspace(0,2*np.pi,100)
                hole_plot = plot_or_fill(ax[1], hole)((position + np.mean(radius)*np.cos(theta))*mmeter,
                                                      np.mean(radius)*np.sin(theta)*mmeter, **kwargs)
                if type(hole_plot[0]).__name__ == 'Polygon':
                    hole_plot[0].set_edgecolor(hole_plot[0].get_facecolor())

    def _plot_valves(self, ax, mmeter, **kwargs):
        for valve in self.valves:
            pos = valve.position.get_value()
            reco_pos = valve.reconnection_position.get_value()
            length = valve.shape.get_length()

            shift_x = (pos + reco_pos - length)*.5

            rad = max(valve.shape.get_radius_at(np.linspace(0,1,10)))
            shift_y = ax[0].get_ylim()[0]/mmeter - rad -5e-3
            line = self._plot_shape(valve.shape, ax, mmeter, shift_x, shift_y,
                                    **kwargs)

            main_bore = self.__localize_hole(pos)
            pos_norm = main_bore.get_xnorm_from_position(pos)
            main_radius = main_bore.get_radius_at(pos_norm)

            main_bore_reco = self.__localize_hole(reco_pos)
            pos_norm_reco = main_bore_reco.get_xnorm_from_position(reco_pos)
            main_radius_reco = main_bore_reco.get_radius_at(pos_norm_reco)

            for axe in ax:
                axe.plot(np.array([pos, pos, shift_x, np.nan, shift_x+length, reco_pos, reco_pos])*mmeter,
                         np.array([main_radius, -main_radius, shift_y+rad, np.nan,
                                   shift_y+rad, -main_radius_reco, main_radius_reco])*mmeter,
                         ':', color=line[0].get_color())


    # %% print and write files
    def print_files(self, generic_name, extension='.txt', **kwargs):
        """
        .. deprecated:: 0.6.0
            Replaced by :py:meth:`InstrumentGeometry.write_files()`
        """
        warnings.warn('This method is deprecated, please use write_files instead.')
        self.write_files(generic_name, extension, **kwargs)

    @staticmethod
    def _get_header(unit, diameter):
        return '! version = {}\n! unit = {:s}\n! diameter = {}\n'.format(openwind.__version__, unit, diameter)


    def write_files(self, generic_name, extension='.txt', digit=5, unit='m', diameter=False, disp_optim=True):
        """
        Write the files corresponding to this InstrumentGeometry.

        Write the three files (MainBore, Holes and FingeringChart) associated
        to this instrument.

        Parameters
        ----------
        generic_name : string
            The generic name for the three files which will be named
            "generic_name_MainBore", "generic_name_Holes" and
            "generic_name_FingeringChart".
        extension : string, optional
            The extension used for the filenames. The default is '.txt'.
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optimization: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.
        """
        opt_param = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        if not extension.startswith('.'):
            extension = '.' + extension
        if generic_name.endswith('.txt'):
            generic_name = generic_name[:-4]
        elif generic_name.endswith('.csv'):
            generic_name = generic_name[:-4]
            extension = '.csv'
        # Main Bore
        filename_bore = generic_name + '_MainBore' + extension
        self.write_main_bore_file(filename_bore, xr_format=False, dx_max=1e-3,
                                  Nx_min=10, **opt_param)
        # Side components
        if self.holes or self.valves:
            filename_side = generic_name + '_SideComponents' + extension
            self.write_side_components_file(filename_side, **opt_param)
        # Fingering Chart
        if len(self.fingering_chart.all_notes()) > 0:
            filename_fing = generic_name + '_FingeringChart' + extension
            self.write_fingering_chart_file(filename_fing)

    def write_single_file(self, name, digit=5, unit='m', diameter=False, disp_optim=True, comments=''):
        """
        Write a unique file where everything is indicated (main bore, holes and fing chart)

        Each block of data is separated by a title such as

        .. code-block:: shell

            **********Main Bore**********
            ...
            **********Side Components**********
            ...
            **********Fingering Chart**********
            ...

        Parameters
        ----------
        name : string
            The name of the file where save the geometry.
        extension : string, optional
            The extension used for the filenames. The default is '.txt'.
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optimization: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.
        comments: string, optional
            Free comments added into the file (like instrument description). Default is ''

        """
        header = self._get_header(unit, diameter)

        comments = ''.join(['\n# ' + line for line in comments.splitlines()])

        opt_param = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        filename = name + '.ow'
        data = header + comments + '\n\n' + self.__str__(opt_param)
        file = open(filename, "w")
        file.write(data)
        file.close()


    def print_main_bore_shape(self, digit=5, unit='m', diameter=False, disp_optim=True):
        """
        Print the main bore shape

        Parameters
        ----------
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optim: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.

        Returns
        -------
        msg : str
            The string corresponding to the main bore shape.

        """

        kwarg = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        len_num = digit+2
        header_form = ('#{:>' + str(len_num-1) + 's}\t{:>' + str(len_num) +
                       's}\t{:>' + str(len_num) + 's}\t{:>' + str(len_num) +
                       's}\t{:>11s}\t{:>' + str(len_num) + 's}\n')
        if diameter:
            msg = header_form.format('x0', 'x1', 'D0', 'D1', 'type', 'param')
        else:
            msg = header_form.format('x0', 'x1', 'r0', 'r1', 'type', 'param')
        for shape in self.main_bore_shapes:
            msg += '{}\n'.format(shape.__str__(**kwarg))

        return msg


    def print_main_bore_xr(self, digit=5, unit='m', diameter=False, disp_optim=False, dx_max=1e-3, Nx_min=10):
        """
        Get string with 2 columns (x,r) in indicated format.

        For conical shape, only the 2 boundaries are given for other shape, the
        x values are distances to at most dx_max, and their is at least Nx_min
        points per shape.
        If the jonction between 2 shapes is continuous, the boundaries are not repeated.

        Parameters
        ----------
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optimization: if true display the information related of
            optimisation (bounds, ~, etc). NOT AVAILABLE FOR (X,R) FORMAT!
            The defauflt is False.
        dx_max : float, optional
            The max distance between 2 points in non-conical shapes. The default is 1e-3.
        Nx_min : int, optional
            The minimal number of point in non-conical shapes. The default is 10.

        Returns
        -------
        msg : str

        """
        if disp_optim:
            warnings.warn('When printing in "x,r" format, the optim options are not available.')

        unit_coef = DesignParameter.UNIT_DICT[unit]
        diam_coef = DesignParameter.DIAMETER_DICT[str(diameter)]

        x, r = self.get_xr_main_bore(dx_max, Nx_min)

        num_format = '{:>' + str(digit+5) + '.' + str(digit) + 'f}'
        msg = ''
        for xi, ri in zip(x,r):
            msg += (num_format + '\t' + num_format + '\n').format(xi*unit_coef, ri*unit_coef*diam_coef)

        return msg

    def write_main_bore_file(self, filename, digit=5, unit='m', diameter=False,
                             disp_optim=True,
                             xr_format=False, dx_max=1e-3, Nx_min=10):
        """
        Write main bore file.

        Parameters
        ----------
        filename : str
            The file name (path).
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optimization: if true display the information related of
            optimisation (bounds, ~, etc). NOT AVAILABLE FOR (X,R) FORMAT!
            The defauflt is True.
        xr_format : booelan, optional
            If true, write the main bore file with (x,r) format. The default is False.
        dx_max : float, optional
            For (x,r) format: the max distance between 2 points in non-conical
            shapes. The default is 1e-3.
        Nx_min : int, optional
            For (x,r) format: the minimal number of point in non-conical shapes.
            The default is 10.

        """
        header = self._get_header(unit, diameter)
        opt_param = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        if xr_format:
            msg = self.print_main_bore_xr(dx_max=dx_max, Nx_min=Nx_min, **opt_param)
        else:
            msg = self.print_main_bore_shape(**opt_param)
        f_bore = open(filename, "w")
        f_bore.write(header)
        f_bore.write(msg)

    def print_holes(self, **kwarg):
        return self.print_side_components(**kwarg)

    def print_side_components(self, all_fields=False, digit=5,
                              unit='m', diameter=False, disp_optim=True):
        """
        Print the side components information.

        Parameters
        ----------
        all_fields: bool, optional
            If true, print all the fields (even variety and recoonection,
            if there is no valve). Necessary for the addition of several instrument.
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optim: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.

        Returns
        -------
        msg : str
            The string corresponding to the holes information.

        """
        opt_param = {'digit':digit, 'unit':unit, 'disp_optim':disp_optim}
        if diameter:
            col_names = ['label', 'variety', 'position', 'length', 'diameter', 'diameter_out', 'reconnection']
        else:
            col_names = ['label', 'variety', 'position', 'length', 'radius', 'radius_out', 'reconnection']
        rows = list()
        pos_val = list()
        for hole in self.holes:
            length = hole.shape.get_endpoints_position()[1].__str__(**opt_param)
            radius = hole.shape.get_endpoints_radius()[0].__str__(**opt_param, diameter=diameter)
            radius_out = hole.shape.get_endpoints_radius()[1].__str__(**opt_param, diameter=diameter)
            pos = hole.position.__str__(**opt_param)
            pos_val.append(hole.position.get_value())
            rows.append([hole.label, 'hole', pos, length, radius, radius_out, '/'])
        for valve in self.valves:
            length = valve.shape.get_endpoints_position()[1].__str__(**opt_param)
            radius = valve.shape.get_endpoints_radius()[0].__str__(**opt_param, diameter=diameter)
            radius_out = valve.shape.get_endpoints_radius()[1].__str__(**opt_param, diameter=diameter)
            pos = valve.position.__str__(**opt_param)
            pos_val.append(valve.position.get_value())
            rows.append([valve.label, 'valve', pos, length, radius, radius_out,
                         valve.reconnection_position.__str__(**opt_param)])
        # sort wr to the position of the side components
        rows = [a for b, a in sorted(zip(pos_val, rows))]
        if all_fields:
            msg = tabulate(rows, col_names)
        elif self.valves or self.holes:
            ind_kept = [0,2,3,4]
            if self.valves:
                ind_kept += [1,6]
            rad_out = [row[4] != row[5] for row in rows]
            if any(rad_out):
                ind_kept += [5]
            ind_sorted = sorted(ind_kept)
            rows_simple = [[row[k] for k in ind_sorted] for row in rows]
            msg = tabulate(rows_simple, [col_names[k] for k in ind_sorted])
        else:
            msg = '' #'None'

        return msg

    def write_side_components_file(self, filename, digit=5, unit='m',
                                   diameter=False, disp_optim=True):
        """
        Write side components (holes and valves) file.

        Parameters
        ----------
        filename : str
            The file name (path).
        digit: int, optional
            The number of digit. The default is 5.
        unit: str {'m', 'mm'}, optional
            The unit (meter or millimeter). Default is 'm' (meter)
        diameter: boolean, optional
            If true print diameter instead of radius. The default is false.
        disp_optim: boolean, optional
            Display optimization: if true display the information related of
            optimisation (bounds, ~, etc). The defauflt is True.

        """
        header = self._get_header(unit, diameter)
        opt_param = {'digit':digit, 'unit':unit, 'diameter':diameter, 'disp_optim':disp_optim}
        f_side = open(filename, "w")
        f_side.write(header)
        f_side.write(self.print_side_components(**opt_param))

    def write_fingering_chart_file(self, filename):
        """
        Write a file with the fingering chart

        Parameters
        ----------
        filename : str
            The file name (path).

        """
        f_fing = open(filename, "w")
        f_fing.write(str(self.fingering_chart))
