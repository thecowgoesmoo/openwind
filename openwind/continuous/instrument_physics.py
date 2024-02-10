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
Create the continuous instrument graph in which each component is associated to
its physical equation.
"""

import numpy as np
import warnings
from openwind.continuous import (Pipe, PhysicalRadiation,
                                 Scaling, Netlist, JunctionTjoint, JunctionSwitch,
                                 JunctionSimple, losses_model, radiation_model,
                                 JunctionDiscontinuity, create_excitator,
                                 Tonehole)
from openwind.design import ShapeSlice
from openwind.technical.instrument_geometry import Hole, BrassValve


class InstrumentPhysics:

    """Create the continuous instrument graph with physical considerations.

    Create the instrument's graph: an ensemble of pipes (main bore, brass valve
    deviation pipe and chimney hole) the ends of which being connected to
    different type of connector:

    * radiation (:py:class:`PhysicalRadiation\
    <openwind.continuous.physical_radiation.PhysicalRadiation>`)
    * junction (:py:class:`PhysicalJunction\
    <openwind.continuous.junction.PhysicalJunction>`)
    * source (:py:class:`Excitator\
    <openwind.continuous.excitator.Excitator>`)

    All the connexions of this graph are specified in a
    :py:class:`Netlist <openwind.continuous.netlist.Netlist>` object.
    Each component of this graph is associated to physical equations (such as
    sound propagation equation for the pipe) and it computes its own physical
    coefficients.

    .. seealso:: :py:class:`Netlist <openwind.continuous.netlist.Netlist>` \
            for more detail on the graph structure

    Parameters
    ----------
    instrument_geometry : :py:class:`InstrumentGeometry \
    <openwind.technical.instrument_geometry.InstrumentGeometry>`
        The geometry of the instrument. This object must contain:

        - a list of :py:class:`DesignShape <openwind.design.design_shape.DesignShape>` to \
            to describe the main bore
        - a list of :py:class:`Hole <openwind.technical.instrument_geometry.Hole>` \
            to describe the holes
        - a list of :py:class:`BrassValve <openwind.technical.instrument_geometry.BrassValve>` \
            to describe the valves
        - a :py:class:`FingeringChart <openwind.technical.fingering_chart.FingeringChart>` \
            object
    temperature : float or callable
        the temperature at which the physical properties are calculated.

        .. warning::

            if the temperature is non constant, it is supposed variable
            along the main tube (main bore). In the holes, the temperature is
            supposed homogeneous. In the valve deviation pipe, the gradient is
            set to have the right temperature at its two connections with the
            main bore.

    player : :py:class:`Player <openwind.technical.player.Player>`
        the controler for the instrument
    losses : {False, True, 'bessel', 'diffrepr', 'wl', 'keefe' ,'minikeefe'}
        How to take into account the thermoviscous losses. This use the method\
        :py:meth:`losses_model\
        <openwind.continuous.thermoviscous_models.losses_model>` \
        to instanciate :py:class:`ThermoviscousModel\
        <openwind.continuous.thermoviscous_models.ThermoviscousModel>`

    radiation_category : str, tuple, dict or :py:class:`PhysicalRadiation \
    <openwind.continuous.physical_radiation.PhysicalRadiation>` , optional
        Model of radiation impedance used. The string must be one of the
        available category. For some of them an additional information is
        needed and given as second member of a tuple. The use of dict gives the
        possibility to use different condition at each opening (cf
        :meth:`rad_model`). Default is 'unflanged'. More details
        on available model names in :py:meth:`radiation_model \
        <openwind.continuous.radiation_model.radiation_model>`.

    nondim : Boolean, optional
        If true all the physical parameters in the equations are
        nondimensionalized (cross section, sound celerity, air density, time).
        The default is False
    is_time_scaled: bool, optional
        If false, the time is not scaled whatever is the nondim option. Default
        is True.
    convention : {'PH1', 'VH1'}, optional
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
        The default is 'PH1'
    spherical_waves : Boolean, optional
        If true, spherical waves are assumed in the pipe. The default is False.
        Option spherical_waves='spherical_area_corr' also enables the area correction
        (otherwise only the length correction is used).
    discontinuity_mass : Boolean, optional
        If true, acoustic mass is included in the junction between two pipe
        with different cross section. The default is True.
    matching_volume : boolean, optional
        Include or not the matching volume between the main and the side tubes
        in the masses of the junction. The default is False.
    loss_factor_alphainv: float, optional
        Multiply the amount of viscothermal losses by this factor. If different
        from 1, losses must be one of [True, 'bessel', 'bessel_new'].
        The default is 1.
    humidity: float, optional
        The humidity rate (between 0 and 1). The default is 0.5.
    carbon: float, optional
        The carbon rate (between 0 and 1). Default: 4.2e-4 (420ppm)
    source_location: string, optional
        The label of the opening where put the acoustique source (typically a hole label).
        Default is "entrance", the input of the main bore.
    physics_opt: keywords arguments, optional
        Other keywords argument of :py:class:`Physics<openwind.continuous.phyics.Physics>`


    .. seealso:: :py:mod:`openwind.continuous` for all the possible components\
        of the graph


    Attributes
    ----------
    netlist : :py:class:`Netlist <openwind.continuous.netlist.Netlist>`
        Graph of all the connected parts of the instrument.
    scaling : :py:class:`Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    rad_dict : dict
        The dictionnary indicating the radiation category for each opening. The
        keys are the labels of the holes and "bell" (main bore opening)

    """

    def __init__(self, instrument_geometry, temperature,  player, losses,
                 radiation_category="unflanged", nondim=False, is_time_scaled=True,
                 convention='PH1', spherical_waves=False,
                 discontinuity_mass=True,
                 matching_volume=False,
                 loss_factor_alphainv=1.0,
                 humidity=.5, carbon=4e-4,
                 assembled_toneholes=False,
                 source_location='entrance',
                 **physics_opt):
        self.instrument_geometry = instrument_geometry
        self.player = player
        self.netlist = Netlist()
        self.temperature = temperature
        self.humidity = humidity
        self.carbon = carbon
        self.physics_opt= physics_opt
        # initialize the scaling object with all the coeffs to one
        self.scaling = Scaling(is_time_scaled=is_time_scaled)

        self.losses = losses_model(losses, loss_factor_alphainv=loss_factor_alphainv)
        # self.rad_model = radiation_model(radiation_category)
        self.rad_dict = dict()
        self.set_radiation_dict(radiation_category)
        self.convention = convention
        self.spherical_waves = spherical_waves
        self.discontinuity_mass = discontinuity_mass
        self.matching_volume = matching_volume
        self.assembled_toneholes = assembled_toneholes

        if not source_location:
            self.source_location = 'entrance'
        else:
            self.source_location = source_location
        holes_labels = [h.label for h in instrument_geometry.holes] + ['entrance']
        if source_location not in holes_labels:
            raise ValueError(f'The source point "{source_location}" does not correspond to any opening label.\n'
                             f'Chose between: {holes_labels}')

        self.__build_netlist()
        self.netlist.set_fingering_chart(instrument_geometry.fingering_chart)


        _, (input_end,) = self.netlist.get_connector_and_ends(self.source_label)
        self._first_pipe = input_end.get_pipe()
        if nondim:
            self.scaling.set_nondimensionalization(self._first_pipe)

        self.__check()

    def get_entry_coefs(self, *labels):
        """Obtain values of physical properties of air at the entry point.

        Parameters
        ----------
        *labels : string...
            the names of the coefficients to take.
            See :py:class:`Physics <openwind.continuous.physics.Physics>`

        Returns
        -------
        values : tuple of (float or array-like)

        Example
        -------

        .. code-block:: python

           instru_phy.get_entry_coefs('Cp', 'c', 'gamma', 'kappa', 'mu', 'rho', 'temp', 'T')

        """
        return self._first_pipe.get_physics().get_coefs(0, *labels)

    def rad_model(self, label, pipe_end):
        """
        Get the radiation model corresponding to a given opening.

        In case of 'pulsating_sphere' model (:py:class:`RadiationPulsatingSphere\
        <openwind.continuous.radiation_pulsating_sphere.RadiationPulsatingSphere>`)
        without indication on the angle, the angle of the radiating sphere is
        computed from the conicity of the pipe, at the location of the
        radiating pipe end.

        Parameters
        ----------
        label : str
            The label of the opening.
        pipe_end : :py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`
            The pipe end corresponding.

        Returns
        -------
        str or :py:class:`PhysicalRadiation\
        <openwind.continuous.physical_radiation.PhysicalRadiation>`
            The radiation model.

        """
        if self.rad_dict[label] == 'pulsating_sphere':
            slope = pipe_end.get_pipe().get_conicity_at(pipe_end.pos.x)
            if slope == 0:
                print("Warnings: For cylinder, 'pulsating_sphere' condition "
                      "correspond to 'total_transmission'.")
            theta = np.arctan(slope)
            return (self.rad_dict[label], theta)
        else:
            return self.rad_dict[label]

    def __repr__(self):
        return ("<openwind.InstrumentPhysics("
                "\n\t{},".format(repr(self.instrument_geometry)) +
                "\n\ttemperature={},".format(repr(self.temperature)) +
                "\n\t{},".format(repr(self.player)) +
                "\n\tlosses={},".format(repr(self.losses)) +
                "\n\t{},".format(repr(self.netlist)) +
                "\n\t{},".format(repr(self.scaling)) +
                "\n\trad_model={},".format((self.rad_dict)) +
                "\n\tconvention='{:s}',".format(self.convention) +
                "\n\tspherical_waves={},".format(self.spherical_waves) +
                "\n\tdiscontinuity_mass={},".format(self.discontinuity_mass) +
                "\n\tmatching_volume={}\n)>".format(self.matching_volume))

    def __str__(self):
        return ("InstrumentPhysics:\n" + "="*20 +
                "\nInstrument Geometry:\n{}\n".format(self.instrument_geometry) +"="*20 +
                "\nTemperature: {}°C\n".format(self.temperature) + "="*20 +
                "\nLosses: {}\n".format(self.losses) + "="*20 +
                "\n{}\n".format(self.player) + "="*20 +
                "\n{}\n".format(self.netlist) + "="*20 +
                "\n{}\n".format(self.scaling) + "="*20 +
                "\nRadiation Model: {}\n".format(self.rad_dict) + "="*20 +
                "\nOptions:" +
                "\n\tconvention: {:s}".format(self.convention) +
                "\n\tspherical_waves: {}".format(self.spherical_waves) +
                "\n\tdiscontinuity_mass: {}".format(self.discontinuity_mass) +
                "\n\tmatching_volume: {}".format(self.matching_volume))

    def set_radiation_dict(self, radiation_category):
        """
        Instanciate the dictionary associating the radiation category at each opening.

        The instanciate dictionnary have the key "bell" specifying the
        radiation category of the maine bore and one key per hole with its
        label. If the point source is not at the entrance, the key "entrance"
        can be used to specify its radiation.

            .. code-block:: python

                rad_dict = {'bell': rad_cat0, hole1_label: rad_cat1,
                            hole2_label: rad_cat2, ...}

        Parameters
        ----------
        radiation_category : str or dict
            The radiation category for all the opening or the dict specifying
            the radiation of each opening.


        - If a string is given, the indicated radiation category is applied to\
        all the opening.

           .. code-block:: python

               radiation_category = "unflanged"

        - If a dictionnary is given, it allows association of different \
        radiation category to each opening. The dictionnary can have the keys

            * "bell", "entrance" and "holes": setting the same radiation category for all \
            the holes

                .. code-block:: python

                    radiation_category = {'bell': 'unflanged', 'holes': 'infinite_flanged'}

            * "bell", "entrance", and all the hole label: giving the possibility to set a \
            different radiation category to each opening

                .. code-block:: python

                    radiation_category = {'entrance': 'closed'; 'bell': 'unflanged',
                                          'hole1': 'infinite_flanged',
                                          'hole2': 'pulsating_sphere',
                                          'hole3': 'unflanged'}

        """
        msg = ('The "radiation_category" must contain one string or a dict '
               'with keys "bell" + if holes, the key "holes" or the label of '
               'each hole).')
        if not isinstance(radiation_category, dict):
            self.rad_dict.update({'bell': radiation_category})
            self.rad_dict.update({'entrance': radiation_category})
            for hole in self.instrument_geometry.holes:
                self.rad_dict.update({hole.label: radiation_category})
        elif 'bell' in radiation_category.keys():
            self.rad_dict.update({'bell': radiation_category['bell']})
            if 'entrance' in radiation_category.keys():
                self.rad_dict.update({'entrance': radiation_category['entrance']})
            else:
                self.rad_dict.update({'entrance': None})
            if len(self.instrument_geometry.holes) > 0:
                labels = [hole.label for hole
                          in self.instrument_geometry.holes]
                if 'holes' in radiation_category.keys():
                    for label in labels:
                        self.rad_dict.update({label: radiation_category['holes']})
                elif all([label in radiation_category.keys() for label
                          in labels]):
                    self.rad_dict = radiation_category
                else:
                    raise ValueError(msg)
        else:
            raise ValueError(msg)


    def __build_netlist(self):
        """
        Build the graph of the instrument by assuming a main bore and
        eventually some side holes and valves deviation pipes. Each deviation
        pipe, hole chimney and the main bore shapes are converted into pipes
        connected by junctions and having radiation condition.

        1. the side elements (valves and holes) are localized on the main bore
        2. the entrance is associated to a source condition
        3. each shape constituing the main bore is converted in one or several\
            pipes if holes are localized on it
        4. the last end (the bell) is associated to a radiation condition
        """
        # 1-localize the holes and valves
        self.side_elements = self._get_side_elements()
        # the 2 end of the valve deviation pipe are not connected together and
        # must be store somewhere.
        self.unconnect_end_valves = dict.fromkeys([valve.label for valve in
                                                self.instrument_geometry.valves])

        # 2-source object
        self.source_label = "source" # self.source_location + f"_{self.player.excitator_type.lower()}_source"
        self.excitator_model = create_excitator(self.player, self.source_label,
                                                self.scaling, self.convention)

        # 3-first shape "entrance"
        entrance_shape = self.instrument_geometry.main_bore_shapes[0]
        (main_end_up,
         main_end_down) = self.__create_main_bore_pipes(entrance_shape, '0')
        if self.source_location=='entrance':
            self.netlist.add_connector(self.excitator_model, main_end_up)
        else: # if the source point is elsewhere, the entrance is associated to its radiation
            rad_entrance_label = 'entrance_radiation'
            rad_entrance = radiation_model(self.rad_model('entrance', main_end_up),
                                           rad_entrance_label, self.scaling,
                                           self.convention)
            self.netlist.add_connector(rad_entrance, main_end_up)

        # 4-entire instrument
        for k, shape in enumerate(self.instrument_geometry.main_bore_shapes[1:]):
            (pipe_end_up,
             pipe_end_down) = self.__create_main_bore_pipes(shape, str(k+1))
            self.__joint_2_main_bore_section(pipe_end_up, main_end_down,
                                             str(k) + '_' + str(k+1))
            main_end_down = pipe_end_down

        # 5-bell
        rad_label = 'bell_radiation'
        rad_bell = radiation_model(self.rad_model('bell', main_end_down),
                                     rad_label, self.scaling, self.convention)
        self.netlist.add_connector(rad_bell, main_end_down)
        self.netlist.check_valid()


    def _update_player(self):
        """
        Method that calls to the :py:class:`Excitator \
        <openwind.continuous.excitator.Excitator` private method \
        :py:meth:`_update_fileds \
        <openwind.continuous.excitator.Excitator._update_fileds>` to update \
        all fields according to the :py:class:`Player \
        <openwind.technical.player.Player>` attributes
        """
        self.excitator_model._update_fields(self.player.control_parameters)

    def update_netlist(self):
        """
        Update the graph after a modification of a geometric parameter.

        Particularly useful in inversion, this method actualize the graph after
        a modification of a geometric parameter.

        .. warning::

            This method can modify the instrument topology.
        """
        self.netlist.reset()
        self.__build_netlist()

    def __joint_2_main_bore_section(self, main_end_up, main_end_down, junc_ID):
        """
        Connect two tubes trough a trivial junction without masse.

        Create a connector component of type :py:class:`JunctionSimple \
        openwind.continuous.junction.JunctionSimple>` and add it to the netlist.

        Parameters
        ----------
        main_end_up : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The end of the "upstream" pipe.
        main_end_down : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The end of the "downstream" pipe.
        junc_ID : str
            The name of the created `openwind.continuous.junction.JunctionSimple`
            object.

        """
        pos_up, _ = main_end_up.get_pipe().get_endpoints_position_value()

        side_at_junc = [elem for elem in self.side_elements if elem[0]==pos_up]
        if len(side_at_junc) > 1:
            raise ValueError('Two side elements have the same location: {}'.format(side_at_junc))
        elif len(side_at_junc) == 1:
            elem = side_at_junc[0]
            if elem[2].is_variable():
                warn_msg = ('It is impossible to vary the position of the {}'
                            'placed at the junction of two pipes.').format(elem[1])
                warnings.warn(warn_msg)
            self._connect_side_element(elem[1], main_end_down, main_end_up)
        else:
            label = 'junction_' + junc_ID
            if self.discontinuity_mass:
                two_junction = JunctionDiscontinuity(label, self.scaling,
                                                     self.convention)
            else:
                two_junction = JunctionSimple(label, self.scaling,
                                              self.convention)

            self.netlist.add_connector(two_junction, main_end_down,
                                       main_end_up)

    def __create_main_bore_pipes(self, shape, pipeID):
        """Convert one shape of the main bore into one pipe or several if holes
        or valves are located on it.

        Parameters
        ----------
        shape : :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
            The shape of the considered pipe.
        pipeID : str
            The name of the considered pipe.

        Returns
        -------
        pipe_ends : tuple of :py:class:`PipeEnd \
        <openwind.continuous.netlist.PipeEnd>`
            The main upstream and downstream ends of the created pipe(s) which
            are still unconnected.
        """
        pos_min, pos_max = shape.get_endpoints_position()

        side_elem_on_shape = [elem for elem in self.side_elements
                              if elem[0] > pos_min.get_value()
                              and elem[0] < pos_max.get_value()]

        label = ('bore' + str(pipeID))
        if len(side_elem_on_shape)>0:
            pipe_ends = self.__slice_shape(side_elem_on_shape, shape, label)
        else: # if no side elements on the shape, no slicing
            _, pipe_ends = self.__create_pipe(shape, label, pos_min, pos_max,
                                           main_bore=True)
        return pipe_ends

    def _get_side_elements(self):
        """
        Get all side elements (holes, valves (entry and reconnection)) and sort them
        wr to their location on the main bore.
        """
        holes = [(hole.position.get_value(), hole, hole.position)
                 for hole in self.instrument_geometry.holes]
        valves = [(valve.position.get_value(), valve, valve.position)
                  for valve in self.instrument_geometry.valves]
        returns_valves = [(valve.reconnection_position.get_value(), valve.label,
                           valve.reconnection_position)
                          for valve in self.instrument_geometry.valves]
        side_comp = sorted(holes + valves + returns_valves, key=lambda x:x[0])
        dist = np.diff([sc[0] for sc in side_comp])
        if np.any(dist==0):
            msg = '\n'
            for k in np.where(dist==0)[0]:
                msg += f'{side_comp[k][1]} and {side_comp[k+1][1]}: {side_comp[k][0]}\n'
            raise ValueError('Several side components are located at the same place:'+msg)
        return sorted(side_comp, key=lambda x:x[0])


    def __create_pipe(self, shape, label, pos_min, pos_max, main_bore=False,
                      add_to_netlist=True):
        """
        Create a pipe component of the graph.

        The object :py:class:`Pipe <openwind.continuous.pipe.Pipe` created is
        added to the netlist.

        Parameters
        ----------
        shape : :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
            The pipe shape.
        label : str
            The pipe name.
        pos_min : :py:class:`DesignParameter <openwind.design.design_parameter.DesignParameter>`
            The upstream end position of the pipe on the main bore axis (used
            for the temperature scale).
        pos_max : :py:class:`DesignParameter <openwind.design.design_parameter.DesignParameter>`
            The downstream end position of the pipe on the main bore axis (used
            for the temperature scale).
        main_bore : bool, optional
            If True, the pipe is considered as a part of the main bore (usefull
            for example for interpolation). The default is False.

        Returns
        -------
        end_up : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The upstream end of the created pipe.
        end_down : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The downstream end of the created pipe.

        """
        temperature_slice = self.slice_temperature(self.temperature,
                                                   pos_min.get_value(),
                                                   pos_max.get_value())
        humidity_slice = self.slice_temperature(self.humidity,
                                                   pos_min.get_value(),
                                                   pos_max.get_value())
        carbon_slice = self.slice_temperature(self.carbon,
                                                   pos_min.get_value(),
                                                   pos_max.get_value())
        pipe = Pipe(shape, temperature_slice, label, self.scaling,
                    self.losses, self.convention, self.spherical_waves,
                    humidity=humidity_slice, carbon=carbon_slice, **self.physics_opt)
        pipe.on_main_bore = main_bore
        if add_to_netlist:
            end_up, end_down = self.netlist.add_pipe(pipe)
            return pipe, (end_up, end_down)
        else:
            return pipe, (None, None)

    def __slice_shape(self, side_elem_on_shape, shape, label):
        """
        Cut the shape at the position of the side elements, create the corresponding
        pipe and create the holes pipe and radiation condition and the valves
        and the deviation pipes.

        Parameters
        ----------
        side_elem_on_shape : list of tuple
            The side elements locate on the considered main bore part.
            Each tuple contains 3 elements: the numerical value of the side
            element location on the main bore, the element, and the
            :py:class`DesignParameter` corresponding to its location.
        shape : :py:class:`DesignShape <openwind.design.design_shape.DesignShape>`
            The main bore shape considered.
        label : str
            The name of the considered shape.

        Returns
        -------
        upstream_end : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The main upstream end of the created pipes (still unconnected).
        end_down : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The main downtream end of the created pipes (stil unconnected).
        """

        side_pos, side_elem, design_pos = zip(*side_elem_on_shape)

        pos_min, pos_max = shape.get_endpoints_position()
        # the slicing positions (DesignParameters)
        x_cut = [pos_min] + list(design_pos) + [pos_max]

        label_slice = label + '_slice' + str(0)
        shape_slice = ShapeSlice(shape, x_cut[0:2], label=label)
        _, (upstream_end, end_down) = self.__create_pipe(shape_slice, label_slice,
                                                    x_cut[0], x_cut[1],
                                                    main_bore=True)
        for p, elem in enumerate(side_elem):
            label_slice = label + '_slice' + str(p+1)
            shape_slice = ShapeSlice(shape, x_cut[p+1:p+3], label=label)
            _, (end_up, end_down_temp) = self.__create_pipe(shape_slice,
                                                       label_slice,
                                                       x_cut[p+1], x_cut[p+2],
                                                       main_bore=True)
            self._connect_side_element(elem, end_down, end_up)
            end_down = end_down_temp

        return upstream_end, end_down

    def _connect_side_element(self, side_elem, end_down, end_up):
        """
        Connect the side element to the two ends of the main bore with
        respect to its type.

        Parameters
        ----------
        side_elem : :py:class:`Hole`, :py:class:`BrassValve` or str
            The element to connect, if it is a string it corresponds to the
            label of the valve the return of which must be connected.
        end_down : :py:class:`PipeEnd`
            The downstream end of the upstream pipe.
        end_up : :py:class:`PipeEnd`
            The upstream end of the downstream pipe

        """
        if type(side_elem) is Hole:
            self.__create_hole(side_elem, end_down, end_up)
        elif type(side_elem) is BrassValve:
            self._connect_valve(side_elem.label, end_down, end_up, False)
        elif type(side_elem) is str:
            self._connect_valve(side_elem, end_up, end_down, True)



    def __create_hole(self, hole, end_down, end_up):
        """
        Create the chimney pipe of the hole and connect it to the main bore.

        The chimney hole is associated to a `openwind.continuous.pipe.Pipe`
        and connect to the main bore by a
        `openwind.continuous.junction.JunctionTjoint`. The other end of the
        chimney pipe is connected to a
        `openwind.continuous.physical_radiation.PhysicalRadiation`

        Parameters
        ----------
        hole : :py:class:`Hole <openwind.technical.instrument_geometry.Hole>`
            The hole considered.
        end_down : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The "downstream" end of the "upstream" pipe.
        end_up : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The "upstream" end of the "downstream" pipe.
        assembled_tonehole : bool
            Whether to assemble the components into a single Tonehole object,
            for locally implicit time stepping. Default is False, keep the
            components separate.
        """
        # hole chimney
        pipe, (end_hole_up, end_hole_down) = self.__create_pipe(hole.shape, hole.label,
                                                        hole.position,
                                                        hole.position,
                                                        add_to_netlist=not self.assembled_toneholes)
        # hole radiation
        if self.source_location==hole.label:
            hole_rad = self.excitator_model
        else:
            rad_label = hole.label + '_radiation'
            hole_rad = radiation_model(self.rad_model(hole.label, end_hole_down),
                                         rad_label, self.scaling, self.convention)


        # hole junction
        junc_label = 'junction_' + hole.label
        junction = JunctionTjoint(junc_label, self.scaling, self.convention,
                                  self.matching_volume)
        if self.assembled_toneholes:
            tonehole = Tonehole(junction, pipe, hole_rad,
                                "tonehole_"+hole.label, self.scaling, self.convention)
            self.netlist.add_connector(tonehole, end_down, end_up)
        else:
            self.netlist.add_connector(hole_rad, end_hole_down)
            self.netlist.add_connector(junction, end_down, end_up, end_hole_up)


    def _connect_valve(self, valve_label, end_main, end_default, is_reconnect):
        """
        Connect the considered end of the deviation pipe assiciated to the
        brass valve.

        If the deviation pipe does not exist yet, it is created, and the other
        end of the pipe is stored in :py:attr:`unconnect_end_valves`

        .. warning::
            The switch is connected in different direction if it is the
            entry or the reconnection of the deviation pipe!

        Parameters
        ----------
        valve_label : str
            The label of the considered brass valve.
        end_down : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The "downstream" end of the "upstream" pipe.
        end_up : :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`
            The "upstream" end of the "downstream" pipe.
        is_reconnect : bool
            If true, the connection convention for the switch is the one of the
            reconnection switch.
        """
        # detect if the end already exists
        end_to_connect = self.unconnect_end_valves.get(valve_label)
        if not end_to_connect:
            # due to the graph construction process, the return of the deviation
            # pipe can be set before its begining, even if it is located after
            # on the main bore.
            valve = [valve for valve in self.instrument_geometry.valves
                     if valve.label==valve_label][0]
            _, (end_valve_up, end_valve_down) = self.__create_pipe(valve.shape,
                                                              valve.label,
                                                              valve.position,
                                                              valve.reconnection_position)
            if is_reconnect:
                end_to_connect = end_valve_down
                self.unconnect_end_valves[valve.label] = end_valve_up
            else:
                end_to_connect = end_valve_up
                self.unconnect_end_valves[valve.label] = end_valve_down

        if is_reconnect:
            junc_label = valve_label + '_reconnection_switch'
        else:
            junc_label = valve_label + '_entry_switch'
        junction = JunctionSwitch(junc_label, self.scaling, self.convention,
                                  self.discontinuity_mass)
        self.netlist.add_connector(junction, end_main, end_default, end_to_connect)


    def __check(self):
        """
        Check that the number of :py:class:`PhysicalRadiation \
        <openwind.continuous.physical_radiation.PhysicalRadiation>` created
        correspond to the number of holes + 1 for the bell.
        """
        # Check how many radiations were created
        assert (len(self.netlist.get_connectors_of_class(PhysicalRadiation))
                    + len(self.netlist.get_connectors_of_class(Tonehole))
                == len(self.instrument_geometry.holes) + 1)

    @staticmethod
    def slice_temperature(temperature, pos_min, pos_max):
        """
        Extract the temperature evolution allong a portion of pipe.

        Return a temperature function which verifies

        .. math::
            y(0) = T(x_{min}) \\\\
            y(1) = T(x_{max})

        with

        - :math:`T(x)` : the temperature evolution with respect to the position
        - :math:`x_{min}, x_{max}` the two endpoints position of the slice.

        It is used to associate a temperature evolution for each part of the
        instrument.

        Parameters
        ----------
        temperature : float or callable
            The temperature with respect to the position.
        pos_min : float
            The minimal position used for the change of variable.
        pos_max : float
            The maximal position used for the change of variable..

        Returns
        -------
        float or callable
            The sliced temperature function.

        """
        if callable(temperature):
            slice_temp = lambda x: temperature(x*(pos_max - pos_min) + pos_min)
            return slice_temp
        else:
            return temperature
