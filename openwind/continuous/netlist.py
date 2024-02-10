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

from collections import UserDict
import enum

from openwind.technical import FingeringChart


class MyDict(UserDict):
    """
    A dictionary that iterates over values, and allows union through '+'.
    """
    def __iter__(self):
        return iter(self.data.values())

    def __add__(self, other):
        assert isinstance(other, MyDict)
        return MyDict({**self.data, **other.data})


class EndPos(enum.Enum):
    """
    Convention for the location of the "plus" and "minus" end of the pipe.

    This class can have only two instanciations: ``MINUS`` and ``PLUS``.

    """

    MINUS = 0
    """The end is located at the normalized abscissa 0 on the pipe\
    and correspond to the first element of the vector in descretized pipe"""

    PLUS = 1
    """The end is located at the normalized abscissa 1 on the pipe\
    and correspond to the last element of the vector in descretized pipe"""

    @property
    def array_pos(self):
        """
        Relative index of the end in an array representing the pipe's data.

        Returns
        -------
        int
            - 0 (first element of the array) if MINUS
            - -1 (last element of the array) if PLUS

        """
        if self == EndPos.MINUS:
            return 0
        if self == EndPos.PLUS:
            return -1
        raise ValueError

    @property
    def x(self):
        """
        Relative abscissa x in the pipe.

        Returns
        -------
        float
            - x=0 if MINUS
            - x=1 if PLUS

        """
        if self == EndPos.MINUS:
            return 0.0
        if self == EndPos.PLUS:
            return 1.0
        raise ValueError


class NetlistConnector:
    """
    Component of the netlist, used to connect the pipe-ends together.

    This is related to :class:`Netlist <openwind.continuous.Netlist>` class

    Parameters
    ----------
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
    """

    def __init__(self, label, scaling, convention):
        self.label = label
        self.scaling = scaling
        self.convention = convention

class PipeEnd:
    """
    A netlist-kind PipeEnd.

    This is related to :class:`Netlist <openwind.continuous.Netlist>` class.
    It knows which pipe it is part of, and which connector it is connected to.

    Parameters
    ----------
    pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
        The current Pipe
    pos : :py:class:`EndPos<openwind.continuous.netlist.EndPos>`
        The position of the end on the pipe
    """
    def __init__(self, pipe, pos):
        self.pipe = pipe
        self.pos = pos
        self.connector = None

    def set_connector(self, connector):
        """
        Plug the end to a connector.

        If the end is already connected, it raises an error.

        Parameters
        ----------
        connector: :py:class:`NetlistConnector\
            <openwind.continuous.netlist.NetlistConnector>`
            The connector to which plug the end.
        """
        if self.is_connected():
            raise ValueError("PipeEnd already connected to {}".format(self.connector))
        self.connector = connector

    def get_connector(self):
        """
        Get the connector to which the end is plugged.

        Returns
        -------
        :py:class:`NetlistConnector\
        <openwind.continuous.netlist.NetlistConnector>`
        """
        return self.connector

    def get_pipe(self):
        """
        Get the pipe corresponding to this end.

        Returns
        -------
        :py:class:`Pipe <openwind.continuous.pipe.Pipe>`

        """
        return self.pipe

    def is_connected(self):
        """
        Is the end already connect to a connector?

        Returns
        -------
        bool
        """
        return self.connector != None

    def __repr__(self):
        return 'PipeEnd({},{})'.format(self.pipe, self.connector)

    def __str__(self):
        return self.__repr__()



class Netlist:
    """Represent the connexions in the instrument as a netlist (graph).

    Two main types of components composed the netlist

    - The Pipes (:py:class:`Pipe <openwind.continuous.pipe.Pipe>`) are stored\
    aside. Each of them has two pipe-ends.

    - The connectors which may each be connected to one or several pipe-ends.\
    they can be of three types:

        * A Radiation (:py:class:`PhysicalRadiation\
        <openwind.continuous.physical_radiation.PhysicalRadiation>`)\
        is connected to one pipe-end
        * A Junction (:py:class:`PhysicalJunction\
        <openwind.continuous.junction.PhysicalJunction>`)\
        may be connected to two or three ends (or maybe more soon?)
        * A source (:py:class:`Excitator\
        <openwind.continuous.excitator.Excitator>`) connected to one pipe-end

    The structure of the graph is typically:

    .. code-block:: shell

                                  Radiation

                                      ^  PipeEnd
                                      |
                                      |
                                     Pipe
                                      |
                                      |
                                      v  PipeEnd

        Excitator  <----Pipe---->  Junction  <----Pipe---->  Junction  <----Pipe---->  Radiation
                PipeEnd      PipeEnd      PipeEnd      PipeEnd      PipeEnd      PipeEnd

    .. warning::
        The order of the ends in connections is important
        (just like on the pins of an electronic component, you can't reverse a
        diode or shuffle the connections to an op-amp), and should be given in
        the order that is conventional for the component.

    Attributes
    ----------
    pipes: dict(tuple(:py:class:`Pipe <openwind.continuous.pipe.Pipe>`, \
    :py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`))
        a dict of all the pipes and their pipe ends
    connectors : dict
        dictionary with all the :py:class:`NetlistConnector \
         <openwind.continuous.netlist.NetlistConnector>` of the graph
    ends : list(:py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`)
        list of all the  pipe-ends of the graph
    fingering_chart : :py:class:`FingeringChart \
    <openwind.technical.fingering_chart.FingeringChart>`
       The fingering chart associated to the netlist.
    """

    def __init__(self):
        # self._netlist : dict string -> (Component, list[pipe_end])
        self.pipes = dict()
        self.connectors = dict()
        self.ends = []
        self.fingering_chart = FingeringChart()

    def reset(self):
        """
        Reinitializes the Netlist: remove every components.

        Used in optimization to re-instanciate the netlist if the graph has
        been modified.
        """
        self.pipes = dict()
        self.connectors = dict()
        self.ends = []

    def _new_pipe_end(self, pipe, pos):
        """
        Get a new pipe_end, to which stuff can be connected.

        Parameters
        ----------
        pipe_end : :py:class:`Pipe <openwind.continuous.netlist.PipeEnd>`
            the new pipe-end

        """
        pipe_end = PipeEnd(pipe, pos)
        self.ends.append(pipe_end)
        return pipe_end

    def add_pipe(self, pipe):
        """
        Add a Pipe to the netlist, and return the labels of its ends.

        Parameters
        ----------
        pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
            the Pipe to add

        Returns
        -------
        pipe_ends : list(:py:class:`PipeEnd <openwind.continuous.netlist.PipeEnd>`)
            The two ends of the pipe.

        """
        if pipe.label in self.pipes:
            raise ValueError(("A Pipe with label '{}' has already been "
                             "added to the netlist").format(pipe.label))
        pipe_ends = [self._new_pipe_end(pipe, pos) for pos in EndPos]
        self.pipes[pipe.label] = (pipe, pipe_ends)
        return pipe_ends

    def add_connector(self, connector, *pipe_ends):
        """
        Add a connector to the netlist and connect it to the specified ends.

        The number of ends depends of the type of connector. The order of the
        ends HAVE an importance.

        Parameters
        ----------
        pipe_ends : list(:py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`)
            The pipe ends to which connect the new connector.

        """
        if connector.label in self.connectors:
            raise ValueError(("A Connector with label '{}' has already been "
                             "added to the netlist").format(connector.label))
        for pipe_end in pipe_ends:
            pipe_end.set_connector(connector)
        self.connectors[connector.label] = (connector, pipe_ends)

    def get_connector_and_ends(self, label):
        """
        Get a connector and its pipe_ends from its label.

        Parameters
        ----------
        label : string
            The label of the connector

        Returns
        -------
        connector : :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`
            The connector corresponding to the label.
        ends : :py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`
            The pipe-ends connected to the connector.

        """
        connector, ends = self.connectors.get(label)
        return connector, ends

    def get_pipe_and_ends(self, label):
        """
        Get a pipe and its pipe_ends from its label.

        Parameters
        ----------
        label : string
            The label of the pipe

        Returns
        -------
        pipe : :py:class:`Pipe<openwind.continuous.pipe.Pipe>`
            The pipe corresponding to the label.
        ends : list(:py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`)
            The pipe-ends of the pipe.

        """
        pipe, ends = self.pipes.get(label)
        return pipe, ends

    def get_free_ends(self):
        """
        List of all the ends that are not connected to a connector.

        Returns
        -------
        free_ends : list(:py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`)
            The free ends of the graph.

        """
        free_ends = []
        for pipe_end in self.ends:
            if not pipe_end.is_connected():
                free_ends.append(pipe_end)
        return free_ends

    def check_valid(self):
        """
        Check if the netlist is valid for building a music instrument.

        It checks if each pipe_end is between one pipe and one connector.

        Returns
        -------
        bool

        """
        free_ends = self.get_free_ends()
        if len(free_ends) > 0:
            raise ValueError('Some pipe-ends are not connected to '
                             'a connector!\n'
                             + str(free_ends))


    def get_connectors_of_class(self, class_):
        """
        List all the connectors of a given class in this netlist.

        Parameters
        ----------
        ``class_`` : class
            The researched class along the \
            :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`.

        Returns
        -------
        list(:py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`)
            The connector founds.

        """
        return [con for con, ends in self.connectors.values()
                if isinstance(con, class_)]

    def set_fingering_chart(self, fingering_chart):
        """
        Associate a fingering chart to the netlist.

        The labels indicated in the fingering chart must correspond to the
        labels of the netlist component.

        Parameters
        ----------
        fingering_chart : :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`
            The fingering chart

        """

        self.fingering_chart = fingering_chart

    def get_fingering_chart(self):
        """
        Get the current fingering chart associated to the netlist.

        Returns
        -------
        :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`
            The fingering chart

        """
        return self.fingering_chart

    def convert_with_structure(self, convert_pipe, convert_connector):
        """
        Create specialized components while preserving graph structure.

        We assume that for each netlist component, we want to build exactly
        one "specialized component" (such as FComponent or TComponent).
        To do so, one factory function must be given for pipes,
        and one for connectors.

        The following diagram illustrates how this method operates.

        1. Convert Pipes to "SpecialPipes" (i.e. TemporalPipes or FrequentialPipes)
        2. Obtain the ends of the SpecialPipes
        3. Create the SpecialConnectors and connect them to the correct ends.

        .. code-block:: shell

              Pipe <--------> PipeEnd <-------------> NetlistConnector
                 |    (2)                    (n)            |
              1. |                                          |
                 |                                          |
                 v                                          | 3.
            SpecialPipe                                     |
                    \                                       |
                  2. \                                      |
                      \                                     |
                       v                                    v
                    SpecialPipeEnd <--------------> SpecialConnector
                         (2)              (n)

        ..
            The following picture is only for the online documentation
            
        .. image :: https://files.inria.fr/openwind/pictures/convert_structure.png
          :width: 1000
          :align: center


        Parameters
        ----------
        convert_pipe: Callable[Pipe, SpecialPipe]
            Function that constructs the appropriate SpecialPipe.
        convert_connector: Callable[(NetlistConnector, Tuple[SpecialPipeEnd]),\
                                     SpecialConnector]
            Function that constructs the appropriate Specialconnector and
            connects it to the given SpecialPipeEnds.

        Returns
        -------
        special_pipes : list(SpecialPipe)
            The converted pipes
        specialconnectors : list(SpecialConnector)
            The converted connectors

        """
        self.check_valid()
        special_pipes = MyDict()
        special_ends = dict()
        for label in self.pipes:
            pipe, ends = self.pipes[label]
            s_pipe = convert_pipe(pipe)
            special_pipes[label] = s_pipe
            try:
                for end, s_end in zip(ends, s_pipe.get_ends()):
                    special_ends[end] = s_end
            except TypeError:
                raise ValueError("Failure in conversion of pipe: object %s "
                                 "does not define .get_ends()")

        special_connectors = MyDict()
        for label in self.connectors:
            connector, ends = self.connectors[label]
            s_ends = [special_ends[end] for end in ends]
            s_connector = convert_connector(connector, s_ends)
            special_connectors[label] = s_connector

        return special_pipes, special_connectors

    def __repr__(self):
        return ("<openwind.continous.Netlist(pipes={}, "
                "connectors={}, fingering_chart={})>").format(
                    list(self.pipes.keys()),
                    list(self.connectors.keys()),
                    self.fingering_chart.all_notes())

    def __str__(self):
        return ("Netlist:\n\tPipes = {};\n"
                "\tConnectors = {};\n\tFingering Chart = {}").format(
                    list(self.pipes.keys()),
                    list(self.connectors.keys()),
                    self.fingering_chart.all_notes())
