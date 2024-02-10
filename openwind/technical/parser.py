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

import warnings
import numpy as np
from packaging import version

import openwind

"""
Different methods to parse the files
"""

def from_files_to_lists(bore_file, hole_file, fing_file):
    """
    If needed, read files and convert to lists usable by openwind.

    Parameters
    ----------
    bore_file : list or str
        The name (+ path) of the file with the main bore data, or the list.
        It can also be the name of the unique file with all the data.
    hole_file : list or str
        The name (+path) of the file with the holes/valves data, or the
        corresponding list.
    fing_file : list or str
        The name (+ path) of the file with the fingering chart data
        (or the corresponding list).

    Raises
    ------
    TypeError
        Raises error is the input arg are not list or str.

    Returns
    -------
    bore_list : list
        The list witht he main bore data.
    hole_list : list
        the list with the holes/valves data.
    fing_list : list
        The list with the fing. chart data.
    bore_opt : dict
        the dictionnary with the main bore options (unit, diameter, etc.).
    hole_opt : dict
        the dictionnary with the holes/valves options (unit, diameter, etc.).

    """

    if not (isinstance(bore_file, str) or isinstance(bore_file, list)):
        raise TypeError(f'Main bore data: expected "str" or "list" instance, not "{type(bore_file).__name__}"')
    if not (isinstance(hole_file, str) or isinstance(hole_file, list)):
        raise TypeError(f'Holes/Valves data: expected "str" or "list" instance, not "{type(hole_file).__name__}"')
    # if not (isinstance(fing_file, str) or isinstance(fing_file, list)):
    #     raise TypeError(f'Fing. Chart data: expected "str" or "list" instance, not "{type(fing_file).__name__}"')

    # main bore and single file
    if isinstance(bore_file, list):
        bore_list = bore_file
        bore_opt = dict()
        hole_list = list()
        hole_opt = dict()
        fing_list = list()

    else:
        with open(bore_file) as file:
            bore_lines = file.readlines()
        bore_list, hole_list, fing_list, bore_opt, hole_opt = interpret_single_file_lines(bore_lines)

    # holes
    hole_list2, hole_opt2 = interpret_data(hole_file)
    if len(hole_list)>0 and len(hole_list2)>0:
        warnings.warn('The holes data from the "single file" are ignored. Only the one from the "holes file/list" are kept.')
        hole_list = hole_list2
        hole_opt = hole_opt2
    elif len(hole_list2)>0: # if the "hole file" is not empty, use this data
        hole_list = hole_list2
        hole_opt = hole_opt2

    # fing_chart
    fing_list2 = interpret_data(fing_file)[0]
    if len(fing_list)>0 and len(fing_list2)>0:
        warnings.warn('The fingering data from the "single file" are ignored. Only the one from the "fingering chart file/list" are kept.')
        fing_list = fing_list2
    if not (isinstance(fing_list2, list) and len(fing_list2)==0):
        fing_list = fing_list2
    return bore_list, hole_list, fing_list, bore_opt, hole_opt

def interpret_single_file_lines(bore_lines:list):
    """
    Convert the lines of the unique file into the corresponding list and dict

    Parameters
    ----------
    bore_text : list
        The list of string lines read from the unique (or main bore) file.

    Returns
    -------
    bore_list : list
        The list witht he main bore data.
    hole_list : list
        the list with the holes/valves data.
    fing_list : list
        The list with the fing. chart data.
    bore_opt : dict
        the dictionnary with the main bore options (unit, diameter, etc.).
    hole_opt : dict
        the dictionnary with the holes/valves options (unit, diameter, etc.).

    """
    hole_list = list()
    fing_list = list()
    hole_opt = dict()

    # find the indices of the "header" lines, starting with a "*"
    header_indices = [k for k, s in enumerate(bore_lines) if s.startswith('*')]
    header_indices += [len(bore_lines)] # add the last index to easily parse the block of data

    if len(header_indices) < 2: # no header => not a single file,
        bore_list, bore_opt = interpret_lines(bore_lines)
    else: # at least 1 block => the first one is the main bore
        bore_opt = parse_options(bore_lines[:header_indices[0]])
        bore_list = parse_lines(bore_lines[header_indices[0]+1:header_indices[1]])
        if len(header_indices)>2: # if a second block it is the holes
            hole_list = parse_lines(bore_lines[header_indices[1]+1:header_indices[2]])
            hole_opt = bore_opt #in a single file the options are common fore bore and holes
        if len(header_indices)>3: # if a 3rd block, it is the fing. chart
            fing_list = parse_lines(bore_lines[header_indices[2]+1:header_indices[3]])
    return bore_list, hole_list, fing_list, bore_opt, hole_opt


def interpret_data(data):
    """
    Interpret the input data as a list of data.

    It is a very general method.
    If the data is a string  it is supposed to be the name of the file
    containing the data, wich is read by the method ```_read_file```.
    If it is a list, it is return identically.

    Parameters
    ----------
    data : list or string
        List of data or a filename.

    Returns
    -------
    List
        List of raw of data.

    """
    if isinstance(data, str):  # We can chose to take real csv files as inputs
        with open(data) as file:
            lines = file.readlines()
        return interpret_lines(lines)
    else:
        return data, dict()             # or using directly lists or ow object


def interpret_lines(lines):
    """
    Transcript list of lines from file in a list of data raw.

    It is a very general method which only read each line, split the
    text w.r. to whitespaces and organise it in a list of list.

    Parameters
    ----------
    lines : list of  string
        The list of the line string containing the data.

    Returns
    -------
    raw_parts : List
        List of raw data.

    """
    raw_parts = parse_lines(lines)
    geom_options = parse_options(lines)
    return raw_parts, geom_options

def interpret_parameter_data(param_data):
    """
    Interpret the data for a parameter (value, bounds, type).

    The numerical value `x` is extracted from `param`, following
    the different format possible (see the class docstring):
        - a float: `x`
        - the string: `'~x'`
        - the string: `'x_min<~x'`
        - the string: `'x_min<~x<x_max'`
        - the string: `'~x%'`

    Parameters
    ----------
    param_data : float or string
        Float or string containing the parameters value.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    value : TYPE
        DESCRIPTION.
    variable : TYPE
        DESCRIPTION.
    relative : TYPE
        DESCRIPTION.
    bounds : TYPE
        DESCRIPTION.

    """
    (lb, ub) = (-np.inf, np.inf)
    if isinstance(param_data, str):
        string_split = param_data.split(sep='<')
        indices_val = [n for n, s in enumerate(string_split) if s.startswith('~')]
        if len(string_split)==1 and len(indices_val)==0:
            variable = False
            relative = False
            value = float(param_data)
        elif len(indices_val) == 1:
            n_val = indices_val[0]
            variable = True
            val_str = string_split[n_val][1:]
            if val_str.endswith('%'):
                relative = True
                value = float(val_str[:-1])
            else:
                relative = False
                value = float(val_str)
            if n_val==1:
                lb = float(string_split[0])
                if len(string_split)==3:
                    ub = float(string_split[2])
            if n_val==0 and len(string_split)==2:
                ub = float(string_split[1])
        else:
            raise ValueError(("the parameters format: '{:s}' is not "
                              "recognized").format(param_data))
    else:
        variable = False
        relative = False
        value = float(param_data)
    bounds = (lb, ub)
    return value, variable, relative, bounds


def parse_lines(lines):
    raw_parts = []
    for line in lines:
        contents = parse_line(line)
        if len(contents) > 0:
            raw_parts.append(contents)
    return raw_parts


def parse_line(line):
    """
    Interpret each line as a list of string.

    Split the lines according to whitespace.
    Anything after a '#' is considered to be a comment

    Parameters
    ----------
    line : string
        A line string.

    Returns
    -------
    List
        List of string obtained from the line.

    """
    # Anything after a '#' is considered to be a comment
    line = line.split('#')[0]

    # Any line starting with '!' is considered to be an option and '*' is considered as header
    if line.startswith('!') or line.startswith('*'):
        return ''
    else:
        # Split the lines according to whitespace
        return line.split()


def parse_opening_factor(s):
    """
    Interpret the opening factor in the fingering chart data.

    Each hole is:
        - open if 'o' or 'open'
        - closed if 'x' or 'closed'
        - semi-closed if '0.5' or '.5'

    Each valve is:
    - 'x' "depressed" or "press down"
    - 'o' "raised" or "open"
    - '0.5' semi-pressed (why not!)

    Parameters
    ----------
    s : string
        string containing the information about the opening

    Returns
    -------
    float
        The opening factor between 0 (entirely closed) and 1 (entirely
        opened)

    """
    opening_factor_from_str = {
            'open': 1, 'o': 1,
            'closed': 0, 'x': 0, 'c': 0,
            '0.5': 0.5, '.5': .5
            }

    if s.lower() in opening_factor_from_str:
        return opening_factor_from_str[s.lower()]

    try:
        # Legacy behavior: 1 is closed, 0 is open
        factor = 1 - float(s)
        warnings.warn("Please use 'o' or 'open' for open holes"
                      " and 'x' or 'closed' for closed")
        return factor
    except ValueError:
        raise ValueError("Invalid string for open/close: {}"
                         .format(s))


def parse_options(lines):
    # get options lines which start with "!"
    options_lines = [s.split('!')[1].split('=') for s in lines if s.startswith('!')]
    geom_options = dict()
    for opt in options_lines:
        if len(opt)<2:
            raise ValueError("The options (line starting with '!') "
                             "must have the format: '! option = value'")
        opt = [s.split()[0] for s in opt]
        geom_options[opt[0]] = opt[1]
    if 'version' in geom_options.keys():
        check_version(geom_options['version'])
        geom_options.pop('version', None) # remove the version from the options
    return geom_options

def check_version(v):
    """
    Compare current OW version to the one indicated in v

    Print a warning if v is more recent than the current version of OW.

    Parameters
    ----------
    v : string
        The version to compare
    """
    # tuple(map(int, (v.split("."))))
    if version.parse(v) > version.parse(openwind.__version__):
        warnings.warn('The file was generated from a more recent version of openwind: {}.\n'
                      'Please update your version ({}) to avoid any troubles.'.format(v,
                                                                                  openwind.__version__))

def get_comments_from_lines(lines):
    bore_header = ['x0', 'x1', 'type']
    comment_lines = [s[1:].strip() for s in lines if s.startswith('#')
                     and not all(head in s[1:].split() for head in bore_header)] # exclude the header line of the main bore
    return '\n'.join(comment_lines)

def get_comments_from_file(filename):
    with open(filename) as file:
        lines = file.readlines()
    return get_comments_from_lines(lines)

def clean_label(label):
    """
    Remove space and # symbole from label

    Parameters
    ----------
    label : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    return label.replace(' ','_').replace('#','_sharp').replace('__','_')
