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

"""Tools for parsing files from other software, and writing OW compatible file."""

import numpy as np
import os

from openwind.technical import parser
from openwind import InstrumentGeometry

def convert_RESONANS_file(filename):
    # add suffix '_OW' to avoid the deletion of source file if a csv is given
    file_ow = os.path.splitext(filename)[0] + '_OW.csv'

    main_bore_OW, holes_OW, fing_chart_OW = convert_RESONANS_to_OW(filename)
    instru_ow = InstrumentGeometry(main_bore_OW, holes_OW, fing_chart_OW)
    instru_ow.write_files(file_ow)
    return file_ow

    
def convert_RESONANS_to_OW(filename):
    """
    Convert RESONANS file .dat into 3 list interpretable by opewind.InstrumentGeometry
    
    .. code-block:: python
        
        main_bore, holes, fing_chart = RESONANS_2_Ow_lists(my_file.dat)
        my_geom = InstrumentGeometry(main_bore, holes, fing_chart)
        

    Parameters
    ----------
    filename : str
        The path/filename of the file to convert.


    Returns
    -------
    main_bore_OW : list
        The list corresponding to the main bore geometry
    hole_OW : list
        The list of holes geometry
    fing_chart_OW : list
        The list corresponding to the fingering chart

    """
    # read RESONANS file format (hole not accounted for yet)
    with open(filename,'r') as file:
        lines = [l.split('\n')[0] for l in file.readlines()]
    main_bore_OW, holes_OW, fing_chart_OW = parse_RESONANS_lines_2_OW(lines)
    return main_bore_OW, holes_OW, fing_chart_OW
        
        
        
def parse_RESONANS_lines_2_OW(lines):
    """
    Convert the content of a resonans file into ow list
    
    Spliting file reading and content parsing is necessary for the graphical interface

    Parameters
    ----------
    lines : string
        The content of the resonans file.

    Returns
    -------
    main_bore_OW : list
        The list corresponding to the main bore geometry
    hole_OW : list
        The list of holes geometry
    fing_chart_OW : list
        The list corresponding to the fingering chart

    """
    # =============================================================================
    # Header
    # =============================================================================

    # first lines are special  
    N = int(lines[0])
    Nb_holes = int(lines[1])
    # if(Nb_holes > 0):
    #     raise ValueError("Holes are not yet accounted for in RESONANS format. "
    #                      +"Please format your instrument in a native Openwind format. "+
    #                      "Feel free to send us a RESONANS file with holes along with format explanations.")
    Nb_fing = int(lines[2])
    temperature = float(lines[3])
    comp_type = lines[4]
    name = lines[5]

    # =============================================================================
    # Geometry
    # =============================================================================

    # the lines corresponding to the geometry (the last ones are for RESONANS visualization)
    geom_lines = lines[6:5*(N)+6]

    # data are provided in blocks of 5 lines with different nature following if it is a main bore element or a hole
    # we create groups of 5 lines to treat them together
    geom_list = [geom_lines[5*k: 5*(k+1)] for k in range(N)]

    # init lists
    input_radii = list()
    output_radii = list()
    location = [0]

    # holes_names = ['hole' + str(Nb_holes - k) for k in range(Nb_holes)]
    holes_names = list()
    loc_holes = list()
    rad_holes = list()
    length_holes = list()

    k_block = 0
    for n_mb in range(N - Nb_holes):
        mb_block = geom_list[k_block]
        
        # For main bore the 5 parameters are: L; R_in; R_out; nb_holes; nb_subdiv
        L = float(mb_block[0])
        location.append(location[-1] + L) 
        input_radii.append(float(mb_block[1]))
        output_radii.append(float(mb_block[2]))
        
        k_block += 1
        n_block_holes = int(float(mb_block[3]))
        for k_hole in range(n_block_holes):
            hole_block = geom_list[k_block]
            # I assume that the first line is the location on the considered main bore segment
            # so the location of the hole is the current location - this length
            loc_holes.append(location[-1] - float(hole_block[0]) )
            # I assume that the second is the radius
            rad_holes.append(float(hole_block[1]))
            # the 3rd is the chimney length
            length_holes.append(float(hole_block[2]))
            #  the 4th is the "open" length of the chimney we can not keep that in OW file
            if float(hole_block[3])  != -1*length_holes[-1]:
                raise ValueError("Openwind ne permet pas d'avoir des hauteurs différentes pour les trous ouverts et fermés.")
            # the 5th is hole "number"
            holes_names.append(f'hole{hole_block[4]}')
            k_block += 1
            
    # =============================================================================
    # The main bore            
    # =============================================================================
    # the instrument is defined from the bell : we flip it and change th location orientation
    x_in = [max(location) - x for x in location[:0:-1]]
    x_out = [max(location) - x for x in location[-2::-1]]
    input_radii.reverse()
    output_radii.reverse()
    # store everythin in a list of list as for ow
    main_bore_OW = [list(a) + ['linear'] for a in zip(x_in, x_out, input_radii, output_radii)]

    # =============================================================================
    # The holes
    # =============================================================================
    # we also reverse the order of the hole for convenience and we reverse the location
    x_holes = [max(location) - x for x in loc_holes[::-1]]
    holes_names.reverse()
    rad_holes.reverse()
    length_holes.reverse()

    # the header necessary for OW files and list
    holes_OW = [['label', 'position', 'radius', 'length']]
    holes_OW += [list(a) for a in zip(holes_names, x_holes, rad_holes, length_holes)]

    # =============================================================================
    # Fingering chart
    # =============================================================================
    if Nb_holes > 0:
        # the end of the file corresponds to the fingering chart
        fing_chart_lines = lines[5*(N)+6:]
        # each fingering as 6 head lines + 1 line per hole
        fing_chart_list = [fing_chart_lines[(Nb_holes+6)*k: (Nb_holes+6)*(k+1)] for k in range(Nb_fing)]
        
        
        fing_chart = [['label'] + holes_names + ['bell']]
        for fing in fing_chart_list:
            # the 5th line corresponds to the note name 
            note = parser.clean_label(fing[5])
            if note == "":
                note = "no_name"
                
            # I think that the first line is the "bell" state
            bell = ['x' if fing[0]=='0' else 'o']
            #then we convert each "hole state" in the corresponding OW format
            # we take care of reversing the hole order
            fing_chart.append([note] + ['x' if s=='0' else 'o' for s in fing[:5:-1]] + bell)
            
        fing_chart_OW = [[row[i] for row in fing_chart] for i in range(len(fing_chart[0]))]
        # transposed_tuples = list(zip(*fing_chart))
    else:
        fing_chart_OW = None

    return main_bore_OW, holes_OW, fing_chart_OW


# %% Parsing methods

# the pafi files can be written in english or french....
fr2eng = {'ordre':'ordering', 'ordering':'ordering',
          'nom':'name', 'name':'name',
          'longueur':'length', 'length':'length',
          "diametre d'entree":"input_diameter", "input_diameter":"input_diameter",
          "diametre de sortie":"output_diameter", "output_diameter":"output_diameter",
          "angle":"angle", "type":"type",
          "position de debut":"start_positioning", "start_positioning":"start_positioning",
          "diametre":"diameter", "diameter":"diameter",
          "hauteur":"height", "height":"height",
          "enfoncement":"sinking", "sinking":"sinking",
          "azimut":"azimuth", "azimuth":"azimuth"}

def read_pafi_lines(lines):
    """
    Read file and transcript it in a list of data raw.

    It is a very general method which only read the file, split the
    text w.r. to lines and ';' and organise it in a list of list.

    Parameters
    ----------
    lines : string
        The string with the file content

    Returns
    -------
    raw_parts : List
        List of raw data.

    """
    raw_parts = []
    for line in lines:
        contents = parse_pafi_line(line)
        if len(contents) > 0:
            raw_parts.append(contents)
    return raw_parts


def parse_pafi_line(line):
    """
    Interpret each line as a list of string.

    Split the lines according to ';'.
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
    # remove trailing space
    line = line.strip()
    # Split the lines according to ;
    return line.split(';')

def convert_PAFI_elements_file(bore_file):
    """
    Convert the Pafi file "elements" with the main bore in list compatible with openwind

    Parameters
    ----------
    bore_file : string
        The path of the "elements"  (main bore) file.

    Returns
    -------
    raw_OW : list
        list_compatible with openwind.

    """
    
    with open(bore_file) as file:
        lines = file.readlines()
    return parse_pafi_elements(lines)
    
def parse_pafi_elements(lines):
    raws = read_pafi_lines(lines)
    col = [fr2eng[label] for label in raws[0]]
    dict_pafi = list()
    ordering = list()
    for line in raws[1:]:
        dict_pafi.append(dict(zip(col, line)))
        ordering.append(float(dict_pafi[-1]['ordering']))
    dict_sorted = [x for _, x in sorted(zip(ordering, dict_pafi))]

    raw_OW = list()
    x_last = 0
    for dict_part in dict_sorted:
        line_OW = [x_last, x_last+float(dict_part['length'])*1e-3,
                   float(dict_part['input_diameter'])*1e-3/2,
                   float(dict_part['output_diameter'])*1e-3/2,
                   'cone'] # dict_part['type']]
        if dict_part['type'] not in  ['Cylinder', 'Cone', 'Cylindre']:
            print('Warnings: currently the type "{}" is treated as a '
                  'cone'.format(dict_part['type']))
        x_last = line_OW[1]
        raw_OW.append(line_OW)

    return raw_OW

def convert_PAFI_holes_file(holes_file):
    """
    Convert the Pafi file "holes" (trou lateraux) with the holes data in list compatible with openwind

    Parameters
    ----------
    holes_file : string
        The path of the "holes" file.

    Returns
    -------
    raw_OW : list
        list_compatible with openwind.

    """    
    with open(holes_file) as file:
        lines = file.readlines()
    return parse_pafi_holes(lines)
    
def parse_pafi_holes(lines):
    raw_holes_pafi = read_pafi_lines(lines)
    raw_OW = [['label', 'position', 'chimney', 'radius']]
    col = [fr2eng[label] for label in raw_holes_pafi[0]]
    for line in raw_holes_pafi[1:]:
        dict_part = dict(zip(col, line))
        hole_label = parser.clean_label(dict_part['name'])
        line_OW = [hole_label, float(dict_part['start_positioning'])*1e-3,
                   float(dict_part['height'])*1e-3,
                   float(dict_part['diameter'])/2*1e-3]
        x_last = line_OW[1]
        raw_OW.append(line_OW)
    return raw_OW

def convert_PAFI_fingerings_file(fing_file):
    """
    Convert the Pafi file "fingerings" (Doigtes) with the fingering chart in list compatible with openwind

    Parameters
    ----------
    fing_file : string
        The path of the "fingerings"  file.

    Returns
    -------
    raw_OW : list
        list_compatible with openwind.

    """
    with open(fing_file) as file:
        lines = file.readlines()
    return parse_pafi_fing(lines)
    
def parse_pafi_fing(lines):
    raw_pafi = read_pafi_lines(lines)
    notes = [note.split(' (')[0] for note in raw_pafi[0][2:]]
    notes_ok = [parser.clean_label(n) for n in notes]
    raw_OW = [['label'] + notes_ok]
    conv_state = {'open':'open', 'closed':'closed', 'half-open':'0.5'}
    for line in raw_pafi[1:]:
        hole_label = parser.clean_label(line[0])
        raw_OW.append([hole_label] + [conv_state[state] for state in line[2:]])
    return raw_OW

def convert_PAFI_to_OW(elements_file, holes_file=None, fingerings_file=None):
    """
    Convert Pafi files into list compatible with openwind

    Parameters
    ----------
    elements_file : string
        The path of the "elements"  (main bore) file.
    holes_file : string, optional
        The path of the "holes" file.  The default is None.
    fingerings_file : string, optional
        The path of the "fingerings"  file. The default is None.

    Returns
    -------
    bore : list
        The main bore list compatible with openwind.
    holes : list
        The holes list compatible with openwind..
    fing_chart : list
        The fingering chart list compatible with openwind..

    """
    bore = convert_PAFI_elements_file(elements_file)
    holes = list()
    fing_chart = list()
    if holes_file is not None:
        holes = convert_PAFI_holes_file(holes_file)
    if fingerings_file is not None:
        fing_chart = convert_PAFI_fingerings_file(fingerings_file)
    return bore, holes, fing_chart


def read_pafi_impedance(filename):
    """
    Read a pafi impedance file.

    Parameters
    ----------
    filename : string
        The name of the file containing the impedance (with the extension).

    Returns
    -------
    frequencies : np.array of float
        The frequencies at which is evaluated the impedance.
    impedance : np.array of float
        The complexe impedance at each frequency.


    """

    with open(filename) as file:
        lines = file.readlines()
    file_freq = []
    file_imped = []
    for line in lines[1:]:
        contents = parse_pafi_line(line)
        if len(contents) > 0:
            file_freq.append(float(contents[1]))
            file_imped.append(float(contents[2]) + 1j*float(contents[3]))
    frequencies = np.array(file_freq)
    impedance = np.array(file_imped)
    frequencies = frequencies[~np.isnan(file_imped)]
    impedance = impedance[~np.isnan(impedance)]
    return frequencies, impedance


