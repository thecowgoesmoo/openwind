#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script transforms examples files from the example folder to rst files in
the documentation source files (ie. docs/source/examples)

To make this script work, you have to respect some rules when writing examples :


- Copy paste the header first (This is important, see why in parse_example() )

.. code-block:: python

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

Then write a docstring that describes the example:

.. code-block:: python

    " " "
    Example file description
    " " "

Write the imports. The **first** import that starts with ``from openwind.module
import SomeClass `` will be added to the documentation with its reference as
follows

.. code-block:: python

    This example uses the :ref:`SomeClass <../modules/openwind.module.some_class>`
    class

- To create a title, use the following convention (``"Matlab"`` style)

.. code-block:: python

# %% This is a title

- Python code will be put after ``.. code-block:: python`` tag

- Comments will be transformed in raw text


If you are trying to understand this file, it was meant to transform python into
markdown, but then I added the parse_from_file line in order to transform it in
rst file, instead of writing it all over again.

It could be usefull to add a --md flag to keep the output as markdown
"""


import os
import sys
import ast
import re
import glob
import importlib
import getopt

from m2r2 import parse_from_file


# Add here the folders and files you don't want to update
SKIP_FOLDER = ['Ernoult-Chabassier-Rodriguez_Humeau_2021', 'Thibault-Chabassier_JASA2020',
               'Tournemenne-Chabassier_ACTA2019', 'Thibault-Chabassier-Boutin-Helie_JSV2022',
               'RR-humidity_Ernoult_2023', 'Thibault_Thesis2023']
SKIP_FILE = ['ALEXIS_cylinder_impulse_response_diffrepr']

# def usage():
#   print "\nThis is the usage function\n"
#   print 'Usage: '+sys.argv[0]+' -i <file1> [option]'


def main():
    # read user options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "ho:v", ["force-replace"])
    except getopt.GetoptError as err:
        # print help information and exit:
        usage()
        print(err)  # will print something like "option -a not recognized"
        sys.exit(2)
    force_flag = False
    verbose = False
    for o, a in opts:
        if o in ("-v", "--verbose"):
            verbose = True
        elif o in ("-f", "--force-replace"):
            force_flag = True
        else:
            assert False, "unhandled option"

    filepath_list = list()
    # If force-replace, just replace one file
    if force_flag:
        filepath_list.append(sys.argv[2])
    # Else: we will create examples for all files in openwind/examples folder
    else:
        for example_file in glob.glob('examples/*/*.py'):
            if (get_module_name(example_file) not in SKIP_FOLDER and
                get_filename(example_file) not in SKIP_FILE):
                filepath_list.append(example_file)

    for filepath in filepath_list:
        filename = get_filename(filepath)
        module = get_module_name(filepath)
        output = os.path.join("docs","source","examples",module+"."+filename+".rst")
        md_output = os.path.join("docs","source","examples",module+"."+filename+".md")

        # Skip file if "how to" already exist, except if force flag is true
        if os.path.exists(output) and force_flag == False:
            if verbose:
                print("WARNING: Example file %s already exists, skipping..."
                      %output)
                print("If you want to force rewrite this file, run"
                      "python update_howto.py --force-replace %s" %filepath)
                continue
        else:
            # Call to parse_example function
            file_content = parse_example(filepath)
            # Write to output
            pre, ext = os.path.splitext(output)
            md_output = pre + ".md"
            # TODO: add a --md flag to chose markdown over restructuredtext
            with open(md_output, 'w') as f:
                for item in file_content:
                    f.write("%s" % item)
                f.close()
            # try:
            rst_content = parse_from_file(md_output)
            with open(output, 'w') as f:
                f.write(rst_content)
                f.close()
            # os.remove(md_output)
            # except:
            #     print("Couldn't parse/write example file in rst format."
            #           "Please try in a command line \n m2r2 %s " %md_output)


def get_filename(filepath):
    """Returns the example file name without extension"""
    base = os.path.basename(filepath)
    return os.path.splitext(base)[0]

def get_module_name(filepath):
    """Returns the folder in which example file is"""
    dir = os.path.dirname(filepath)
    return os.path.basename(dir)


def parse_example(filepath):
    """ Converts examples python files to documentation examples"""
    contents = list()
    # Parse filename to title
    print("filepath:", filepath)
    filename_list = [f.capitalize() for f in get_filename(filepath).split("_")]
    # If there is an example number in the filename, we add it in the title
    try:
        ex_number = re.search(r"\d+?", filename_list[0]).group(0)
        title = str("# Ex. %s: %s" %(ex_number, " ".join(filename_list[1:])))
    except:
        title = str("# %s " %(" ".join(filename_list[1:])))
    contents.append(title)
    contents.append("\n \n")

    # Parse docstring to subtitle
    with open(filepath, 'r') as f:
        docstring = ast.get_docstring(ast.parse(f.read()))
        contents.append(docstring)
        contents.append("\n")
        f.close()

    # Add link to source file
    source_ref = "Source file available [here](https://gitlab.inria.fr/openwind/openwind/-/blob/master/%s).\n"%(filepath)
    contents.append('\n' + source_ref)

    # Add the module documentation location
    with open(filepath, "r") as f:
        lines = f.readlines()
        mod = get_module_name(filepath)
        for line in lines:
            if (line.startswith("from openwind")):
                cls = line.split("import ")[1][:-1]
                cls_mod = re.findall('[A-Z][^A-Z]*', cls)
                if mod in line:
                    doc_path = mod + "." + "_".join(cls_mod).lower()
                else:
                    doc_path =  "_".join(cls_mod).lower()
                contents.append("\nThis example uses the [%s]"
                                "(../modules/openwind.%s) class.\n\n"
                                %(cls,doc_path))
                break
        f.close()

    # Create imports title
    contents.append("\n" + "## Imports")
    contents.append("\n")

    # remove headers
    del lines[:21]

    # Parse comments to text and code to markdown code
    contents.append("\n" + "```python" + "\n")
    for line in lines:
        if not (line.startswith('"""') or line.find(docstring) != -1):
            if line.startswith("# %% ") or line.startswith("#%% "):
                # section titles
                contents.append("```" + "\n")
                contents.append("\n" + line.replace("# %% ","## ").title() + "\n")
                contents.append("\n" + "```python" + "\n")
            elif line.startswith("# %%% ") or line.startswith("#%%% "):
                # section titles
                contents.append("```" + "\n")
                contents.append("\n" + line.replace("# %%% ","### ").title() + "\n")
                contents.append("\n" + "```python" + "\n")

            elif line == "#\n":
                # empty comment lines
                contents.append("```" + "\n")
                contents.append("\n")
                contents.append("\n" + "```python" + "\n")
            elif line.startswith("#"):
                # Comments become text
                contents.append("```" + "\n")
                contents.append(line.replace("# ",""))
                contents.append("\n" + "```python" + "\n")
            elif line != "\n":
                contents.append(line)
    contents.append("```" + "\n")

    # remove useless ```python ``` from list
    reduntant_index = []
    for i, x in enumerate(contents):
        if x == "\n" + "```python" + "\n":
            if contents[i+1] == "```" + "\n":
                reduntant_index.append(i)
                reduntant_index.append(i+1)

    # returns contents list except reduntant_index
    return [i for j, i in enumerate(contents) if j not in reduntant_index]


if __name__ == "__main__":
    main()
