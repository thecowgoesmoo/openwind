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


""" Test all the examples files"""


import os
import subprocess
import sys



path = os.path.dirname(os.path.abspath(__file__))

all_loc = [os.path.join(path, f) for f in os.listdir(path)]
files_loc = [f for f in all_loc if os.path.isfile(f) and f.endswith('.py') and f!=__file__]
folders = [f for f in all_loc if os.path.isdir(f)]

failed = list()
for file in files_loc:
    # if file in files_to_test:
        print(file)
        echec = subprocess.call([sys.executable, file])
        if echec:
            failed.append(file)

# %%
for folder in folders:
    os.chdir(path)
    fold_files = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f)) and f.endswith('.py') ]
    os.chdir(folder)
    for file in fold_files:
        # if file in files_to_test:
            print(file)
            echec = subprocess.call([sys.executable, file])
            if echec:
                failed.append(file)


print('\n\n==============================\n Result'
      '\n==============================')
print('{} examples failed:'.format(len(failed)))

for failed_name in failed:
    print(f'{failed_name}\n')


# %%
