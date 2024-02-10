#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 13:49:48 2023

@author: augustin
"""

import os
import time
import numpy as np

from classe_threshold_blowing_pressure import ThresholdBlowingPressure


my_path = '/home/augustin/Documents/Recherches/OpenWIND/openwind/easy_of_play/'

MainBore = os.path.join(my_path, 'clarinet_avec_bec_MainBore.csv')
SideComponents =  os.path.join(my_path, 'clarinet_avec_bec_SideComponents.csv')
FingeringChart =  os.path.join(my_path, 'clarinet_FingeringChart.csv')
freq = np.linspace(50,2000,100)


my_thresh = ThresholdBlowingPressure(MainBore, SideComponents, FingeringChart, freq, zeta=[1.])
print(my_thresh.resonances['E3'])

# %%
t0 = time.time()
gamma1, f1, gamma2, f2, f0 = my_thresh.compute_threshold('E3')
print(time.time() - t0)

print(gamma1)
print(gamma2)
