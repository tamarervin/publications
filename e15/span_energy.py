#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 12 17:30:03 2024

@author: tamarervin
"""

import numpy as np
import pyspedas
from pytplot import tplot, get_data, cdf_to_tplot
time_range = ['2023-03-17/00:00', '2023-03-17/06:00']
pvars = pyspedas.psp.spi(trange=time_range, datatype='sf00_l3_mom', 
                            level='l3', time_clip=True)
tplot(['psp_spi_VEL_INST'])
vinst = get_data('psp_spi_VEL_INST')
vinstx = vinst.y[:, 0]
vinsty = vinst.y[:, 1]
vinstz = vinst.y[:, 2]
vxm, vym, vzm = np.nanmean(vinstx), np.nanmean(vinsty), np.nanmean(vinstz)
vmag = np.sqrt(vxm**2 + vym**2 + vzm**2)
print(vxm, vym, vzm, vmag)