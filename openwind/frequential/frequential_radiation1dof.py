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
from scipy.sparse import lil_matrix
import pdb
from openwind.continuous import RadiationPade
from openwind.frequential import FrequentialRadiation


class FrequentialRadiation1DOF(FrequentialRadiation):
    """
    Compute for every frequency the radiation data for the linear system to
    solve

    .. math::

        A_h U_h = L_h

    This component contributes to the matrix :math:`A_h`, and the dependence on
    frequency is linear :

    .. code-block::

                         ┌                ┐
                         │ .  .  .   .  . │
                         │ .  .  .   .  . │
            Ah_contrib = │ .  .  βY -√ɑ . │ ← line of the pipe end's d.o.f.
                         │ .  .  √ɑ jωZ . │ ← line of the radiation's d.o.f.
                         │ .  .  .   .  . │
                         └                ┘

    where Y = 1/Z = S/ρc is the characteristic impedance at the opening.
    This assumes 'PH1' convention, and the use of a Padé radiation.

    Parameters
    ----------

    rad : :py:class:`RadiationPade <openwind.continuous.physical_radiation.RadiationPade>`
        the radiation model used

    freq_ends : tuple of one :py:class:`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>` or \
        :py:class:`TMMPipeEnd <openwind.frequential.frequential_pipe_tmm.TMMPipeEnd>`
        Frequential Pipe end associated to this radiation condition.
    opening_factor: float, optional
        The ratio of opening of the radiating opening (1: open, 0: closed).
        Defaults to 1.0

    """

    def __init__(self, rad, freq_ends, opening_factor=1.0):        
        super().__init__(rad, freq_ends, opening_factor=1.0)
        assert isinstance(rad, RadiationPade)

    
    def get_number_dof(self):
        return 1

    def get_contrib_Mh(self):
        idR = self.get_first_index() # Index of this FRadiation's d.o.f.
        radius, rho, celerity = self.freq_end.get_physical_params()       
        #kr, Zc, alpha, beta = self.rad._rad_coeff(omegas_scaled, radius, rho, celerity, self._opening_factor)
        Zc = self.rad._get_caract_impedance(radius, rho, celerity) 
        #pdb.set_trace()
        #Mh = np.zeros(self.ntot_dof,
        #                    dtype='float64')
        #Mh[idR] = Zc
        #return Mh
        return idR, Zc
    
    def get_contrib_Kh(self):
        idR = self.get_first_index() # Index of this FRadiation's d.o.f.
        idE = self.freq_end.get_index() # Index of the PipeEnd's d.o.f
        radius, rho, celerity = self.freq_end.get_physical_params()

        Zc = self.rad._get_caract_impedance(radius, rho, celerity) # rho*celerity / (np.pi*radius**2)
        alpha = self.rad.alpha_unscaled * self._opening_factor 
        alpha = alpha*celerity/radius * self.rad.scaling.get_time()
        beta = self.rad.beta * self._opening_factor**2

        data = np.array([beta/Zc, -np.sqrt(alpha), np.sqrt(alpha)])
        row = np.array([idE, idE, idR])
        col = np.array([idE, idR, idE])
        
        return row, col, data
        # Kh = lil_matrix((self.ntot_dof, self.ntot_dof),
        #                     dtype='float64')
        # #pdb.set_trace()
        # Kh[idE, idE] = beta/Zc
        # Kh[idE, idR] = -np.sqrt(alpha)
        # Kh[idR, idE] = np.sqrt(alpha)
        # return Kh

    def get_contrib_freq(self, omegas_scaled):
        idR = self.get_first_index() # Index of this FRadiation's d.o.f.
        radius, rho, celerity = self.freq_end.get_physical_params()
        kr, Zc, alpha, beta = self.rad._rad_coeff(omegas_scaled, radius, rho, celerity, self._opening_factor)
        Zc = Zc / self.rad.scaling.get_impedance()
        
        my_contrib = 1j*kr*Zc*celerity/radius  
        return idR, my_contrib
    
        # Ah_diags = lil_matrix((self.ntot_dof, len(omegas_scaled)),
        #                     dtype='complex128')
        # Ah_diags[idR, :] = 1j*kr*Zc*celerity/radius  
        # return Ah_diags

    def get_contrib_indep_freq(self):
        idR = self.get_first_index() # Index of this FRadiation's d.o.f.
        idE = self.freq_end.get_index() # Index of the PipeEnd's d.o.f
        radius, rho, celerity = self.freq_end.get_physical_params()

        Zc = rho*celerity / (np.pi*radius**2)  / self.rad.scaling.get_impedance()
        alpha = self.rad.alpha_unscaled * self._opening_factor  
        alpha = alpha*celerity/radius* self.rad.scaling.get_time()
        beta = self.rad.beta * self._opening_factor**2

        #Ah_nodiag = lil_matrix((self.ntot_dof, self.ntot_dof),
        #                    dtype='complex128')
        
        data = np.array([beta/Zc, -np.sqrt(alpha), np.sqrt(alpha)])
        row = np.array([idE, idE, idR])
        col = np.array([idE, idR, idE])
        
        return row, col, data
    
        #Ah_nodiag[idE, idE] = beta/Zc
        #Ah_nodiag[idE, idR] = -np.sqrt(alpha)
        #Ah_nodiag[idR, idE] = np.sqrt(alpha)

        #return Ah_nodiag
