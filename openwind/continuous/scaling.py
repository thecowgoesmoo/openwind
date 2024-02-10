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
Compute the coefficient used to scale (normalize) the physical equations.
"""

import numpy as np


class Scaling:
    """
    Compute the coefficients used to scale (normalize) the physical equations.

    The scaling coefficients are computed from different reference coefficients

    - :math:`S^{\\ast}` : the reference cross section area
    - :math:`\\rho^{\\ast}` : the reference air density
    - :math:`c^{\\ast}` : the reference sound celerity
    - :math:`\\ell^{\\ast}` : the reference length

    Parameters
    ----------
    section : float, optional
        The reference cross section area. The default is 1.
    rho : float, optional
        The reference air density. The default is 1.
    c : float, optional
        The reference sound celerity. The default is 1.
    length : float, optional
        The reference length. The default is 1.

    """
    def __init__(self, section=1, rho=1, c=1, length=1, is_time_scaled=True):
        self._section = section
        self._rho = rho
        self._c = c
        self._length = length
        self._is_time_scaled = is_time_scaled

    def __repr__(self):
        if all([val==1 for val in self.__dict__.values()]):
            return "<openwind.continuous.Scaling('unscaled')>"
        else:
            attributes = ('section={:.3g}, rho={:.2g}, c={:.4g}, '
                          'length={:.3g}'.format(self._section, self._rho,
                                                 self._c, self._length))
            return "<openwind.continuous.Scaling({})>".format(attributes)

    def __str__(self):
        msg = "Scaling Coefficients:"
        msg += "\n\tCross-section:{}".format(self._section)
        msg += "\n\tAir Density:{}".format(self._rho)
        msg += "\n\tSound Celerity:{}".format(self._c)
        msg += "\n\tLength:{}".format(self._length)
        return msg

    def set_nondimensionalization(self, pipe_ref):
        """
        The entrance of the designated pipe is used as the reference.

        The reference coefficients are set equal to their value at the entrance
        (x=0) of the specified "referent pipe". It corresponds typically to the entrance
        of the instrument.

        Parameters
        ---------
        pipe_ref : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
            The referent pipe.
        """
        physics_ref = pipe_ref.get_physics()
        self._section = pipe_ref.get_radius_at(0)**2 * np.pi
        self._rho = physics_ref.rho(0)
        self._c = physics_ref.c(0)
        self._length = physics_ref.c(0)#pipe_ref.get_length()

    def set_is_time_scaled(self, is_time_scaled):
        self._is_time_scaled = is_time_scaled

    def reset_defaultscaling(self):
        """
        Reset all the scaling coefficient to 1.
        """
        self._section = 1
        self._rho = 1
        self._c = 1
        self._length = 1

    def get_time(self):
        """
        Compute the reference time used to scale the equations.

        .. math::
            T^{\\ast} = \\frac{\\ell^{\\ast}}{c^{\\ast}}

        Returns
        -------
        float
            The reference time.

        """
        if self._is_time_scaled==True:
            return self._length/self._c
        else:
            return 1

    def get_impedance(self):
        """
        Compute the reference impedance used to scale the equations.

        .. math::
            Z^{\\ast} = \\frac{\\rho^{\\ast} c^{\\ast}}{S^{\\ast}}

        Returns
        -------
        float
            The reference impedance.

        """
        return self._rho*self._c/self._section

    def get_scaling_flow(self):
        """
        Compute the reference flow used to scale the equations.

        .. math::
            u^{\\ast} =  S^{\\ast} c^{\\ast}

        Returns
        -------
        float
            The reference flow.

        """
        return self._section * self._c

    def get_scaling_pressure(self):
        """
        Compute the reference pressure used to scale the equations.

        .. math::
            p^{\\ast} = \\rho^{\\ast} c^{\\ast 2}

        Returns
        -------
        float
            The reference pressure.

        """
        return self._rho * self._c**2

    def get_scaling_Zv(self):
        """Scaling coefficient associated to the flow equation.

        This coefficient is used to nondimensionalize :math:`Z_v`:

        .. math::
            Z_v^{\\ast} = Z_c^{\\ast}  T^{\\ast}

        where:

        - :math:`Z_c^{\\ast}` : the reference characteristic impedance
        - :math:`T^{\\ast}` : the reference time

        See Also
        --------
        :py:class:`Scaling <openwind.continuous.scaling.Scaling>`


        Returns
        -------
        float
            The coefficient :math:`Z_v^{\\ast}`

        """
        return self.get_impedance() * self.get_time()

    def get_scaling_Yt(self):
        """
        Scaling coefficient associated to the pressure equation.

        This coefficient is used to nondimensionalize :math:`Y_t`:

        .. math::
            Y_t^{\\ast} = \\frac{T^{\\ast}}{Z_c^{\\ast}}

        where:

        - :math:`Z_c^{\\ast}` : the reference characteristic impedance
        - :math:`T^{\\ast}` : the reference time

        See Also
        --------
        :py:class:`Scaling <openwind.continuous.scaling.Scaling>`


        Returns
        -------
        float
            The coefficient :math:`Y_t^{\\ast}`

        """
        return self.get_time() / self.get_impedance()

    def get_scaling_power(self):
        return self.get_scaling_flow()*self.get_scaling_pressure()
