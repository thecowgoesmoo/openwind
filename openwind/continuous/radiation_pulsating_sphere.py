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

from openwind.continuous import PhysicalRadiation


class RadiationPulsatingSphere(PhysicalRadiation):
    """
    Radiation impedance model based on a pulsating portion of a sphere.

    Formulas from [Helie_RadSph]_, equation (15). For a bell with an opening
    angle :math:`\\theta`:

    .. math::
        Z_r = \\frac{\\rho c }{\\pi R^2} \\frac{j \\alpha \\frac{\\nu}{\\nu_c}\
            - \\left( \\frac{\\nu}{\\nu_c} \\right)^2}\
            {1 + 2j \\xi \\frac{\\nu}{\\nu_c}\
            - \\left( \\frac{\\nu}{\\nu_c} \\right)^2}

    with :math:`R` the opening radius,  :math:`\\nu = R f/c` a scaled wavenumber,
    :math:`\\alpha, \\xi, \\nu_c` dimensionless coefficient depending on
    :math:`\\theta`

    .. danger::
        When :math:`\\theta` tends towards 0 (cylinder), the radiation impedance
        tends towards :math:`Z_r=Z_c` (anechoic radiation) and not the one
        of a cylinder!

    Parameters
    ----------
    theta : float
        Bell opening angle in radians.
    label : str
        the label of the radiation component
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}, optional
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
        The default is 'PH1'

    Parameters
    ----------
    theta : float
        Bell opening angle in radians.

    Attributes
    ----------
    theta : float
        Bell opening angle in radians.
    xi : float
        Dimensionless coefficient :math:`\\xi` depending on theta
    alpha : float
        Dimensionless coefficient :math:`\\alpha` depending on theta
    nu_c : float
        characteristic frequency coefficient :math:`\\nu_c` depending on theta

    References
    ----------
    .. [Helie_RadSph] Hélie, T., Hézard, T., Mignot, R., & Matignon, D. (2013).\
        One-dimensional acoustic models of horns and comparison with \
        measurements. Acta acustica united with Acustica, 99(6), 960-974.
    """

    def __init__(self, theta, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)
        self.theta = theta
        assert theta >= 0
        assert theta <= np.pi/2
        # Compute parameters of the model
        self.xi = 0.0207*theta**4 - 0.144*theta**3 + 0.221*theta**2 \
            + 0.0799*theta + 0.72
        self.alpha = 1/(0.1113*theta**5 - 0.6360*theta**4 + 1.162*theta**3
                        - 1.242*theta**2 + 1.083*theta + 0.8788)
        self.nu_c = 1/(-0.198*theta**5 + 0.2607*theta**4 - 0.424*theta**3
                       - 0.07946*theta**2 + 4.704*theta + 0.022)

    @property
    def name(self):
        return "pulsating_sphere"

    def __repr__(self):
        return "<RadiationPulsatingSphere(theta={:.5f})>".format(self.theta)

    def _open_imp(self, omegas_scaled, radius, rho, c):
        if self.theta > 0:
            r_sphere = radius / np.sin(self.theta)
            nu_ratio = omegas_scaled/c * r_sphere/(2*np.pi * self.nu_c)
            impedance = ((1j*self.alpha*nu_ratio - nu_ratio**2)
                         / (1 + 2j*self.xi*nu_ratio - nu_ratio**2))
        else:
            impedance = np.ones_like(omegas_scaled)
        return impedance * rho*c/(radius**2*np.pi)

    def get_impedance(self, omegas_scaled, radius, rho, c, opening_factor):
        return self._open_imp(omegas_scaled, radius, rho, c)/opening_factor**2

    def get_admitance(self, omegas_scaled, radius, rho, c, opening_factor):
        return opening_factor**2/self._open_imp(omegas_scaled, radius, rho, c)

    def get_diff_impedance(self, *args, **kwargs):
        """
        This radiation type can not yet be used in inversion
        """
        raise NotImplementedError

    def get_diff_admitance(self, *args, **kwargs):
        """
        This radiation type can not yet be used in inversion
        """
        raise NotImplementedError

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        """
        Not implemented for this type of radiation.

        It can not be used in temporal domain.
        """
        raise NotImplementedError('Currently, the pulsating sphere can not be'
                                  ' used in temporal domain')
