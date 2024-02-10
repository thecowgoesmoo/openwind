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
from scipy.optimize import least_squares

from openwind.continuous import PhysicalRadiation, Physics, RadiationPade
from openwind.impedance_tools import read_impedance


def radiation_from_data(datasource, data_temperature, data_radius, label,
                        scaling, convention, **physics_kwargs):
    """
    Construct a RadiationModel object from radiation data.

    .. warning::
        The impedance must be scaled by the characteristic impedance of the
        opening :math:`\\rho c / S`

    .. warning::
        The temperature and the radius used for the simulation
        must be consistant (equal) with the one used to obtain the data!

    Parameters
    ----------
    datasource : string or tuple of np.array
        The source of the data. It can be a filename in which the data are
        saved, or a tuple of arrays containing the frequencies and the complex
        impedance.
    data_temperature : float
        Temperature of the data.
    data_radius : float
        Radius of the radiating opening of the data.

    Returns
    -------
    :py:class:`RadiationFromData<openwind.continuous.radiation_from_data.RadiationFromData>`
        A :py:class:`PhysicalRadiation\
        <openwind.continuous.physical_radiation.PhysicalRadiation>` object \
        build from data.

    """
    if type(datasource) == str:
        data = read_impedance(datasource)
    elif type(datasource) == tuple:
        data = datasource
    return RadiationFromData(data, data_temperature, data_radius, label,
                             scaling, convention, **physics_kwargs)


class RadiationFromData(PhysicalRadiation):
    """
    Impose a radiation condition from a data.

    The data contains a radiation impedance (measured or simulated).

    .. warning::
        The impedance must be scaled by the characteristic impedance of the
        opening :math:`\\rho c / S`

    .. warning::
        In temporal domain, the coefficients of the Padé development are
        obtained here by fitting the impedance given in the data. Precautions
        must be taken if the radius is different that the one used to obtain
        the reference impedance.

    .. warning::
        The temperature and the radius used for the simulation
        must be consistant (equal) with the one used to obtain
        the data!

    Parameters
    ----------
    data : tuple of np.array
        tuple of two arrays containing: the frequency axis and the complex
        impedance values
    data_temperature : float
        Temperature of the data.
    data_radius : float
        Radius of the radiating opening of the data
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

    Attributes
    ----------
    data_imped : np.array
        The complex impedance vector
    data_kr : np.array
        The scaled wavenumber :math:`kr` obtained from the data frequency and
        the data radius
    rad_pade : :py:class:`RadiationPade\
        <openwind.continuous.physical_radiation.RadiationPade>`
        A first order Padé development obtained by fitting the data for kr<1.
        It is used in the temporal domain.
    """

    def __init__(self, data, data_temperature, data_radius, label, scaling,
                 convention='PH1', **physics_kwargs):
        super().__init__(label, scaling, convention)
        self._set_impedance(data, data_temperature, data_radius, **physics_kwargs)
        self._set_pade_coeffs()

    def _set_impedance(self, data, data_temperature, data_radius,
                       **physics_kwargs):
        data_freq, self.data_imped = data
        rho, c = Physics(data_temperature, **physics_kwargs).get_coefs(0, 'rho', 'c')
        self.data_kr = data_freq * 2*np.pi / c * data_radius

        med_imped = np.median(np.abs(self.data_imped))
        if med_imped > 1e3:
            self.data_imped /= rho * c / (np.pi * data_radius**2)
            msg = ('The radiation impedance modulus is particularly '
                   'high (median={:.2e}), it has been automatically normalized'
                   ' by the characteristics impedance.'.format(med_imped))
            warnings.warn(msg)

    @property
    def name(self):
        return ('kr = [{:.2g}, {:.2g}]').format(np.min(self.data_kr),
                                                np.max(self.data_kr))

    def __str__(self):
        return "Radiation from data: ({})".format(self.name)

    def get_impedance(self, omegas, radius, rho, c, opening_factor):
        """
        Radiation impedance value at a given pulsation.

        Compute the radiation condition (impedance or admittance depending on
        the convention) at the given puslation `omegas` by interpolating the
        reference radiation impedance indicated in the data.

        Parameters
        ----------
        omegas : float, np.array of float, list of float
            The pulsations at which are computed the radiation condition (s^-1)
        radius : float
            Radius of the opening.
        rho : float
            Air density at the opening.
        c : float
            Sound celerity at the opening.
        opening_factor : float
            Factor between 0 and 1 which allows to close the opening.

        Raises
        ------
        ValueError
            If some puslations at which the radiation condition must be
            computed are outside the range in which the reference impedance is
            given, an error occurs.

        Returns
        -------
        radiation : np.array
            The radiation impedance at the given pulsations.

        """
        kr = omegas * radius / c
        min_kr = np.min(self.data_kr)
        max_kr = np.max(self.data_kr)
        tresh = 1e-5  # 1e-7
        if np.max(kr) > max_kr + tresh or np.min(kr) < min_kr - tresh:
            raise ValueError('The normalized wavenumber range at which must be'
                             ' evaluated the impedance: '
                             '[{:.2g}, {:.2g}]'.format(np.min(kr), np.max(kr))
                             + ' is wider than the range of the given data: '
                             '[{:.2g}, {:.2g}].'.format(min_kr, max_kr))
        Zc = rho*c/(np.pi*radius**2) / opening_factor
        return np.interp(kr, self.data_kr, Zc*self.data_imped)

    def get_admitance(self, omegas, radius, rho, c, opening_factor):
        Zr = self.get_impedance(omegas, radius, rho, c, 1)
        return opening_factor/Zr

    def get_diff_impedance(self, dr, omegas, radius, rho, c, opening_factor):
        raise NotImplementedError('It is impossible to optimize the radius of '
                                  'an  opening for which the radiation is '
                                  'interpolated from data.')

    def get_diff_admitance(self, dr, omegas, radius, rho, c, opening_factor):
        raise NotImplementedError('It is impossible to optimize the radius of '
                                  'an  opening for which the radiation is '
                                  'interpolated from data.')

    # %% Fit the pade development

    @staticmethod
    def residual(params, kr, imped):
        Zr_pade = 1j*kr / (params[1] + 1j*kr*params[0])
        residual = Zr_pade - imped
        return np.append(residual.real, residual.imag)

    @staticmethod
    def jacobian(params, kr, imped):
        grad = np.zeros((2*len(kr), 2))
        diff_gamma = -1j*kr / (params[1] + 1j*kr*params[0])**2
        diff_beta = (kr / (params[1] + 1j*kr*params[0]))**2
        grad[:, 0] = np.append(diff_beta.real, diff_beta.imag)
        grad[:, 1] = np.append(diff_gamma.real, diff_gamma.imag)
        return grad

    def _set_pade_coeffs(self):
        # The pade approximation is valid for kr<1
        fit_kr = self.data_kr[self.data_kr <= 1]
        fit_imped = self.data_imped[self.data_kr <= 1]
        delta = 0.6133
        beta = 0.25/(delta**2)
        params_init = [beta, 1/delta]  # initiated with unflanged parameters
        res = least_squares(self.residual, params_init, jac=self.jacobian,
                            method='lm', verbose=0, args=(fit_kr, fit_imped))
        beta = res.x[0]
        alpha_unscaled = res.x[1]
        self.rad_pade = RadiationPade((alpha_unscaled, beta),
                                      label=self.label, scaling=self.scaling,
                                      convention=self.convention)

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        return self.rad_pade.compute_temporal_coefs(radius, rho, c,
                                                    opening_factor)

    def get_impedance_from_pade(self, omegas, radius, rho, celerity,
                                opening_factor):
        """
        Radiation impedance value at a given pulsation.

        Compute the radiation condition (impedance or admittance depending on
        the convention) at the given puslation `omegas` by using the fitted
        Padé development.

        Parameters
        ----------
        omegas : float, np.array of float, list of float
            The pulsations at which are computed the radiation condition (s^-1)
        radius : float
            Radius of the opening.
        rho : float
            Air density at the opening.
        c : float
            Sound celerity at the opening.
        opening_factor : float
            Factor between 0 and 1 which allows to close the opening.

        Returns
        -------
        radiation : np.array
            The radiation impedance at the given pulsations.

        """
        return self.rad_pade.get_impedance(omegas, radius, rho, celerity,
                                           opening_factor)

    # %% Plot
    def plot_impedance(self, kr0s=np.linspace(0, 4, 200), opening_factor=1.0,
                       axes=None, **kwargs):
        kr_plot = kr0s[np.logical_and(kr0s > np.min(self.data_kr),
                                      kr0s < np.max(self.data_kr))]
        PhysicalRadiation.plot_impedance(self, kr_plot, opening_factor, axes,
                                      **kwargs)
        self.rad_pade.plot_impedance(kr0s, opening_factor, axes, **kwargs)
