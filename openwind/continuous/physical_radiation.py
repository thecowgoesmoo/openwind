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
from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

from openwind.continuous import NetlistConnector, Physics


class PhysicalRadiation(NetlistConnector, ABC):
    """
    Netlist connector corresponding to a radiating end.

    This is the mother class of all the radiation conditions. It imposes a
    relation between the acoustic pressure and flow at a pipe-end thanks to
    a radiation impedance (or admittance).

    Parameters
    ----------
    label : string
        The label of the component
    scaling : :py:class: `Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : str, optional
        Must be one out of {'PH1', 'VH1'} \
        The convention used in this component. The default is 'PH1'.

    """

    @abstractmethod
    def get_impedance(self, omegas, radius, rho, c, opening_factor):
        """
        The value of the radiation impedance at the given frequency.

        The impedance is not scaled. It has the dimension of :math:`\\rho c /S`

        Parameters
        ----------
        omegas : array(float)
            pulsation in s^-1.
        radius : float
            radius of the circular opening.
        rho : float
            air density.
        c : float
            speed of sound in air.
        opening_factor : float
            1 for open pipe, 0 for closed pipe.

        Returns
        -------
        array(float)
        """

    @abstractmethod
    def get_admitance(self, omegas, radius, rho, c, opening_factor):
        """
        The value of the radiation admitance at the given frequency.

        The admitance is not scaled. It has the dimension of :math:`S /\\rho c`

        Parameters
        ----------
        omegas : array(float)
            pulsation in s^-1.
        radius : float
            radius of the circular opening.
        rho : float
            air density.
        c : float
            speed of sound in air.
        opening_factor : float
            1 for open pipe, 0 for closed pipe.

        Returns
        -------
        array(float)
        """

    @abstractmethod
    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        """
        Compute the coefficients for the temporal formulation.

        To be passed in the temporal domain, the radiation impedance must be
        written:

        .. math::
            \\frac{\\rho c}{\\pi R^2} \\frac{j \\omega}{\\alpha \\frac{c}{r} +\
                                                        \\beta j \\omega}

        This method returns the three coefficients :math:`\\frac{\\rho c}\
        {\\pi R^2}, \\alpha \\frac{c}{r}` (which depend on the radius) and
        :math:`\\beta`.

        Parameters
        ----------
        radius : float
            radius of the circular opening.
        rho : float
            air density.
        c : float
            speed of sound in air.
        opening_factor : float
            1 for open pipe, 0 for closed pipe.

        Returns
        -------
        alpha : float
            Coefficient corresponding to :math:`\\alpha \\frac{c}{r}`.
        beta : float
            Coefficient corresponding to :math:`\\beta`.
        Zplus : float
            Coefficient corresponding to :math:`\\frac{\\rho c}{\\pi R^2}`.

        """

    @property
    @abstractmethod
    def name(self):
        """
        Name of this model of radiation used for plotting, and for repr().
        """

    @abstractmethod
    def get_diff_impedance(self, dr, omegas, radius, rho, c, opening_factor):
        """
        Variation of impedance with respect to radius at the given frequencies.

        Parameters
        ----------
        dr : float
            Radius derivative w.r. to the design parameter.
        omegas : array(float)
            pulsation in s^-1.
        radius : float
            radius of the circular opening.
        rho : float
            air density.
        c : float
            speed of sound in air.
        opening_factor : float
            1 for open pipe, 0 for closed pipe.

        Returns
        -------
        array(float)
        """

    @abstractmethod
    def get_diff_admitance(self, dr, omegas, radius, rho, c, opening_factor):
        """
        Variation of admitance with respect to radius.

        Parameters
        ----------
        dr : float
            Radius derivative w.r. to the design parameter.
        omegas : array(float)
            pulsation in s^-1.
        radius : float
            radius of the circular opening.
        rho : float
            air density.
        c : float
            speed of sound in air.
        opening_factor : float
            1 for open pipe, 0 for closed pipe.

        Returns
        -------
        array(float)
        """

    def plot_impedance(self, kr0s=np.linspace(0.1, 2, 200), opening_factor=1.0,
                       axes=None, **kwargs):
        """Plot the radiation impedance of the pipe.

        Parameters
        ----------
        kr0s : array(float), optional
            Values of omega/c * r0 to use. Default `np.linspace(0.1, 2, 200)`.
        opening_factor : float, optional
            1 for open pipe, 0 for closed pipe. Default 1.
        axes : matplotlib Axes, optional
            Where to plot. Default creates a new figure.
        **kwargs :
            Keyword arguments for plt.plot().
        """
        r0 = 1.0
        rho, c = Physics(20).get_coefs(0, 'rho', 'c')
        omegas = kr0s * c/r0
        imped = self.get_impedance(omegas, r0, rho, c, opening_factor)
        imped /= rho*c/(np.pi*r0**2)
        if not axes:
            axes = plt.gca()
        line = axes.plot(kr0s, np.real(imped),
                         label='Real({})'.format(self), **kwargs)[0]
        axes.plot(kr0s, np.imag(imped), '--', color=line.get_color(),
                  label='Imag(' + str(self) + ')', **kwargs)
        axes.set_xlabel('$k r_0$')
        axes.set_ylabel('$Z_R/Z_C$')
        axes.legend()
        axes.grid()

    def __repr__(self):
        return "<{class_}({name})>".format(class_=type(self).__name__,
                                           name=repr(self.name))

    def get_radiation_at(self, omegas_scaled, radius, rho, c, opening_factor):
        """
        Radiation coefficient, to be put in the matrix.

        Following the convention it corresponds to the admittance or the
        impedance normalized by the convenient value from :py:class:`Scaling\
        <openwind.continuous.scaling.Scaling>`

        Parameters
        ----------
        omegas_scaled : float or array(float)
            Normalized angular frequencies.
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
            The radiation coefficients

        """
        omegas = omegas_scaled / self.scaling.get_time()
        if self.convention == 'PH1':
            Yr = self.get_admitance(omegas, radius, rho, c, opening_factor)
            return Yr * self.scaling.get_impedance()
        elif self.convention == 'VH1':
            Zr = self.get_impedance(omegas, radius, rho, c, opening_factor)
            return Zr / self.scaling.get_impedance()
        assert False  # Should not be reached

    def get_diff_radiation_at(self, dr, omegas_scaled, radius,
                              rho, c, opening_factor):
        """
        Variation of the radiation coefficient wr to the radius to be put in matrix

        Following the convention it corresponds to the variation of the
        admittance or of the impedance normalized by the convenient value from
        :py:class:`Scaling<openwind.continuous.scaling.Scaling>`

        Parameters
        ----------
        dr : float
            Radius derivative w.r. to the design parameter.
        omegas_scaled : float or array(float)
            Normalized angular frequencies.
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
        np.array

        """
        omegas = omegas_scaled / self.scaling.get_time()
        if self.convention == 'PH1':
            d_Yr = self.get_diff_admitance(dr, omegas, radius, rho, c,
                                           opening_factor)
            return d_Yr * self.scaling.get_impedance()
        elif self.convention == 'VH1':
            d_Zr = self.get_diff_impedance(dr, omegas, radius, rho, c,
                                           opening_factor)
            return d_Zr / self.scaling.get_impedance()
        assert False  # Should not be reached

    def is_compatible_for_modal(self):
        # default value : not compatible with modal computation
        return False

def taylor_to_pade(delta, beta_chaigne):
    """
    Convert coef of Taylor serie to coef of Pade development.

    The Taylor serie of the radiation impedance is [Chaigne_Rad]_ (Chap.12.6.1):

    .. math::
        Z_r = \\frac{\\rho c}{\\pi R^2} \\left(\\delta j kR + \\beta_c (k R)^2\
                                               \\right)

    with :math:`R` the opening radius and :math:`k=\\omega/c` the wavenumber.
    The Padé development of the radiation impedance gives [Rabiner]_:

    .. math::
        Z_r = \\frac{\\rho c}{\\pi R^2} \\frac{jkR}{\\alpha + jkR \\beta}

    By identification it gives:

    .. math::
        \\begin{align}
        \\alpha &= \\frac{1}{\\delta} \\\\
        \\beta &= \\frac{\\beta_c}{\\delta^2}
        \\end{align}

    Parameters
    ----------
    delta, beta_chaigne: float
        The coefficients of the Taylor serie.

    Returns
    -------
    alpha, beta : float
        The coefficients of the Padé serie.

    References
    ----------
    .. [Chaigne_Rad] Chaigne, A., Kergomard, J., 2016. *Acoustics of Musical\
        Instruments*, Modern Acoustics and Signal Processing. Springer, New York.\
        https://doi.org/10.1007/978-1-4939-3679-3

    .. [Rabiner] Rabiner, L., Schafer, R., 1978. *Digital Processing of speech\
        signals.*
    """
    alpha = 1 / delta
    beta = beta_chaigne / delta**2
    return alpha, beta


class RadiationPade(PhysicalRadiation):
    """Radiation model in Padé form.

    The equation is:

    .. math::
        Z_r = Z_c \\frac{j kr}{\\alpha + \\beta j kr} = \
            Z_c \\frac{j \\omega}{\\alpha \\frac{c}{r} + \\beta j \\omega}

    Parameters
    ----------
    coefs : tuple of 2 floats
        The 2 coefficient values, depending of the radiation type in the
        order: :math:`\\alpha, \\beta`
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

    """

    COEFS = {"planar_piston": taylor_to_pade(8/(3*np.pi), 0.5),
             "unflanged": taylor_to_pade(0.6133, 0.25),
             "infinite_flanged": taylor_to_pade(0.8236, 0.5),
             "total_transmission": (0, 1),
             "closed": (0, 0)}
    """
    Some radiation conditions and the associated values of coefficients used in
    :py:meth:`radiation_model<openwind.continuous.radiation_model.radiation_model>`.
    """

    def __init__(self, coefs, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)
        if type(coefs) is str:
            self.alpha_unscaled, self.beta = self.COEFS[coefs]
            self._name = coefs
        else:
            self._name = None
            self.alpha_unscaled, self.beta = coefs

    def is_compatible_for_modal(self):
        return True
    
    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return 'alpha={:.3g}, beta={:.3g}'.format(self.alpha_unscaled,
                                                      self.beta)

    def __str__(self):
        return ("Radiation in Padé form: (alpha={:.3g}, "
                "beta={:.3g})".format(self.alpha_unscaled,
                                      self.beta))
    
    def _get_caract_impedance(self,radius, rho, celerity):
        Zc = rho*celerity / (np.pi*radius**2) / self.scaling.get_impedance()
        return Zc

    def _rad_coeff(self, omegas, radius, rho, celerity, opening_factor):
        kr = omegas*radius/celerity
        Zc = rho*celerity / (np.pi*radius**2)
        alpha = self.alpha_unscaled * opening_factor
        beta = self.beta * opening_factor**2
        return kr, Zc, alpha, beta

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        scaling = self.scaling
        Zplus = (rho * c / (np.pi * radius**2))/scaling.get_impedance()
        alpha = (self.alpha_unscaled * scaling.get_time() * c
                 / radius * opening_factor)
        beta = self.beta * opening_factor**2
        return alpha, beta, Zplus

    def get_impedance(self, omegas, radius, rho, celerity, opening_factor):
        kr, Zc, alpha, beta = self._rad_coeff(omegas, radius, rho, celerity,
                                              opening_factor)
        return Zc * 1j*kr / (alpha + 1j*kr*beta)

    def get_diff_impedance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        kr, Zc, alpha, beta = self._rad_coeff(omegas, radius, rho, celerity,
                                              opening_factor)
        dZc = -2*dr/radius * Zc
        dkr = omegas*dr/celerity
        return (dZc * 1j*kr / (alpha + 1j*kr*beta)
                + Zc * 1j*dkr*alpha / (alpha + 1j*kr*beta)**2)

    def get_admitance(self, omegas, radius, rho, celerity, opening_factor):
        kr, Zc, alpha, beta = self._rad_coeff(omegas, radius, rho, celerity,
                                              opening_factor)
        return (alpha + 1j*kr*beta) / (Zc * 1j*kr)

    def get_diff_admitance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        kr, Zc, alpha, beta = self._rad_coeff(omegas, radius, rho, celerity,
                                              opening_factor)
        dZc = -2*dr/radius * Zc
        dkr = omegas*dr/celerity
        return (dZc*1j*kr*(alpha + beta*1j*kr) + Zc*1j*dkr*alpha) / (Zc*kr)**2


class RadiationPerfectlyOpen(PhysicalRadiation):
    """
    Perfectly open radiation impedance (pressure=0)

    It correspond to :math:`Z_r=0`

    .. warning::
        Due to the 'PH1' convention, this radiation condition is not treated
        as the other ones in the frequential and temporal domains.

    See Also
    --------
    :py:class:`TemporalPressureCondition\
    <openwind.temporal.tpressure_condition.TemporalPressureCondition>`
        The temporal version of this radiation condition
    :py:class:`FrequentialPressureCondition\
    <openwind.frequential.frequential_pressure_condition.FrequentialPressureCondition>`
        The temporal version of this radiation condition

    Parameters
    ----------
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
    """

    @property
    def name(self):
        return "perfectly_open"

    def get_impedance(self, *args, **kwargs):
        return 0

    def get_diff_impedance(self, *args, **kwargs):
        return 0

    def get_admitance(self, *args, **kwargs):
        return np.infty

    def get_diff_admitance(self, *args, **kwargs):
        return 0

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        """
        Not implemented for this type of radiation.

        In temporal domain a different convention is used to imposed the
        pressure.

        See Also
        --------
        :py:class:`TemporalPressureCondition\
            <openwind.temporal.tpressure_condition.TemporalPressureCondition>`
        """
        raise NotImplementedError('Currently, the perfectly open radiation can'
                                  ' not be used in temporal domain')
