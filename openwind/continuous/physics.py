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
import warnings

class Physics:
    """
    Compute all the physical coefficient with respect to the temperature and the humidity rate.

    The temperature and humidity can be constant or a function of the location.
    All the physical coefficient are supposed variable with respect to the location.

    All the formula used come from the research report [report_humidity]_ or the
    Chaigne and Kergomard book [Chaigne_cst]_ (Chap. 5, p.241).

    .. warning::
        In order to allow the variation of the different quantities with the position
        along the main bore of the instrument (e.g. due to a temperature gradient),
        all the quantities are associated to callable needing the position "x" as
        input argument.

    Parameters
    ----------
    temp : float or callable
        temperature in Celcius degree (°C). If a float is given, the temperature
        is uniform in the instrument. A callable must return an array for an
        array a return the temperature at the given temperature.
    humidity: float, optional
        The humidity rate (between 0 and 1). The default is 0.5.
    carbon: float, optional
        The carbon rate (between 0 and 1). Default: 4.2e-4 (420ppm)
    ref_phy_coef: string, optional
        Specify the reference used to compute the different quantities. See
        :py:attr:`.Physics.AVAILABLE_REFERENCE`. Default is 'RR', using
        expressions from [report_humidity]_.

    References
    ----------
    .. [report_humidity] A. Ernoult 2023. "Effect of air humidity and carbon \
        dioxide in the sound propagation for the modeling of wind musical \
        instruments". Research Report #RR-9500. Inria. https://hal.inria.fr/hal-04008847.


    .. [Chaigne_cst] Chaigne, Antoine, and Jean Kergomard. 2016. "Acoustics of \
        Musical Instruments". Modern Acoustics and Signal Processing. New York:\
        Springer. https://doi.org/10.1007/978-1-4939-3679-3.


    Attributes
    ----------
    T : callable
        temperature in Kelvin
    rho : callable
        air density in kg/m3
    c : callable
        air sound speed in m/s
    mu : callable
        viscosity in kg/(m.s)
    Cp : callable
        specific heat with constant pressure in J/(kg K)
    kappa : callable
        thermal conductivity in J/(m s K)
    gamma : callable
        ratio of specific heats (dimensionless)
    khi : callable
        isentropic compressibility in kg/(m.s)
    lt : callable
        characteristic width of thermal effect in m
    lv : callable
         characteristic width of viscous effect in m
    c_lt : callable
        velocity times the characteristic distance lt for loss computation
    c_lv : callable
        velocity times the characteristic distance lv for loss computation

    """

    AVAILABLE_REFERENCE=['Chaigne_Kergomard', 'RR', 'RR_Zuckerwar', 'RR_Tsilingiris']

    def __init__(self, temp, humidity=.5, carbon=4e-4, ref_phy_coef='RR'):

        if ref_phy_coef not in self.AVAILABLE_REFERENCE:
            raise ValueError(f'Reference "{ref_phy_coef}" is not repertoried. Please chose between: {self.AVAILABLE_REFERENCE}')

        T0 = 273.15
        T20 = T0 + 20
        self.T0 = T0

        self.temp, unif_temp = self._convert_to_callable(temp)
        self.humidity, unif_humid = self._convert_to_callable(humidity)
        self.carbon, unif_carbon = self._convert_to_callable(carbon)
        self._uniform = all([unif_temp, unif_humid, unif_carbon])

        if self.humidity(0)>1 or self.humidity(0)<0:
            raise ValueError(f'The humidity rate must be between 0 and 1 (here:{humidity})')
        if self.carbon(0)>1 or self.carbon(0)<0:
            raise ValueError(f'The carbon rate must be between 0 and 1 (here:{carbon})')
        if self.temp(0)>100:
            warnings.warn(f'The temperature must be given in °C. The current value: t={temp}°C')


        self.T = lambda x: self.temp(x) + T0
        self.h = lambda x: self.humidity(x)*10**(5.21899 -5.8294*T20/self.T(x) -1.0252*(T20/self.T(x))**2) # molar frac, Tay. exp. combined Eq.(16.11) and Eq.(16.13.a), Handbook Zuck., Chap.16.3.3

        if ref_phy_coef=='Chaigne_Kergomard': # Values from Chaigne and Kergo
           if np.any(self.humidity(0)>0) or np.any(self.carbon(0)>0) or not unif_humid or not unif_carbon:
                     warnings.warn('With Chaigne and Kergomard values, the humidity and carbon dioxide rates are ignored.')
           self.Cp, self.gamma, self.rho, self.c, self.mu, self.kappa = self.Chaigne_Kergomard_expressions()

        elif ref_phy_coef.startswith('RR'): # expressions from Report
            self.Cp, self.gamma, self.rho, self.c = self.Research_Report_expressions()
            if ref_phy_coef=='RR_Tsilingiris':
                self.mu, self.kappa = self.thermo_viscous_Tsilingiris()
            elif ref_phy_coef=='RR_Zuckerwar':
                self.mu, self.kappa = self.thermo_viscous_Zuckerwar()
            else:
                self.mu, self.kappa = self.thermo_viscous_dry()

        # common from Chaigne and Kergomard
        self.khi   = lambda x: 1 / (self.rho(x) * self.c(x) ** 2)
        self.c_lt  = lambda x: self.kappa(x) / (self.rho(x) * self.Cp(x))
        self.c_lv  = lambda x: self.mu(x) / self.rho(x)
        self.lt    = lambda x: self.c_lt(x) / self.c(x)
        self.lv    = lambda x: self.c_lv(x) / self.c(x)

    @staticmethod
    def _convert_to_callable(y):
        """
        Convert float to callable relative to position

        Parameters
        ----------
        y : float or callable
            The value to convert

        Returns
        -------
        y_cal : callable
            The callable returning the value.
        unif : boolean
            True if y was a float false either.

        """
        if callable(y):
            y_cal = y
            unif = False
        else:
            y_cal = lambda x: np.full_like(x, y, dtype=float)
            unif = True
        return y_cal, unif

    def Chaigne_Kergomard_expressions(self):
        """
        Expressions from [Chaigne_cst]_ (Chap. 5, p.241).

        Returns
        -------
        Cp : callable
            The specific heat at constant pressure in J/(K kg).
        gamma : callable
            The heat capacity ratio.
        rho : callable
            The densitiy in kg/m3.
        c : callable
            The speed of sound in m/s.
        mu : callable
            The viscosity [kg/(m s)].
        kappa : callable
            The thermal conductivity [J/(m s K)].

        """
        Cp    = lambda x: 240  * 4.184 * np.ones_like(x) # converted Cal/(g.°C) in SI units: J/(kg.K)
        gamma = lambda x: 1.402 * np.ones_like(x)
        rho   = lambda x: 1.2929 * self.T0 / self.T(x)
        c     = lambda x: 331.45 * np.sqrt(self.T(x) / self.T0)
        mu    = lambda x: 1.708e-5 * (1 + 0.0029 * self.temp(x))
        kappa = lambda x: 5.77 * 1e-3 * (1 + 0.0033 * self.temp(x)) * 4.184 # converted Cal/(cm.s.°C) in SI: J/(m.s.K)
        return Cp, gamma, rho, c, mu, kappa

    def Research_Report_expressions(self):
        """
        Expressions from [report_humidity]_

        Returns
        -------
        Cp : callable
            The specific heat at constant pressure in J/(K kg).
        gamma : callable
            The heat capacity ratio.
        rho : callable
            The densitiy in kg/m3.
        c : callable
            The speed of sound in m/s.

        """
        CO2_ref = 4.2e-4 # reference molar fraction of CO2: 420ppm (ambient rate in 2022)
        h_ref = 1.1571e-2 # ref humidity rate: value obtained at 20°C for 50% of humidity
        T20 = self.T0 + 20 # ref temp: 20°C

        dh = lambda x: self.h(x) - h_ref
        dCO2 = lambda x: self.carbon(x) - CO2_ref
        dT = lambda x: self.T(x)/T20 - 1

        Cp = lambda x: 1012.25*(1 + 0.5438*dh(x) + 0.638 *dh(x)**2
                                - 0.1594*dCO2(x) + 0.075*dCO2(x)**2
                                + 9.52e-3*dT(x) + 4.06e-2*dT(x)**2
                                + 0.3976*dCO2(x)*dT(x) )

        gamma = lambda x: 1.40108*(1 - 0.060*dh(x) - 0.104*dCO2(x) - 0.0087*dT(x)
                                   - 0.154*dCO2(x)*dT(x))
        # First order approx
        # Cp = lambda x: 1012.25*(1 + 0.5438*dh(x) - 0.1594*dCO2(x) + 9.52e-3*dT(x))
        # gamma = lambda x: 1.40108*(1 - 0.060*dh(x) - 0.104*dCO2(x) - 0.0087*dT(x))

        rho = lambda x: 1.19930*T20/self.T(x) * (1 - 0.3767*dh(x) + 0.4162*dCO2(x)
                                                 - 0.00291*dT(x) )
        c = lambda x: 343.986*np.sqrt(self.T(x)/T20*(1 + 0.314*dh(x) - 0.520*dCO2(x)
                                                     + 0.25*dCO2(x)**2 - 0.16*dCO2(x)*dT(x))
                                      )
        return Cp, gamma, rho, c

    def thermo_viscous_dry(self):
        T20 = self.T0 + 20
        mu_dry = lambda x: 1.8206e-05*( 1 + 0.77013*(self.T(x)/T20-1))
        kappa_dry = lambda x: 2.5562e-02*(1 + 0.8490*(self.T(x)/T20 - 1) )
        return mu_dry, kappa_dry

    def thermo_viscous_Zuckerwar(self):
        """
        Viscosity and Thermal conductivity of humid air from [Zuckerwar]_

        Eq.(11) and eq.(12) linearized around the point T20, h_ref

        References
        ----------
        .. [Zuckerwar] A.J. Zuckerwar and R.W. Meredith 1985. "Low‐frequency \
            absorption of sound in air". The Journal of the Acoustical Society \
            of America. 78(3), p.946–955. https://asa.scitation.org/doi/abs/10.1121/1.392927.

        Returns
        -------
        mu : callable
            The viscosity [kg/(m s)].
        kappa : callable
            The thermal conductivity [J/(m s K)].

        """
        T20 = self.T0 + 20
        h_ref = 1.1571e-2
        mu = lambda x: 1.813e-05*(1 + 0.77*(self.T(x)/T20 -1) -0.10*(self.h(x) - h_ref))
        kappa = lambda x: 2.5181e-2*(1 + 0.900*(self.T(x)/T20 -1) + 0.0664*(self.h(x) - h_ref) )
        return mu, kappa

    def thermo_viscous_Tsilingiris(self):
        """
        Viscosity and Thermal conductivity of humid air from [Tsilingiris]_

        References
        ----------
        .. [Tsilingiris] P.T. Tsilingiris 2008. "Thermophysical and transport \
            properties of humid air at temperature range between 0 and 100°C". \
            Energy Conversion and Management. 49(5), p.1098–1110. \
            https://www.sciencedirect.com/science/article/pii/S0196890407003329.

        Returns
        -------
        mu : callable
            The viscosity [kg/(m s)].
        kappa : callable
            The thermal conductivity [J/(m s K)].

        """
        T20 = self.T0 + 20
        Mdry = 28.9647e-3 # HANDBOOK chap.16.4.2, p.14 // masse molaire air sec https://fr.wikipedia.org/wiki/Constante_universelle_des_gaz_parfaits
        Mv =  18.01534e-3 #https://fr.wikipedia.org/wiki/Eau
        M_av =  Mdry/Mv

        mu_dry, kappa_dry = self.thermo_viscous_dry()
        mu_v = lambda x: 8.8582e-6*(1 + 1.3239*(self.T(x)/T20 - 1))
        kappa_v = lambda x:  1.8778e-2*(1 +  0.972*(self.T(x)/T20 -1))

        mu_av  = lambda x: mu_dry(x) / mu_v(x)
        phi_av = lambda x: np.sqrt(2)/4* 1/np.sqrt(1 + M_av) * (1 + np.sqrt(mu_av(x))/M_av**.25 )**2 # Tsilingiris Eq.(22)
        phi_va = lambda x: np.sqrt(2)/4* 1/np.sqrt(1 + 1/M_av) * (1 + M_av**.25 / np.sqrt(mu_av(x)))**2 # Tsilingiris Eq.(23)

        mu = lambda x: ( (1-self.h(x)) * mu_dry(x) / ((1-self.h(x)) +  self.h(x)*phi_av(x))
                        + self.h(x)*mu_v(x) / (self.h(x) +  (1-self.h(x)) * phi_va(x))
                        ) # Tsilingiris Eq.(21)
        kappa = lambda x: ( (1-self.h(x)) * kappa_dry(x) / ((1-self.h(x)) +  self.h(x)*phi_av(x))
                           + self.h(x)*kappa_v(x) / (self.h(x) +  (1-self.h(x))*phi_va(x))
                           ) # Tsilingiris Eq.(28)
        return mu, kappa

    def get_coefs(self, x, *names):
        """Get the values of several coefficients at the same time.

        Parameters
        ----------
        x : float or array-like
            where to evaluate the coefficients.
        *names : string...
            the names of the coefficients to take

        Returns
        -------
        values : tuple of (float or array-like)
        """
        coefs = tuple(getattr(self, name) for name in names)
        return tuple(coef(x) if callable(coef)
                     else coef
                     for coef in coefs)

    @property
    def uniform(self):
        """Are the coefficients independent of space?

        False if the physical coefficients depend on x, True otherwise.

        Returns
        -------
        bool

        """
        return self._uniform
