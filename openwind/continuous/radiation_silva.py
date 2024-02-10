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


class RadiationPade2ndOrder(PhysicalRadiation):
    """
    Radiation model in Padé form at the second order as proposed by [Silva.C]_ .

    Following the correction proposed in the Corrigendum and with the good
    complex convention, the equation is:

    .. math::
        Z_r = \\frac{- (n_1 - d_1) jkr + d_2 (j k r)^2}\
            {2 + (d_1 + n_1) jkr + d_2 (jkr)^2}.

    Parameters
    ----------
    coefs: tuple of 3 floats
        The 3 coefficient values, depending of the radiation type (flanged,
        unflanged) in the order: :math:`n_1, d_1, d_2`
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
        The default is 'PH1'.


    References
    ----------
    .. [Silva.C] F.Silva, P.Guillemain, J.Kergomard, B.Mallaroni, and A.N.
        Norris, *"Approximation formulae for the acoustic radiation impedance
        of a cylindrical pipe"*, Journal of Sound and Vibration, vol.322,
        no.1–2, pp.255–263, Apr.2009, https://doi.org/10.1016/j.jsv.2008.11.008

    """
    COEFS = {'unflanged_2nd_order': (0.167, 1.393, 0.457),
             'flanged_2nd_order': (0.182, 1.825, 0.649)}
    """
    Some radiation conditions and the associated values of coefficients used in
    :py:meth:`radiation_model<openwind.continuous.radiation_model.radiation_model>`.
    """

    def __init__(self, coefs, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)
        if type(coefs) is str:
            self.n1, self.d1, self.d2 = self.COEFS[coefs]
            self._name = coefs
        else:
            self._name = None
            self.n1, self.d1, self.d2 = coefs

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            return 'n1={:.5f}, d1={:.5f}, d2={:.5f}'.format(self.n1, self.d1,
                                                            self.d2)

    def _rad_coeff(self, omegas, radius, rho, celerity, opening_factor):
        kr = omegas*radius/celerity
        Zc = rho*celerity / (np.pi*radius**2)

        num = (self.n1 - self.d1)*-1j*kr - self.d2*kr**2*opening_factor
        den = (2*opening_factor + (self.d1 + self.n1)*1j*kr*opening_factor**2
               - self.d2*kr**2*opening_factor**3)
        return Zc, num, den

    def _diff_rad_coeff(self, dr, omegas, radius, rho, celerity,
                        opening_factor):
        kr = omegas*radius/celerity
        dZc = -2*dr/radius * rho*celerity / (np.pi*radius**2)
        dkr = omegas*dr/celerity
        d_num = -1j*(self.n1 - self.d1)*dkr - 2*self.d2*kr*dkr*opening_factor
        d_den = ((self.d1 + self.n1)*1j*dkr*opening_factor**2
                 - 2*self.d2*kr*dkr*opening_factor**3)
        return dZc, d_num, d_den

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        """
        Not implemented for this type of radiation.

        It can not be used in temporal domain.
        """
        raise NotImplementedError('Currently, the radiation with second order '
                                  'approximation can not be used in temporal '
                                  'domain')

    def get_impedance(self, omegas, radius, rho, celerity, opening_factor):
        Zc, num, den = self._rad_coeff(omegas, radius, rho, celerity,
                                       opening_factor)
        return Zc * num / den

    def get_diff_impedance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        Zc, num, den = self._rad_coeff(omegas, radius, rho, celerity,
                                       opening_factor)
        dZc, d_num, d_den = self._diff_rad_coeff(dr, omegas, radius, rho,
                                                 celerity, opening_factor)
        return dZc*num/den + Zc*(d_num*den - num*d_den)/den**2

    def get_admitance(self, omegas, radius, rho, celerity, opening_factor):
        Zc, num, den = self._rad_coeff(omegas, radius, rho, celerity,
                                       opening_factor)
        return den / (num*Zc)

    def get_diff_admitance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        Zc, num, den = self._rad_coeff(omegas, radius, rho, celerity,
                                       opening_factor)
        dZc, d_num, d_den = self._diff_rad_coeff(dr, omegas, radius, rho,
                                                 celerity, opening_factor)
        return -dZc/Zc**2 * den/num + (d_den*num - den*d_num)/(num**2 * Zc)


class RadiationNoncausal(PhysicalRadiation):
    """
    Non-causal radiation model at the second order as proposed by [Silva.NC]_.

    Following the adapted complex convention, the equations are:

    .. math::
        \\begin{align}
        Z_r &= Z_c  j \\tan(kr L + 0.5j \\log(|R|)) \\\\
        L &= \\eta \\frac{ 1 + b_1 (kr)^2 }{ 1 + b_2 (kr)^2 + b_3 (kr)^4\
                                             + b_4 (kr)^6)}\\\\
        |R| &= \\frac{1 + a_1(kr)^2}{1 + (\\beta+a_1)(kr)^2 \
                                     + a_2(kr)^4 + a_3(kr)^6}
        \\end{align}


    Parameters
    ----------
    coefs : tuple of 9 float
        The 9 coefficient values, depending of the radiation type (flanged,
        unflanged) in the order: :math:`a_1, a_2, a_3, b_1, b_2, b_3, b_4,\
        \\beta, \\eta`
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


    References
    ----------
    .. [Silva.NC] F.Silva, P.Guillemain, J.Kergomard, B.Mallaroni, and A.N.
        Norris, *"Approximation formulae for the acoustic radiation impedance
        of a cylindrical pipe"*, Journal of Sound and Vibration, vol.322,
        no.1–2, pp.255–263, Apr.2009, https://doi.org/10.1016/j.jsv.2008.11.008

    """

    COEFS = {'unflanged_non_causal': (.8, .266, .0263, .0599, .238, -.0153,
                                      .0015, .5, .6133),
             'flanged_non_causal':   (.73, .372, .0231, .244, .723, -.0198,
                                      .00366, 1., .8216)
             }
    """
    Some radiation conditions and the associated values of coefficients used in
    :py:meth:`radiation_model<openwind.continuous.radiation_model.radiation_model>`.
    """

    def __init__(self, coefs, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)
        if type(coefs) is str:
            self.a1, self.a2, self.a3 = self.COEFS[coefs][0:3]
            self.b1, self.b2, self.b3, self.b4 = self.COEFS[coefs][3:7]
            self.beta, self.eta = self.COEFS[coefs][7:]
            self._name = coefs
        else:
            self._name = None
            self.a1, self.a2, self.a3 = coefs[0:3]
            self.b1, self.b2, self.b3, self.b4 = coefs[3:7]
            self.beta, self.eta = coefs[7:]

    @property
    def name(self):
        if self._name:
            return self._name
        else:
            msg = 'a1={:.5f}, a2={:.5f}, a3={:.5f}'.format(self.a1, self.a2,
                                                           self.a3)
            msg += 'b1={:.5f}, b2={:.5f}, b3={:.5f}, b4={:.5f}'.format(self.b1,
                                                                       self.b2,
                                                                       self.b3,
                                                                       self.b4)
            msg += 'beta={:.5f}, eta={:.5f}'.format(self.beta, self.eta)
            return msg

    def _rad_coeff(self, omegas, radius, rho, celerity, opening_factor):
        kr = omegas*radius/celerity
        Zc_real = rho*celerity / (np.pi*radius**2)

        Zc = Zc_real
        kr_open = kr*opening_factor

        numR = 1 + self.a1*kr_open**2
        denR = (1 + (self.beta+self.a1)*kr_open**2 + self.a2*kr_open**4
                + self.a3*kr_open**6)
        Rmod = numR / denR
        numL = 1 + self.b1*kr_open**2
        denL = 1 + self.b2*kr_open**2 + self.b3*kr_open**4 + self.b4*kr_open**6
        L = self.eta * numL / denL
        return kr, Zc, Rmod, L

    def _diff_rad_coeff(self, dr, omegas, radius, rho, celerity,
                        opening_factor):
        kr_open = omegas*radius/celerity*opening_factor
        dZc = -2*dr/radius * rho*celerity / (np.pi*radius**2)
        dkr = omegas*dr/celerity*opening_factor

        numR = 1 + self.a1*kr_open**2
        dnumR = 2*self.a1*kr_open*dkr
        denR = (1 + (self.beta+self.a1)*kr_open**2 + self.a2*kr_open**4
                + self.a3*kr_open**6)
        d_denR = dkr*(2*(self.beta+self.a1)*kr_open + 4*self.a2*kr_open**3
                      + 6*self.a3*kr_open**5)
        dRmod = (dnumR*denR - numR*d_denR) / denR**2

        numL = 1 + self.b1*kr_open**2
        dnumL = 2*self.b1*kr_open*dkr
        denL = 1 + self.b2*kr_open**2 + self.b3*kr_open**4 + self.b4*kr_open**6
        d_denL = dkr*(2*self.b2*kr_open + 4*self.b3*kr_open**3
                      + 6*self.b4*kr_open**5)
        dL = self.eta * (dnumL*denL - numL*d_denL) / denL**2

        return dZc, dRmod, dL

    def compute_temporal_coefs(self, radius, rho, c, opening_factor):
        """
        Not implemented for this type of radiation.

        It can not be used in temporal domain.
        """
        raise NotImplementedError('Non-causal radiation can not be used in '
                                  'temporal domain.')

    def get_impedance(self, omegas, radius, rho, celerity, opening_factor):
        kr, Zc, Rmod, L = self._rad_coeff(omegas, radius, rho, celerity,
                                          opening_factor)
        return Zc/opening_factor * 1j*np.tan(kr*L + .5j*np.log(Rmod))

    def get_diff_impedance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        kr, Zc, Rmod, L = self._rad_coeff(omegas, radius, rho, celerity,
                                          opening_factor)
        dZc, dRmod, dL = self._diff_rad_coeff(dr, omegas, radius, rho,
                                              celerity,  opening_factor)
        dkr = omegas*dr/celerity

        dtan = ((dkr*L + kr*dL + .5j*dRmod/Rmod)
                / np.cos(kr*L + .5j*np.log(Rmod))**2)
        return (dZc*1j*np.tan(kr*L + .5j*np.log(Rmod))
                + Zc*1j*dtan) / opening_factor

    def get_admitance(self, omegas, radius, rho, celerity, opening_factor):
        kr, Zc, Rmod, L = self._rad_coeff(omegas, radius, rho, celerity,
                                          opening_factor)
        return opening_factor / (Zc * 1j*np.tan(kr*L + .5j*np.log(Rmod)))

    def get_diff_admitance(self, dr, omegas, radius, rho, celerity,
                           opening_factor):
        kr, Zc, Rmod, L = self._rad_coeff(omegas, radius, rho, celerity,
                                          opening_factor)
        dZc, dRmod, dL = self._diff_rad_coeff(dr, omegas, radius, rho,
                                              celerity,  opening_factor)
        dkr = omegas*dr/celerity

        dtan = ((dkr*L + kr*dL + .5j*dRmod/Rmod)
                / np.cos(kr*L + .5j*np.log(Rmod))**2)
        dZ = dZc*1j*np.tan(kr*L + .5j*np.log(Rmod)) + Zc*1j*dtan
        return -opening_factor*dZ / (Zc*1j*np.tan(kr*L + .5j*np.log(Rmod)))**2
