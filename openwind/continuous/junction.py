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

from numpy import pi, sqrt, sign
import numpy as np
from openwind.continuous import NetlistConnector
import warnings

class PhysicalJunction(NetlistConnector):
    """
    This is just an abstraction to understand how PhysicalJunctions are
    nothing more than a
    :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`,
    which is important to understand the graph representation of the instrument
    """
    pass

class JunctionSimple(PhysicalJunction):
    """
    Models a simple junction between two pipes without masses.

    This junction assure the continuity of pression and flow between the two
    pipes.

    Parameters
    ----------
    label : string
        The label of the junction
    scaling : :py:class:`Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : {'PH1', 'VH1'}, optional
        The convention used in this junction. The default is 'PH1'.
    """

    def __init__(self, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)

    def is_compatible_for_modal(self):
        return True

class JunctionDiscontinuity(PhysicalJunction):
    """
    Models the junction between 2 pipes with a discontinuity of section.

    An acoustic mass is added due to the discontinuity of section.

    The mass takes into account the possible discontinuity of cross section
    between the two pipes.
    The formulation used is the one propose by [Taillard]_ (Appendix A.3.7). This
    formulation unifiates the two formulations proposed by [Kergomard]_ by assuring
    that the mass and its derivative are null when the two raddi are equal.

    .. math::
        \\begin{align}
        & \\alpha = r_{min}/r_{max} \\\\
        & m_{disc} = (0.09616\\alpha^6 - 0.12386\\alpha^5 + 0.03816\\alpha^4\
            + 0.0809\\alpha^3 - 0.353\\alpha + 0.26164)
        \\end{align}

    The equations at the junction are the flow conservation :math:`u_1 + u_2=0`
    and :math:`p_2 = p_1 + m_{disc} \\partial_t u_1`. In openwind, it is
    rewritten as a linear system by introducing an additional variable
    :math:`\\gamma`:

    .. math::
        \\begin{cases}
        \\left( \\begin{smallmatrix} u_1 \\\\ u_2 \\end{smallmatrix}\\right) = T^{\\star} \\gamma \\\\
        M \\partial_t \\gamma + T \\left( \\begin{smallmatrix} p_1 \\\\ p_2 \\end{smallmatrix}\\right) = 0
        \\end{cases}

    with :math:`T=(-1, 1)` and :math:`M=(m_{disc})`.


    Parameters
    ----------
    label : string
        The label of the junction
    scaling : :py:class:`Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : {'PH1', 'VH1'}, optional
        The convention used in this junction. The default is 'PH1'.

    References
    ----------
    .. [Taillard] P.A.Taillard. 2018. “Theoretical and Experimental Study of the \
        Role of the Reed in Clarinet Playing.” Phd Thesis, Le Mans University. \
        https://tel.archives-ouvertes.fr/tel-01858244/document.

    .. [Kergomard] J.Kergomard, and A. Garcia. 1987. "Simple discontinuities in \
        acoustic waveguides at low frequencies: Critical analysis and formulae". \
        JSV 114 (3): 465‑79. https://doi.org/10.1016/S0022-460X(87)80017-2.
    """

    def __init__(self, label, scaling, convention='PH1'):
        super().__init__(label, scaling, convention)

    def is_compatible_for_modal(self):
        return True

    def get_scaling_masses(self):
        """
        Get the scaling coefficient for acoustic mass.

        Returns
        -------
        float
            The scaling coefficient

        """
        return self.scaling.get_impedance() * self.scaling.get_time()

    def compute_mass(self, r1, r2, rho):
        """
        Compute mass of a Junction from physical parameters.

        Parameters
        ----------
        r1 : float
            Radius of one pipe, at the point of junction
        r2 : float
            Radius of the other pipe, at the point of junction
        rho : float
            Air density, at the point of the junction

        Returns
        -------
        float
            The scaled acoustic mass associated to the junction.
            The mass is zero for identical radii.

        """
        coef_scaling_masses = self.get_scaling_masses()

        rmin = min(r1, r2)
        rmax = max(r1, r2)
        alpha = rmin/rmax
        mass = (.09616*alpha**6 - .12386*alpha**5 + 0.03816*alpha**4
            + 0.0809*alpha**3 - .353*alpha + .26164)
        result = rho/rmin*mass/coef_scaling_masses
        return result

    def get_diff_mass(self, r1, r2, rho, dr1, dr2):
        """
        Derivate the mass w.r. to the radii.

        Parameters
        ----------
        r1 : float
            Radius of one pipe, at the point of junction
        r2 : float
            Radius of the other pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        dr1 : float
            The derivative of the first pipe radius.
        dr2 : float
            The derivative of the second pipe radius.

        Returns
        -------
        float
            The derivative of the mass.

        """
        coef_scaling_masses = self.get_scaling_masses()


        if r1>r2:
            rmin, rmax, drmin, drmax = (r2, r1, dr2, dr1)
        else:
            rmin, rmax, drmin, drmax = (r1, r2, dr1, dr2)

        alpha = rmin/rmax
        dalpha = drmin/rmax - drmax*rmin/rmax**2
        dmass = (6*.09616*alpha**5 - 5*.12386*alpha**4 + 4*0.03816*alpha**3
                + 3*0.0809*alpha**2 - .353)*dalpha

        mass = self.compute_mass(r1, r2, rho)
        return rho/rmin*dmass/coef_scaling_masses - drmin/rmin*mass

class JunctionTjoint(PhysicalJunction):
    """
    Models a T-joint junction where a side tube is branched on a main tube.

    It is done with three acoustic masses :math:`m_{11}, m_{12}, m_{22}`,
    computed them through empirical formulas given in [Chaigne]_ or [Dubos]_, by
    using the short chimney height approximation.

    We have here :math:`m_{11}=m_{22} = m_s + m_a/4`and
    :math:`m12 = m_s - m_a/4` with

    .. math::
        \\begin{align}
        m_s & = \\frac{\\rho}{\\pi r_{side}} \\left(0.82 - 0.193 \\delta - 1.09 \
            \\delta^2 + 1.27 \\delta^3 - 0.71 \\delta^4 \\right) \\\\
        m_a & = \\dfrac{\\rho r_{side}}{\\pi r_{main}^2}  \\left(-0.37 + 0.087\
            \\delta \\right)\\delta^2
        \\end{align}

    with :math:`\\delta=r_{side}/r_{main}` the ratio of the side tube radius
    over the main tube radius.

    If the matching volume is included to the computation, the mass

    .. math::
        m_{mv} = \\dfrac{\\rho}{\\pi r_{side}} \\frac{\\delta}{8} \
            \\left(1 + 0.207 \\delta^3 \\right)

    is added to the mass :math:`m_s`.

    The flow conservation at the junction and the effect of the mass are taken
    into account by adding two additional variables :math:`\\gamma, \\zeta`:

    .. math::
        \\begin{cases}
        \\left( \\begin{smallmatrix} u_1 \\\\ u_2 \\\\ u_3 \\end{smallmatrix} \\right) \
            = T^{\\star} \\left( \\begin{smallmatrix} \\gamma \\\\ \\zeta \\end{smallmatrix} \\right) \\\\
        M \\partial_t \\left( \\begin{smallmatrix} \\gamma \\\\ \\zeta \\end{smallmatrix} \\right) \
            + T \\left( \\begin{smallmatrix} p_1 \\\\ p_2 \\\\ p_3 \\end{smallmatrix} \\right) = 0
        \\end{cases}

    with :math:`T=\\left( \\begin{smallmatrix} 1 & 0 & -1 \\\\ 0 & 1 & -1 \\end{smallmatrix} \\right)` and
    :math:`M=\\left( \\begin{smallmatrix} m_{11} & m_{12} \\\\ m_{12} & m_{11} \\end{smallmatrix} \\right)`.
    More information are given in [Ernoult]_.

    Parameters
    ----------
    label : string
        The label of the junction
    scaling : :py:class:`Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : {'PH1', 'VH1'}, optional
        The convention used in this junction. The default is 'PH1'.
    matching_volume : boolean, optional
        Include or not the matching volume between the main and the side tubes
        in the masses of the junction. The default is False.

    References
    ----------
    .. [Chaigne] Antoine Chaigne and Jean Kergomard, "Acoustics of Musical \
        Instruments". Springer, New York, 2016.

    .. [Dubos] V Dubos, J. Kergomard, A. Khettabi, J.-P. Dalmont, D. H. Keefe, \
        and C. J. Nederveen, "Theory of sound propagation in a duct with a \
        branched tube using modal decomposition," Acta Acustica united with \
        Acustica, vol. 85, no. 2, pp. 153–169, 1999.

    .. [Ernoult] Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full\
        waveform inversion for bore reconstruction of woodwind-like \
        instruments", submitted to Acta Acustica. https://hal.inria.fr/hal-03231946

    """

    def __init__(self, label, scaling, convention='PH1', matching_volume=False):
        super().__init__(label, scaling, convention)
        self.matching_volume = matching_volume

    def is_compatible_for_modal(self):
        return True

    def get_scaling_masses(self):
        """
        Get the scaling coefficient for acoustic mass.

        An acoustic mass has the dimension of an impedance over a frequency.

        Returns
        -------
        float
            The scaling coefficient

        """
        return self.scaling.get_impedance() * self.scaling.get_time()

    def is_passive(self, m11, m12, m22):
        """
        Returns True if the junction is passive and False if not.

        The passivity condition is :math:`|m_{12}|^2 < m_{11} m_{22}`

        Parameters
        ----------
        m11, m12, m11 : float
            acoustic masses of the junction

        """
        return abs(m12)**2 < m11 * m22

    def compute_passive_masses(self, r_main, r_side, rho, cond=20):
        """
        Returns a passive version of the junction, usable for time simulation.

        Parameters
        ----------
        r_main : float
            Radius of the main pipe, at the point of junction
        r_side : float
            Radius of the side pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        cond : float
            The largest allowed conditioning number for the mass matrix.

            .. warning:: Large conditioning means that some parasite modes\
                will resonate strongly, resulting in numerical error

        """
        eps = 1/cond
        m11, m12, m22 = self.compute_masses(r_main, r_side, rho)
        large_mass = (m11 + m22)/2 + abs(m12)
        m11_new = max(m11, eps * large_mass)
        m22_new = max(m22, eps * large_mass)
        if abs(m12)**2 >= m11 * m22 * (1-eps):
            m12_new = sqrt(m11_new * m22_new) * sign(m12) * (1-eps)
        else:
            m12_new = m12
        assert abs(m12_new)**2 < m11_new * m22_new
        return m11_new, m12_new, m22_new

    def compute_diagonal_masses(self, r_main, r_side, rho):
        """
        Diagonalized mass matrix M, and interaction matrix T.

        Evolution equation of the junction variables gamma is :

        .. math ::
            M dt \\left( \\begin{smallmatrix} \\gamma_1 \\\\ \\gamma_2 \\end{smallmatrix}\\right)
            - T \\left( \\begin{smallmatrix}p_1 \\\\ p_2 \\\\ p_3 \\end{smallmatrix}\\right) = 0

        Contribution to the flow is :math:`\\lambda_{123} = T^* \\gamma`

        Parameters
        ----------
        r_main : float
            Radius of the main pipe, at the point of junction
        r_side : float
            Radius of the side pipe, at the point of junction
        rho : float
            Air density, at the point of the junction

        Returns
        -------
        M : (2,) array
            Diagonal of the new mass matrix
        T : (2, 3) array
            New interaction matrix

        """
        m11, m12, m22 = self.compute_masses(r_main, r_side, rho)

        kappa = (m11 - m22)/(2*m12)
        D = np.sqrt(kappa**2 + 1)
        m_mean = (m11 + m22)/2
        m_plus = m_mean + m12*D
        m_minus = m_mean - m12*D
        tau_plus = -1/np.sqrt(2)*sqrt(1+kappa/D)
        tau_minus = -1/np.sqrt(2)*sqrt(1-kappa/D)

        T_new = np.array([[-tau_minus, tau_plus, tau_minus - tau_plus],
                          [tau_plus, tau_minus, -tau_plus - tau_minus]])
        M_new = np.array([m_minus, m_plus])

        return M_new, T_new

    def __compute_length_corr(self, a, b):
        d = b/a
        t_i = b*(0.82 - 0.193*d - 1.09*d**2 + 1.27*d**3 - 0.71*d**4)
        t_mv = 0
        if self.matching_volume:
            t_mv += b*d/8*(1 + .207*d**3)
        t_a = b*(-0.37 + 0.087*d)*d**2
        return t_i, t_mv, t_a

    def compute_masses(self, r_main, r_side, rho):
        """
        Compute masses of the junction from physical parameters.

        Parameters
        ----------
        r_main : float
            Radius of the main pipe, at the point of junction
        r_side : float
            Radius of the side pipe, at the point of junction
        rho : float
            Air density, at the point of the junction

        Returns
        -------
        m11, m12, m11 : float
            acoustic masses of the junction

        """

        assert r_main > 0
        assert r_side > 0
        if r_main < r_side:
            msg = ("The radius of the main pipe cannot be smaller"
                   " than that of a side pipe !")
            # raise ValueError(msg)
            warnings.warn(msg)

        coef_scaling_masses = self.get_scaling_masses()
        a = r_main  # Radius of the main pipe
        b = r_side  # Radius of the hole
        t_i, t_mv, t_a = self.__compute_length_corr(a, b)
        t_s = t_i + t_mv
        m_s = rho*t_s/(pi*b**2) / coef_scaling_masses
        m_a = rho*t_a/(pi*a**2) / coef_scaling_masses

        m11 = m_s + m_a/4
        m12 = m_s - m_a/4

        return m11, m12, m11

    def get_diff_masses(self, r_main, r_side, rho, d_r_main, d_r_side):
        """
        Compute the derivate of the masses from physical parameters.

        Parameters
        ----------
        r_main : float
            Radius of the main pipe, at the point of junction
        r_side : float
            Radius of the side pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        d_r_main : float
            The derivate of the main bore radius
        d_r_side : float
            The derivate of the side hole radius

        Returns
        -------
        dm11, dm12, dm11 : float
            the derivate of the acoustic masses

        """
        coef_scaling_masses = self.get_scaling_masses()

        a = r_main  # Radius of the main pipe
        b = r_side  # Radius of the hole
        da = d_r_main
        db = d_r_side
        d = b/a
        ddelta = (db/a - da/a*d)

        t_i, t_mv, t_a = self.__compute_length_corr(a, b)

        t_s = t_i + t_mv
        m_s = rho*t_s/(pi*b**2) / coef_scaling_masses

        # d_ts = (db*(0.82 - 2*0.193*d - 3*1.09*d**2 + 4*1.27*d**3 - 5*0.71*d**4)
        #       + da*(0.193*d**2 + 2*1.09*d**3 - 3*1.27*d**4 + 4*0.71*d**5))
        d_ti = db/b*t_i + b*ddelta*(-0.193 - 2*1.09*d + 3*1.27*d**2 - 4*0.71*d**3)
        d_tmv = 0
        if self.matching_volume:
            d_tmv += db/b*t_mv + ddelta*b/8*(1 + 4*.207*d**3)
        d_ts = d_ti + d_tmv
        d_ms = (-2*db/b + d_ts/t_s)*m_s

        t_a = b*(-0.37 + 0.087*d)*d**2
        m_a = rho*t_a/(pi*a**2) / coef_scaling_masses
        # d_ta = db*(-3*0.37 + 4*0.087*d)*d**2 + da*(2*0.37 - 3*0.087*d)*d**3
        d_ta = db/b*t_a + b*ddelta*(-2*0.37*d + 3*0.087*d**2)
        d_ma = (-2*da/a + d_ta/t_a)*m_a

        dm11 = d_ms + d_ma/4
        dm12 = d_ms - d_ma/4
        return dm11, dm12, dm11

    def diff_diagonal_masses(self, r_main, r_side, rho, d_r_main, d_r_side):
        """
        Differentiate the junction matrices, for diagonal mass matrix.

        .. warning::
            Computation valid only if m11==m22!!!

        Parameters
        ----------
        r_main : float
            Radius of the main pipe, at the point of junction
        r_side : float
            Radius of the side pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        d_r_main : float
            The derivate of the main bore radius
        d_r_side : float
            The derivate of the side hole radius.

        Returns
        -------
        dM : float
            Derivate of the diagonal mass matrix
        dT : float
            Derivate of the interaction matrix T (dT=0, every time)

        """
        dm11, dm12, dm22 = self.get_diff_masses(r_main, r_side, rho, d_r_main,
                                                d_r_side)
        assert dm11 == dm22
        dm_plus = dm11 + dm12
        dm_minus = dm11 - dm12
        dM = np.array([dm_minus, dm_plus])
        dT = np.array([[0, 0, 0], [0, 0, 0]])
        return dM, dT

class JunctionSwitch(PhysicalJunction):
    r"""
    Models the switch junction between 3 pipes with discontinuity of sections.

    Following the opening state of the switch the acoustic waves are oriented
    from pipe1 to pipe2 (open switch) or from pipe 1 to pipe 3 (closed switch).
    Intermediate state are accepted.
    An acoustic mass can be added in case of discontinuity of sections
    (cf. :py:class:`JunctionDiscontinuity<JunctionDiscontinuity>`).

    The equations at the junction are the flow conservation

    .. math::
        \begin{cases}
        u_1 - u_2 - u_3 =0 \\
        p_1 = \alpha p_2 + (1 - \alpha) p_3 - \left(\alpha m_{12} + (1-\alpha) m_{13} \right) \partial_t u_1
        \end{cases}

    And :math:`u_2=\alpha u_1`; :math:`u_3=(1 - \alpha) u_1`.
    In openwind, it is rewritten as a linear system by introducing an additional
    variable :math:`\gamma`:

    .. math::
        \begin{cases}
        \left( \begin{smallmatrix} u_1 \\ - u_2 \\ -u_3 \end{smallmatrix} \right) = T^{\star} \gamma \\
        M \partial_t \gamma + T \left( \begin{smallmatrix} p_1 \\ p_2 \\ p_3 \end{smallmatrix} \right) = 0
        \end{cases}

    where :math:`T=(1, -\alpha, -(1 - \alpha))` and
    :math:`M=\left(\alpha m_{12} + (1-\alpha)m_{13} \right)` with
    :math:`m_{12}, m_{13}` the masses due to cross-section discontinuity between
    pipes 1 and 2 and pipes 1 and 3 respectively


    Parameters
    ----------
    label : string
        The label of the junction
    scaling : :py:class:`Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : {'PH1', 'VH1'}, optional
        The convention used in this junction. The default is 'PH1'.
    discontinuity_mass : boolean, optional.
        If False, the masses are set to 0. Default is False.

    """

    def __init__(self, label, scaling, convention='PH1', discontinuity_mass=False):
        super().__init__(label, scaling, convention)
        self.discontinuity_mass = discontinuity_mass
        self.disc_junc = JunctionDiscontinuity(label, scaling, convention)

    def is_compatible_for_modal(self):
        return True

    def compute_masses(self, r1, r2, r3, rho, opening_factor):
        """
        Compute masses of the Junction from physical parameters.

        Parameters
        ----------
        r1 : float
            Radius of the 1st pipe, at the point of junction
        r2 : float
            Radius of the 2nd pipe, at the point of junction
        r3 : float
            Radius of the 3rd pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        opening_factor : float within [0, 1]
            The switch state.

            - 1: the switch is "open". The pipe 1 is connected to pipe 2
            - 2: the switch is "closed". The pipe 1 is connected to pipe 3

        Returns
        -------
        float
            The scaled acoustic mass associated to the junction.

        """
        assert r1 > 0
        assert r2 > 0
        assert r3 > 0
        if r1 < r3 or r2 < r3:
            msg = ("The radius of the main pipe is smaller"
                   " than that of a side pipe !")
            # raise ValueError(msg)
            warnings.warn(msg)

        T_J = np.array([[1, -opening_factor, opening_factor-1]])
        if self.discontinuity_mass:
            m12 = self.disc_junc.compute_mass(r1, r2, rho)
            m13 = self.disc_junc.compute_mass(r1, r3, rho)
            M_J = np.array([[m13 + opening_factor*(m12 - m13)]])
        else:
            M_J = np.array([[0.]])
        return M_J, T_J

    def get_diff_mass(self, r1, r2, r3, rho, opening_factor, dr1, dr2, dr3):
        """
        Derivate the mass w.r. to the radii.

        Parameters
        ----------
        r1 : float
            Radius of the 1st pipe, at the point of junction
        r2 : float
            Radius of the 2nd pipe, at the point of junction
        r2 : float
            Radius of the 3rd pipe, at the point of junction
        rho : float
            Air density, at the point of the junction
        opening_factor : float within [0, 1]
            The switch state.

            - 1: the switch is "open". The pipe 1 is connected to pipe 2
            - 2: the switch is "closed". The pipe 1 is connected to pipe 3
        dr1 : float
            The derivative of the first pipe radius.
        dr2 : float
            The derivative of the second pipe radius.
        dr2 : float
            The derivative of the third pipe radius.

        Returns
        -------
        float
            The derivative of the mass.

        """
        if self.discontinuity_mass:
            dm12 = self.disc_junc.get_diff_mass(r1, r2, rho, dr1, dr2)
            dm13 = self.disc_junc.get_diff_mass(r1, r3, rho, dr1, dr3)
            return dm13 + opening_factor*(dm12 - dm13)
        else:
            return 0.
