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


def implemented_observation(observable: str):
    """
    One observable function and its derivative w.r. to the scaled impedance.

    Chose between the already implemented observalbe :math:`\\phi(Z)`. The
    three following function must be implemented:

    .. math::
        \\phi(Z) \\\\
        \\frac{d\\phi}{dZ} \\\\
        \\frac{d\\phi}{dZ}

    The derivatives must take into account that :math:`Z` is complex:

     .. math ::
         \\frac{d \\phi}{dZ} = \\frac{1}{2} \\left( \\frac{d \\phi }\
        {d\\Re(Z)} -j \\frac{d \\phi}{d \\Im(Z)} \\right)

    .. warning ::
        Here the impedance is supposed to be scaled by the input
        characteristic impedance :math:`Z_c`.

    Parameters
    ----------
    observable : str
        The name of the observation. It can be:

            - 'impedance' : given :math:`\\phi(Z) = Z`
            - 'impedance_modulus' :  given :math:`\\phi(Z) = |Z|`
            - 'impedance_phase' :  given :math:`\\phi(Z) = angle(Z)`
            - 'reflection' :  given :math:`\\phi(Z) = R = (Z - 1)/(Z + 1)`
            - 'reflection_modulus' :  given :math:`\\phi(Z) = |R|`
            - 'reflection_phase' :  given :math:`\\phi(Z) = angle(R)`
            - 'reflection_phase_unwraped' :  given the unwrapped angle of R


    Returns
    -------
    callable
        The function returning the observation on Z
    callable
        The derivative of the observation with respect to Z and the derivative
        of the conjugate of the observation with respect to Z.

    """

    if observable == 'impedance':
        return (impedance, diff_impedance_wrZ)
    elif observable == 'impedance_modulus':
        return (module_square, diff_module_square_wrZ)
    elif observable == 'reflection':
        return (reflection, diff_reflection_wrZ)
    elif observable == 'impedance_phase':
        return (impedance_phase, diff_impedance_phase_wrZ)
    elif observable in ['reflection_phase', 'reflection_phase_unwraped']:
        return (reflection_phase, diff_reflection_phase_wrZ)
    elif observable == 'reflection_modulus':
        return (reflection_modulus_square, diff_reflection_modulus_square_wrZ)
    else:
        raise ValueError("Unknown observable. Chose between: \n"
                         "{'impedance', 'impedance_modulus', 'reflection',"
                         "'impedance_phase', 'reflection_phase',"
                         "'reflection_modulus', 'reflection_phase_unwraped'}")


# %% Impedance
def impedance(Z):
    """
    The complex impedance

    The observation is here the identity.

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    Z : array
        The complex scaled impedance

    """

    return Z


def diff_impedance_wrZ(Z):
    """
    The derivative of the impedance and its conjugate w.r. to Z

    .. math ::
        \\frac{dZ}{dZ} = 1 \\\\
        \\frac{d\\overline{Z}}{dZ} = 0

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """

    diff_obs = 1
    diff_conj_obs = 0
    return diff_obs, diff_conj_obs


# %% Impedance modulus
def module_square(Z):
    """
    The squared modulus of the complex impedance

    .. math ::
        \\phi(Z) = ||Z||^2

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    array
        The modulus squared of the scaled impedance

    """
    return np.abs(Z)**2


def diff_module_square_wrZ(Z):
    """
    The derivative of the squared modulus and its conjugate w.r. to Z

    .. math ::
        \\frac{d\\phi(Z)}{dZ} =  \\frac{d\\overline{\\phi(Z)}}{dZ} \
            = \\overline{Z}^T

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """
    diff_obs = Z.conj().T
    diff_conj_obs = diff_obs
    return diff_obs, diff_conj_obs


# %% reflection function
def reflection(Z):
    """
    The reflection function

    .. math ::
        \\phi(Z) = ||frac{Z - 1}{Z + 1}

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    array
        The reflection function

    """
    return (Z - 1)/(Z + 1)


def diff_reflection_wrZ(Z):
    """
    The derivative of the reflection function and its conjugate w.r. to Z

    .. math ::
        \\begin{align}
        \\frac{d\\phi(Z)}{dZ} &= \\frac{2}{(Z + 1)^2} \\\\
        \\frac{d\\overline{\\phi(Z)}}{dZ} &= 0
        \\end{align}

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """
    diff_obs = 2/(Z + 1)**2
    diff_conj_obs = 0
    return diff_obs, diff_conj_obs


# %% Impedance angle
def impedance_phase(Z):
    """
    The impedance angle

    .. math ::
        \\phi(Z) = angle(Z)

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    array
        The impedance angle

    """
    return np.angle(Z)


def diff_impedance_phase_wrZ(Z):
    """
    The derivative of the impedance angle and its conjugate w.r. to Z

    .. math ::
        \\frac{d\\phi(Z)}{dZ} = \\frac{d\\overline{\\phi(Z)}}{dZ} \
        = \\frac{-j \\overline{Z}}{2 |Z|^2}

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """
    diff_obs = -1j * Z.conjugate() / (2 * np.abs(Z)**2)
    diff_conj_obs = diff_obs
    return diff_obs, diff_conj_obs


# %% Reflection angle
def reflection_phase(Z):
    """
    The reflection function angle

    .. math ::
        \\phi(Z) = angle(R_{ef}) = angle\\left( \\frac{Z-1}{Z+1} \\right)

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    array
        The reflection function angle

    """

    return (np.angle(reflection(Z)))


def diff_reflection_phase_wrZ(Z):
    """
    The derivative of the reflection function angle and its conjugate w.r. to Z

    The derivative are computed by combination of the derivative of the
    reflection function in :func:`diff_reflection_wrZ()` and of the angle in
    :func:`diff_impedance_phase_wrZ`

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """
    diff_R_wrZ, _ = diff_reflection_wrZ(Z)
    diff_obs_wrR, diff_conj_obs_wrR = diff_impedance_phase_wrZ(reflection(Z))
    return diff_obs_wrR*diff_R_wrZ, diff_conj_obs_wrR*diff_R_wrZ


# %% Reflection modulus
def reflection_modulus_square(Z):
    """
    The squared modulus of the reflection function.

    .. math ::
        \\phi(Z) = ||R_{ef}||^2 = \\left\\Vert \\frac{Z-1}{Z+1} \\right\\Vert^2

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    array
        The squared modulus of reflection function

    """
    return np.abs(reflection(Z))**2


def diff_reflection_modulus_square_wrZ(Z):
    """
    The derivative of the squared modulus of the reflection function and its
    conjugate w.r. to Z

    The derivative are computed by combination of the derivative of the
    reflection function in :func:`diff_reflection_wrZ()` and of the modulus in
    :func:`diff_module_square_wrZ`

    Parameters
    ----------
    Z : array
        The complex scaled impedance.

    Returns
    -------
    diff_obs : array
        The derivative of the observable w.r. to Z.
    diff_conj_obs : array
        The derivative of the conjugate of the observable w.r. to Z.

    """
    diff_R_wrZ, _ = diff_reflection_wrZ(Z)
    diff_obs_wrR, diff_conj_obs_wrR = diff_module_square_wrZ(reflection(Z))
    return diff_obs_wrR*diff_R_wrZ, diff_conj_obs_wrR*diff_R_wrZ
