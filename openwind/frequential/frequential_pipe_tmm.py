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

from openwind.continuous import (ThermoviscousBessel,
                                 ThermoviscousLossless,
                                 Keefe,
                                 MiniKeefe,
                                 EndPos)
from openwind.frequential import FrequentialComponent, FPipeEnd
from .tmm_tools import cone_lossy, cone_lossless


class TMMPipeEnd(FPipeEnd):
    """
    Access to one end of a FrequentialPipeTMM.

    Parameters
    ----------
    f_pipe : :py:class:`FrequentialPipeTMM`
        the pipe to which this End corresponds
    pos : {:py:class:`EndPos.MINUS, EndPos.PLUS<openwind.continuous.netlist.EndPos>`}
        Position in the pipe's indices. MINUS if start of the pipe, PLUS if end.

    Attributes
    ----------
    convention: {'PH1', 'VH1'}
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
    """


    def __init__(self, f_pipe, pos):
        self.f_pipe = f_pipe
        assert isinstance(pos, EndPos)
        self.pos = pos
        self.convention = 'PH1'

    def get_index(self):
        """
        Index of the H1 variable at this pipe-end, in the assembled vector
        of all unknowns Uh.

        Returns
        -------
        int
        """
        if self.pos == EndPos.MINUS:
            return self.f_pipe.get_first_index() + 4
        elif self.pos == EndPos.PLUS:
            return self.f_pipe.get_first_index() + 6
        assert False


class FrequentialPipeTMM(FrequentialComponent):
    """
    Frequential pipe whose transfer matrix is given.

    A dirty trick is used to place the frequency-dependent
    coefficients A, B, C, D on the diagonal:
    we add four additional variables, such that the Schur complement of
    the new matrix restores the correct system of equations.

    From a transfer matrix such as:

    .. math::
        \\begin{pmatrix} p_{(-)} \\\\ v_{(-)} \\end{pmatrix}
        \\begin{pmatrix} A & B \\\\ C & D \\end{pmatrix}
         = \\begin{pmatrix} p_{(+)} \\\\ v_{(+)} \\end{pmatrix}

    It becomes:

    .. math::
        \\begin{pmatrix}
        \\frac{1}{A}    & \\cdot  & \\cdot & \\cdot & \\cdot & \\cdot & 1      & \\cdot \\\\
        \\cdot & \\frac{1}{B} & \\cdot & \\cdot & \\cdot & \\cdot & \\cdot & 1      \\\\
        \\cdot & \\cdot  & \\frac{1}{C}& \\cdot & \\cdot & \\cdot & 1      & \\cdot \\\\
        \\cdot & \\cdot  & \\cdot & \\frac{1}{D} & \\cdot & \\cdot & \\cdot & 1      \\\\
        \\cdot & \\cdot  & \\cdot & \\cdot & \\cdot & 1      & \\cdot & \\cdot \\\\
        1      & 1       & \\cdot & \\cdot & 1      & \\cdot & \\cdot & \\cdot \\\\
        \\cdot & \\cdot  & \\cdot & \\cdot & \\cdot & \\cdot & \\cdot & -1 \\\\
        \\cdot & \\cdot  & 1      & 1      & \\cdot & 1      & \\cdot & \\cdot
        \\end{pmatrix}
        \\begin{pmatrix}
            \\gamma_1 \\\\ \\gamma_2 \\\\ \\gamma_3 \\\\
            \\gamma_4 \\\\ p_{(-)} \\\\ v_{(-)} \\\\ p_{(+)} \\\\ v_{(+)}
        \\end{pmatrix}
        =
        \\begin{pmatrix}
            \\cdot \\\\ \\cdot \\\\ \\cdot \\\\
            \\cdot \\\\ -\\lambda_{(-)} \\\\ \\cdot \\\\ -\\lambda_{(+)} \\\\ \\cdot
        \\end{pmatrix}

    The :math:`A, B, C, D` are the coefficients given by [Chaigne_cone]_ for a
    cone (eq.(7.83), p.325).  The thermo-viscous losses are taken equal to the
    ones of a cylinder with a radius: :math:`R_{eq}=(2 R_{min} + R_{max})/3`.

    This blocks are placed in the global matrices :math:`A_h, L_h` at the indices
    corresponding to the 8 dof of this pipe.

    .. seealso::
        :py:mod:`tmm_tools<openwind.frequential.tmm_tools>`
            More information on the TMM coefficients.


    .. warning::
        This solution can be used only with pipes discribed by a
        :py:class:`Cone<openwind.design.cone.Cone>`
        (for which :py:meth:`is_TMM_compatible()\
        <openwind.design.design_shape.DesignShape.is_TMM_compatible>` is True).

        It also necessitates a uniform temperature.

    Parameters
    ----------
    pipe : :py:class:`Pipe <openwind.continuous.pipe.Pipe>`
        the pipe that will be converted
    nb_sub : int, optional
        The number of sub-division of the pipe. Useful to improve the losses
        approximation in conical pipe. Defaults to 1.
    reff_tmm_losses : {'integral', 'third', 'mean'}, optional
        Formula used to compute the effective radius used in the losses.
        Default is 'integral'.

    References
    ----------
    .. [Chaigne_cone] Chaigne, Antoine, and Jean Kergomard. 2016. "Acoustics \
        of Musical Instruments. Modern Acoustics and Signal Processing. New \
        York: Springer. https://doi.org/10.1007/978-1-4939-3679-3.


    Attributes
    ----------
    end_minus, end_plus: :py:class:`TMMPipeEnd`
        The start and end of the pipe
    nL2, nH1: int
        The number of dof corresponding to the L2 and H1 variables in this pipe
    ntot_dof: int
        The total number of dof in the entire instrument (graph)

    """
    def __init__(self, pipe, nb_sub=1, reff_tmm_losses='integral'):
        self.pipe = pipe

        # Supports only conical shapes with constant temperature
        if not pipe.get_shape().is_TMM_compatible():
            raise ValueError('TMM can only be used with Cones and not'
                             ' {}'.format(pipe.get_shape()))
        if not self.pipe.get_physics().uniform:
            raise ValueError('TMM can only be used with uniform temperature.')

        self.end_minus = TMMPipeEnd(self, EndPos.MINUS)
        self.end_plus = TMMPipeEnd(self, EndPos.PLUS)
        self._nb_sub = nb_sub
        self._reff_tmm_losses = reff_tmm_losses

    def get_number_dof(self):
        return 8

    def get_ends(self):
        """
        The two ends of the pipe (minus and plus)

        Returns
        -------
        tuple of :py:class:`TMMPipeEnd`
        """
        return self.end_minus, self.end_plus

    def _compute_tmm_coefs(self, omegas_scaled):
        physics = self.pipe.get_physics()
        omegas = omegas_scaled / self.pipe.get_scaling().get_time()
        lpart = self.pipe.get_length()
        R0 = self.pipe.get_radius_at(0)
        R1 = self.pipe.get_radius_at(1)
        nb_sub = self._nb_sub
        sph = self.pipe.is_spherical_waves()
        losses = self.pipe.get_losses()
        if isinstance(losses, ThermoviscousBessel):  # with bessel losses
            A, B, C, D = cone_lossy(physics, lpart, R0, R1,
                                        omegas, nb_sub, sph, reff_tmm_losses=self._reff_tmm_losses)
        elif isinstance(losses, Keefe): # Keefe losses
            A, B, C, D = cone_lossy(physics, lpart, R0, R1,
                                          omegas, nb_sub, sph, 'keefe', reff_tmm_losses=self._reff_tmm_losses)
        elif isinstance(losses, MiniKeefe): # Keefe losses
            A, B, C, D = cone_lossy(physics, lpart, R0, R1,
                                          omegas, nb_sub, sph, 'minikeefe', reff_tmm_losses=self._reff_tmm_losses)
        elif isinstance(losses, ThermoviscousLossless):  # lossless
            A, B, C, D = cone_lossless(physics, lpart, R0, R1,
                                           omegas, sph)
        else:
            raise ValueError("FPipeTMM only supports losses"
                             " {False, 'bessel, 'keefe', 'minikeefe'}, not " + str(type(losses)))

        # Nondimensionalization of the TMM matrix
        B /= self.pipe.get_scaling().get_impedance()
        C *= self.pipe.get_scaling().get_impedance()
        assert np.allclose(A*D - B*C, 1.0)  # determinant should be 1
        return A, B, C, D

    def _compute_diags(self, omegas_scaled):
        A, B, C, D = self._compute_tmm_coefs(omegas_scaled)

        assert all(coef.shape == (len(omegas_scaled),) for coef in [A, B, C, D])
        local_Ah_diags = np.zeros((8, len(omegas_scaled)), dtype='complex128')
        for i, coef in enumerate([A, B, C, D]):
            local_Ah_diags[i, :] = 1/coef
        return local_Ah_diags

    def get_contrib_freq(self, omegas_scaled):
        local_Ah_diags = self._compute_diags(omegas_scaled)
        return self.get_indices(), local_Ah_diags

    def get_contrib_indep_freq(self):
        # local_Ah = coo_matrix([[0, 0, 0, 0,   0, 0, 1, 0],
        #                         [0, 0, 0, 0,   0, 0, 0, 1],
        #                         [0, 0, 0, 0,   0, 0, 1, 0],
        #                         [0, 0, 0, 0,   0, 0, 0, 1],

        #                         [0, 0, 0, 0,   0, 1, 0, 0],
        #                         [1, 1, 0, 0,   1, 0, 0, 0],
        #                         [0, 0, 0, 0,   0, 0, 0, -1],
        #                         [0, 0, 1, 1,   0, 1, 0, 0]])
        row = [r + self.get_first_index() for r in [0, 1, 2, 3, 4, 5, 5, 5, 6, 7, 7, 7]]
        col = [r + self.get_first_index() for r in [6, 7, 6, 7, 5, 0, 1, 4, 7, 2, 3, 5]]
        data = [ 1,  1,  1,  1,  1,  1,  1,  1, -1,  1,  1,  1]
        return row, col, data

    def get_contrib_dAh_freq(self, omegas_scaled, diff_index):
        """
        The gradient can not yet be computed with this discretization.
        """
        # TODO if we want to do optimization with TMM
        raise NotImplementedError

    def get_contrib_dAh_indep_freq(self, diff_index):
        """
        The gradient can not yet be computed with this discretization.
        """
        # TODO if we want to do optimization with TMM
        raise NotImplementedError
