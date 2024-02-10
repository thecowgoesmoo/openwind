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
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

from openwind.technical import InstrumentGeometry
# from openwind.algo_optimization import LevenbergMarquardt


class AdjustInstrumentGeometry:
    """
    Adjust one instrument geometry on another one.

    The adjustement of the two
    :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
    is based only on geometrical aspects. The norm of the deviation between the
    radius at the same points is minimized.

    .. warning::
        - Only the main bore is adjusted.
        - The two main bore pipes must have the same total length.

    Parameters
    ----------

    mm_adjust: :py:class:`InstrumentGeometry \
    <openwind.technical.instrument_geometry.InstrumentGeometry>`
        The geometry which must be adjusted (typically the simplest one)

    mm_target: :py:class:`InstrumentGeometry \
    <openwind.technical.instrument_geometry.InstrumentGeometry>`
        The target geometry, typically the more complex as a measured one

    """

    def __init__(self, mm_adjust, mm_target):
        ltot1 = mm_target.get_main_bore_length()
        ltot2 = mm_adjust.get_main_bore_length()
        if not np.isclose(ltot1, ltot2):
            raise ValueError("The two geometry must have the same total"
                             " length! Here {:.2e} and {:.2e}"
                             .format(ltot1, ltot2))
        self.x_evaluate = np.arange(0, ltot2, 1e-3)
        self.mm_adjust = mm_adjust
        self.mm_target = mm_target

    def _get_radius_mm(self, instrument_geometry):
        """
        Get the radius of the considered geometry at the position x_evaluate.

        Parameters
        ----------
        instrument_geometry : :py:class:`InstrumentGeometry \
        <openwind.technical.instrument_geometry.InstrumentGeometry>`
            The geometry for which the radius is estimated.

        Returns
        -------
        radius : np.array
            The array of the radius values.

        """
        radius = np.zeros_like(self.x_evaluate)
        for shape in instrument_geometry.main_bore_shapes:
            x_min, x_max = shape.get_endpoints_position()
            x_norm = ((self.x_evaluate - x_min.get_value()) /
                      (x_max.get_value() - x_min.get_value()))
            x_in = np.where(np.logical_and(x_norm >= 0, x_norm <= 1))
            radius[x_in] = shape.get_radius_at(x_norm[x_in])
        return radius

    def _get_diff_radius_mm(self, diff_index):
        """
        The radius differentiation w.r. to one design parameter.

        Only the radius of the mm_adjust is needed and computed...

        Parameters
        ----------
        diff_index : int
            The index of the design parameter considered in `optim_params` of
            the adjusted :py:class:`InstrumentGeometry \
            <openwind.technical.instrument_geometry.InstrumentGeometry>`.

        Returns
        -------
        diff_radius : np.array
            The value of the differentiation at each point.

        """
        diff_radius = np.zeros_like(self.x_evaluate)
        for shape in self.mm_adjust.main_bore_shapes:
            x_min, x_max = shape.get_endpoints_position()
            x_norm = ((self.x_evaluate - x_min.get_value()) /
                      (x_max.get_value() - x_min.get_value()))
            x_in = np.where(np.logical_and(x_norm >= 0, x_norm <= 1))
            diff_radius[x_in] = shape.get_diff_radius_at(x_norm[x_in],
                                                         diff_index)
        return diff_radius

    def _compute_residu(self, radius_adjust, radius_target):
        """
        Compute the residue between the radii.

        It is simply the difference between the two radius vector.

        Parameters
        ----------
        radius_adjust : np.array
            The radius array corresponding to the adjusted geometry.
        radius_target : np.array
            The radius array corresponding to the target geometry.

        Returns
        -------
        np.array
            The residue vector.

        """
        return radius_adjust - radius_target

    def _get_cost_grad_hessian(self, params):
        """
        Compute the cost, the gradient and the hessian of the problem.

        Parameters
        ----------
        params : np.array, list
            The value of the design parameters for which must be estimated the
            cost, the gradient and the hessian.

        Returns
        -------
        cost : float
            The cost.
        gradient : np.array
            The gradient vector.
        hessian : np.array
            The hessian matrix.

        """
        self.mm_adjust.optim_params.set_active_values(params)
        radius_target = self._get_radius_mm(self.mm_target)
        radius_adjust = self._get_radius_mm(self.mm_adjust)
        cost = self._compute_cost(radius_adjust, radius_target)
        gradient, hessian = self._compute_grad_hessian(radius_adjust,
                                                       radius_target)
        return cost, gradient, hessian

    def get_residual(self, params):
        """
        Compute the residual between the two geometries.

        Parameters
        ----------
        params : np.array, list
            The value of the design parameters for which must be estimated the
            cost, the gradient and the hessian.

        Returns
        -------
        residual : array
            The residual.
        """
        self.mm_adjust.optim_params.set_active_values(params)
        radius_target = self._get_radius_mm(self.mm_target)
        radius_adjust = self._get_radius_mm(self.mm_adjust)
        residual = self._compute_residu(radius_adjust, radius_target)
        return residual

    def get_jacobian(self, params):
        """
        The jacobian of the residual.

        Parameters
        ----------
        params : np.array, list
            The value of the design parameters for which must be estimated the
            cost, the gradient and the hessian.

        Returns
        -------
        jacob : np.array
            The jacobian.
        """
        self.mm_adjust.optim_params.set_active_values(params)
        nderiv = len(self.mm_adjust.optim_params.get_active_values())
        jacob = np.zeros([len(self.x_evaluate), nderiv])
        for diff_index in range(nderiv):
            jacob[:, diff_index] = self._get_diff_radius_mm(diff_index)
        return jacob

    def _get_cost_grad_hessian(self, params):
        """
        Compute the cost, the gradient and the hessian of the problem.

        Parameters
        ----------
        params : np.array, list
            The value of the design parameters for which must be estimated the
            cost, the gradient and the hessian.

        Returns
        -------
        cost : float
            The cost.
        gradient : np.array
            The gradient vector.
        hessian : np.array
            The hessian matrix.

        """
        residual = self.get_residual(params)
        jacobian = self.get_jacobian(params)
        cost = 0.5*np.linalg.norm(residual)**2
        gradient = jacobian.T.dot(residual)
        hessian = jacobian.T.dot(jacobian)
        return cost, gradient, hessian

    def optimize_geometry(self, max_iter=100, minstep_cost=1e-8, tresh_grad=1e-10,
                     iter_detailed=False):
        """
        Minimize the radius deviation between the two geometries.

        The minimmization used the Levenberg-Marquardt algorithm to reduce
        the mean-square deviation between the radius of the two geometries.

        Parameters
        ----------
        max_iter : int, optional
            The maximal number of iteration. The default is 100.
        minstep_cost : float, optional
            The minimal realtive evolution of the cost. The default is 1e-8.
        tresh_grad : float, optional
            The minimal value of the gradient. The default is 1e-10.
        iter_detailed : boolean, optional
            If the detail of each iteration is printed. The default is False.

        Returns
        -------
        :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
            The adjusted geometry

        """
        lb, ub = tuple(zip(*self.mm_adjust.optim_params.get_active_bounds()))
        if all(np.isinf(lb+ub)):
            algo = 'lm'
        else:
            algo = 'trf'
        result = least_squares(self.get_residual,
                               self.mm_adjust.optim_params.values,
                               jac=self.get_jacobian, bounds=(lb, ub),
                               verbose=1, method=algo, ftol=minstep_cost,
                               max_nfev=max_iter, gtol=tresh_grad)

        # result = LevenbergMarquardt(self._get_cost_grad_hessian,
        #                                   self.mm_adjust.optim_params.values,
        #                                   max_iter, minstep_cost,
        #                                   tresh_grad, iter_detailed)
        print('Residual error; {:.2e}'.format(result.cost))
        return self.mm_adjust


if __name__ == '__main__':
    """
    An example for which a spline with 4 points is adjusted on a geometry
    composed of 10 conical parts.
    """
    # the target geometry composed of ten conical parts
    x_targ = np.linspace(0, .1, 10)
    r_targ = np.linspace(5e-3, 1e-2, 10) - 2e-3*np.sin(x_targ*2*np.pi*10)
    Geom = np.array([x_targ, r_targ]).T.tolist()
    mm_target_test = InstrumentGeometry(Geom)

    # the geometry which will be adjusted
    mm_adjust_test = InstrumentGeometry([[0, .1, 5e-3, '~5e-3', 'spline',
                                          '.03', '.06', '~7e-3', '~4e-3']])

    # plot initial state
    fig = plt.figure()
    mm_target_test.plot_InstrumentGeometry(figure=fig, label='target')
    mm_adjust_test.plot_InstrumentGeometry(figure=fig, label='initial',
                                           linestyle='--')

    # the optimization
    test = AdjustInstrumentGeometry(mm_adjust_test, mm_target_test)
    adjusted = test.optimize_geometry(iter_detailed=True)

    # plot final state
    adjusted.plot_InstrumentGeometry(figure=fig, label='final', linestyle=':')
