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
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix, vstack, csr_matrix
from scipy.sparse.linalg import splu
from scipy.optimize import minimize, least_squares, Bounds, LinearConstraint, NonlinearConstraint
import warnings

from openwind.frequential import FrequentialSolver, FrequentialInterpolation
from openwind.algo_optimization import homemade_minimization, print_cost
from openwind.inversion.observation import implemented_observation
from openwind.inversion.display_inversion import heatmap, annotate_heatmap


class InverseFrequentialResponse(FrequentialSolver):
    """
    Inverse frequential problem.

    Besides the direct problem (compute the frequential response from the
    geometry of the instrument), this class gives the possibility to compute
    the inverse problem: estimate geometric paramters from acoustical
    consideration. It can be useful to perform bore reconstruction or design.

    In particular this method allows to compute

    - the cost function: defined as the squared norm of the residual between\
    an observation on the simulated impedance and the same observation on a\
    target (measured) impedance .
    - the gradient of this cost function with respect to the design variables\
    defined in the :py:class:`OptimizationParameters\
    <openwind.design.OptimizationParameters>` stocked in the\
    :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`
    - perform the minimization of this cost function.

    More details are given in the paper [Ernoult_FWI]_

    Parameters
    ----------
    instru_physics : :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`
        The physics of the instruments studied.
    frequencies : np.array
        The frequency array used in the problem.
    target_impedances : list of array
        The scaled impedances (e.g. measured) used as target in the inverse
        problem.
    observable : string or tuple of callable, optional
        The observable use to construct the cost function. See
        :py:func:`set_observation` for more details. The default is 'reflection'.
    diffus_repr_var : boolean, optional
        Whether to use additional variables when computing the diffusive
        representation. See :py:class:`FrequentialSolver\
        <openwind.frequential.frequential_solver.FrequentialSolver>`. The default is
        False.
    notes : list of string, optional
        The list of notes corresponding to the target. It must correspond to
        the note names of the
        :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`.
        The default is None corresponding to all holes open.
    **discr_params :
        Discretization parameters. See\
        :py:class:`Mesh<openwind.discretization.mesh.Mesh>`

    References
    ----------
    .. [Ernoult_FWI] Ernoult A., Chabassier J., Rodriguez S., Humeau A., "Full\
        waveform inversion for bore reconstruction of woodwind-like \
        instruments", submitted to Acta Acustica. https://hal.inria.fr/hal-03231946


    Attributes
    ----------
    observable : tuple of two callables
        The function giving the observable from the impedance and its derivative
        with respect to the impedance
    restriction : coo_matrix
        The matrix extracting the impedance from the array of acoustic fields
        :math:`U_h`
    optim_params: :py:class:`OptimizationParameters<openwind.design.OptimizationParameters>`
        The object in which are stocked the design parameters of the problem.
    """

    def __init__(self, instru_physics, frequencies, target_impedances,
                 observable='reflection', diffus_repr_var=False,
                 notes=None, **discr_params):
        self.discr_params = discr_params
        self.n_obs = None  # give possibility to use other observation points
        super().__init__(instru_physics, frequencies,
                         diffus_repr_var=diffus_repr_var,
                         note=None, **discr_params)
        self.instru_physics = instru_physics
        self._assert_uniform_temperature()
        self.set_observation(observable)
        self.set_targets_list(target_impedances, notes)
        self.optim_params = instru_physics.instrument_geometry.optim_params
        self.set_restriction_matrix(self.n_obs)

    def set_observation(self, observable):
        """
        Define the observation used on the impedance in the cost function.

        The cost function is defined as the norm of the residual between an
        observation :math:`\\phi` on the simulated impedance :math:`Z` and the
        same observation on the target impedance :math:`Z^{\\odot}`:

            .. math ::
                F = ||\\phi(Z) - \\phi(Z^{\\odot})||

        To perform the optimization it is necessary to solve both the direct
        and inverse problem. It is therefore necessary to give both the
        definition of the observable and its derivative with respect to
        :math:`Z` by taking into account that :math:`Z` is complex:

            .. math ::
                \\frac{d \\phi}{dZ} = \\frac{1}{2} \\left( \\frac{d \\phi }\
                {d\\Re(Z)} -j \\frac{d \\phi}{d \\Im(Z)} \\right)

        .. warning ::
            Here the impedance is supposed to be scaled by the input
            characteristic impedance :math:`Z_c`.

        Parameters
        ----------
        observable : string or tuple of callable
            The observable.

            - If it is a string it must be associated to one of the observable\
            defined in :func:`openwind.inversion.implemented_observation`.
            - if it is a tuple of callable it must correspond to:

                - the observable :math:`\\phi`
                - its derivative :math:`d\\phi/dZ` and its conjugate \
                derivative :math:`d\\overline{\\phi}/dZ`

        """
        if observable == 'reflection_phase_unwraped':
            self.is_unwrap = True
        else:
            self.is_unwrap = False
        if type(observable) == tuple:
            self.observable = observable
        else:
            self.observable = implemented_observation(observable)

    def set_targets_list(self, target_impedances, notes):
        """
        Modify the impedance targets list and the corresponding notes.

        .. warning ::
            The given impedances must be scaled by the entrance characteristic
            impedance :math:`Z_c`.

        Parameters
        ----------
        target_impedances : list of float
            The list of target impedances.
        notes : list of string
            List of the note names corresponding to the given impedance.
        """

        if not type(target_impedances) == list:
            target_impedances = [target_impedances]
        if not type(notes) == list:
            notes = [notes]
        targets_list = []
        for impedance in target_impedances:
            if len(impedance.shape) == 1:
                impedance = np.array([impedance[:]])
            if np.mean(np.abs(impedance))>100:
                warnings.warn('The target impedance does not seem normalized (mean magnitude > 100; it should be around 1)')
            targets_list.append(impedance)
        assert len(notes) == len(targets_list)
        assert targets_list[0].shape[1] == len(self.frequencies)
        self.imped_targets = targets_list
        self.notes = notes

    def set_restriction_matrix(self, n_obs=None):
        """
        Construct the restriction matrix used to extract the acoustic fields
        at the observation point.

        By default, this matrix :math:`\\mathcal{R}` has only one non-zero
        value at the d.o.f.  corresponding to the pressure at the entrance of
        the instrument, given simply:

        .. math ::
            Z = \\mathcal{R} U

        Parameters
        ----------
        n_obs : list of int, optional
            The index of the dof corresponding to the observation points. The
            default is None, given the dof of the entrance pressure.
        """
        # TODO: adapter pour des points d'observation quelconques avec interp
        if not n_obs:  # by default the observation point is the source entry
            n_obs = [self.source_ref.get_source_index()]
        self._assert_rad_input_fixed()
        Nobs = len(n_obs)
        coef_obs = np.ones(Nobs) * self.scaling.get_impedance()
        self.restriction = coo_matrix((coef_obs,
                                      (list(range(Nobs)), n_obs)),
                                      shape=(Nobs, (self.n_tot)))

    def _assert_rad_input_fixed(self):
        """ Check that the radius is not variable at the observation point."""
        Nderiv = len(self.optim_params.get_active_values())
        diff_rad_in = np.zeros(Nderiv)
        for index in range(Nderiv):
            diff_rad_in[index] = self.source_ref.end.get_diff_radius(index)
        if any(diff_rad_in != 0):
            raise ValueError('The entrance radius can not yet be optimized.')

    def _assert_uniform_temperature(self):
        for f_pipe in self.f_pipes:
            if not f_pipe.pipe.get_physics().uniform:
                raise ValueError('Instrument with non-uniform temperature can not yet be optimized')


    def modify_parts(self, new_optim_values):
        """
        Modify the geometry and recompute frequential elements.

        After having modifying the geometric values:

            - the netlist is updated (allowing to modify the graph)
            - the frequential components are recomputed (allowing to modify \
                                                         the mesh)
            - the matrices corresponding to the frequential elements are \
                recomputed
            - the restriction matrix is updated

        Parameters
        ----------
        new_optim_values : list of float
            The new value of the design parameters.
        """

        self.optim_params.set_active_values(new_optim_values)
        self.instru_physics.update_netlist()
        self._convert_frequential_components()
        self._organize_components()
        self._construct_matrices_pipes()
        self._construct_matrices_connectors()
        self.set_restriction_matrix(self.n_obs)

# %% Define and compute cost
    def __observation(self, impedance):
        return self.observable[0](impedance)

    def __diff_observation_wrZ(self, impedance):
        """return the derivative of the observable w.r. to Z, AND the
        derivative of the CONJUGATE of the observable w.r. to Z.
        Warnings! it is a derivation wr to a complex vector: \
            d/dZ = (d/d(real(Z)) -j*d/d(imag(Z)))/2
        """
        return self.observable[1](impedance)

    def __impedance_scaled(self, Uh):
        return self.restriction.dot(Uh) / self.get_ZC_adim()

    def __target_norm(self):
        if self.is_unwrap:
            return np.sqrt(np.sum(np.abs(np.unwrap(self._target))**2))
        else:
            return np.sqrt(np.sum(np.abs(self._target)**2))

    def __diff_imped_scaled_wrUh(self):
        return self.restriction / self.get_ZC_adim()

    def __compute_residu(self, Uh, ind_freq):
        impedance = self.__impedance_scaled(Uh)
        observation = self.__observation(impedance)
        residu = (observation - self._target[:, ind_freq])/self.__target_norm()
        return np.append(residu.real, residu.imag)

    def __diff_observation_wrU(self, Uh):
        impedance = self.__impedance_scaled(Uh)
        (diff_obs_wrZ,
         diff_conj_obs_wrZ) = self.__diff_observation_wrZ(impedance)
        diff_imped = self.__diff_imped_scaled_wrUh()
        diff_obs = coo_matrix(diff_obs_wrZ).dot(diff_imped)
        diff_conj_obs = coo_matrix(diff_conj_obs_wrZ).dot(diff_imped)
        return diff_obs, diff_conj_obs

    def __diff_residu_wrUh(self, Uh):
        diff_obs, diff_conj_obs = self.__diff_observation_wrU(Uh)
        diff_real_residu = 0.5*(diff_obs + diff_conj_obs)/self.__target_norm()
        diff_imag_residu = -.5j*(diff_obs - diff_conj_obs)/self.__target_norm()
        return vstack([diff_real_residu, diff_imag_residu])

    def __compute_cost(self, residu):
        """WARNING: do not change this method or all the gradient computation
        must be modified"""
        return .5 * np.linalg.norm(residu)**2

# %% Gradient computation
    def __computedAH(self):
        omegas_scaled = 2*np.pi*self.frequencies * self.scaling.get_time()
        n_tot = self.n_tot
        # self.dAh_nodiag_tot = list()
        self.dAh_diags_tot = list()
        for diff_index in range(len(self.optim_params.get_active_values())):
            # initiate matrices
            # row = list()
            # col = list()
            # data = list()
            dAh_diags = np.zeros((n_tot, len(omegas_scaled)),
                                 dtype='complex128')
            # fill the matrices
            for f_comp in self.f_components:
                # row_comp, col_comp, data_comp = f_comp.get_contrib_dAh_indep_freq(diff_index)
                # row.append(row_comp)
                # col.append(col_comp)
                # data.append(data_comp)
                ind_f, data_f = f_comp.get_contrib_dAh_freq(omegas_scaled,
                                                            diff_index)
                dAh_diags[ind_f, :] += data_f

            # row_a = np.concatenate(row)
            # col_a = np.concatenate(col)
            # data_a = np.concatenate(data)
            # dAh_nodiag = csr_matrix((data_a,(row_a, col_a)), shape=(n_tot, n_tot),
            #                         dtype='complex128' )
            # assert np.all(dAh_nodiag.diagonal() == 0)
            # self.dAh_nodiag_tot.append(dAh_nodiag)
            self.dAh_diags_tot.append(dAh_diags)

    def __compute_dAhU(self, Uh, diff_index, ind_freq):
        # dAh = self.dAh_nodiag_tot[diff_index]
        # instead of setting the diagonal of dAH, the multiplication is divided
        # in 2 parts. It is faster
        # return dAh.dot(Uh) + self.dAh_diags_tot[diff_index][:, ind_freq]*Uh
        return self.dAh_diags_tot[diff_index][:, ind_freq]*Uh

    def __GradAdjoint(self, residu, Ahlu, Uh, ind_freq):
        Nderiv = len(self.optim_params.get_active_values())
        dresidu_dU = 2*self.__diff_residu_wrUh(Uh)
        sourceAdj = dresidu_dU.conj().T.dot(residu)
        lambdaAdjconj = Ahlu.solve(-1*sourceAdj.conjugate(), 'T')
        grad = np.zeros([Nderiv])
        for diff_index in range(Nderiv):
            dAhU = self.__compute_dAhU(Uh, diff_index, ind_freq)
            grad[diff_index] = (lambdaAdjconj @ dAhU).real
        return grad

    def __GradFrechet(self, residu, Ahlu, Uh, ind_freq):
        Nderiv = len(self.optim_params.get_active_values())
        dresidu_dU = 2*self.__diff_residu_wrUh(Uh)
        grad = np.zeros([Nderiv])
        jacob = np.zeros([self.restriction.shape[0]*2, Nderiv])
        for diff_index in range(Nderiv):
            dAhU = self.__compute_dAhU(Uh, diff_index, ind_freq)
            dU = -1 * Ahlu.solve(dAhU)
            jacob[:, diff_index] = dresidu_dU.dot(dU).T.real
            grad[diff_index] = jacob[:, diff_index].dot(residu)
        hessian = jacob.T.dot(jacob)
        return grad, hessian

    def __GradFiniteDiff(self, cost_init, stepSize=1e-8):
        Nderiv = len(self.optim_params.get_active_values())
        gradFor = np.zeros(Nderiv)
        gradBack = np.zeros(Nderiv)
        params_init = self.optim_params.get_active_values()
        params = self.optim_params.get_active_values()
        # For finite diff we can suppose that the netlist and the must can stay
        # unchanged
        for diff_index in range(Nderiv):
            params[diff_index] = params_init[diff_index] + stepSize
            self.optim_params.set_active_values(params)
            self._construct_matrices_pipes()
            self._construct_matrices_connectors()
            costFor = self.get_cost_grad_hessian()[0]
            gradFor[diff_index] = (costFor - cost_init) / stepSize

            params[diff_index] = params_init[diff_index] - stepSize
            self.optim_params.set_active_values(params)
            self._construct_matrices_pipes()
            self._construct_matrices_connectors()
            costBack = self.get_cost_grad_hessian()[0]
            gradBack[diff_index] = (cost_init - costBack) / stepSize

            params[diff_index] = params_init[diff_index]
        self.get_cost_grad_hessian(params_init)
        return (gradFor + gradBack) / 2

    def __jacobian(self, Ahlu, Uh, ind_freq):
        Nderiv = len(self.optim_params.get_active_values())
        dresidu_dU = 2*self.__diff_residu_wrUh(Uh)
        jacobian = np.zeros([self.restriction.shape[0]*2, Nderiv])
        for diff_index in range(Nderiv):
            dAhU = self.__compute_dAhU(Uh, diff_index, ind_freq)
            dU = -1 * Ahlu.solve(dAhU)
            jacobian[:, diff_index] = dresidu_dU.dot(dU).T.real
        return jacobian

# %% Global cost, gradient, hessian evaluations

    def __residuals_jacobian_1note(self):
        Nderiv = len(self.optim_params.get_active_values())
        Nres = self._target.shape[0]*2  # observation points *2 (real and imag)
        residuals = np.zeros(Nres*self._target.shape[1])
        jacobian = np.zeros((Nres*self._target.shape[1], Nderiv))
        Ah, ind_diag = self._initialize_Ah_diag()
        Lh = self.Lh.toarray()
        self.__computedAH()
        for nf in range(len(self.frequencies)):
            # Ah.setdiag(self.Ah_diags[:, nf])
            Ah.data[ind_diag] = self.Ah_diags[:, nf]
            Ahlu = splu(Ah, permc_spec='NATURAL')
            Uh = Ahlu.solve(Lh)[:, 0]
            residuals[Nres*nf:Nres*(nf+1)] = self.__compute_residu(Uh, nf)
            jacobian[Nres*nf:Nres*(nf+1), :] = self.__jacobian(Ahlu, Uh, nf)
        if self.is_unwrap:
            residuals = (np.unwrap(residuals*self.__target_norm())
                         / self.__target_norm())
        return residuals, jacobian

    def residuals_jacobian(self, params_values=list()):
        """
        The residual and its jacobian for a set of design variables.

        The direct problem is solved to compute the residual. The returned
        vector combines the real and imaginary part of the residual for
        each frequency and each fingerings considered, organized as:

            .. math ::
                (\\Re(r_0^0), \\Im(r_0^0), \\Re(r_1^0), \\ldots, r_i^j, \
                \\ldots, \\Im(r_n^N))

        with :math:`n` the number of frequencies taken into account and
        :math:`N` the number of fingering. The total length of the residual is
        :math:`2\\times n \\times N`

        The jacobian is computed by solving the inverse problem.

        Parameters
        ----------
        params_values : list of float, optional
            The value of the design variable. The defautl is an empty list,
            for which the design variables keep their current values.

        Returns
        -------
        residual: np.array
            The residual vector
        jacobian: np.array
            The jacobian matrix
        """
        if len(params_values) > 0:
            self.modify_parts(params_values)
        Nres = (self.imped_targets[0].shape[0]*2
                * self.imped_targets[0].shape[1])
        Nderiv = len(self.optim_params.get_active_values())
        residuals = np.zeros(Nres*len(self.imped_targets))
        jacobian = np.zeros((Nres*len(self.imped_targets), Nderiv))
        for k_note in range(len(self.notes)):
            self._target = self.__observation(self.imped_targets[k_note])
            self.set_note(self.notes[k_note])
            residuals1, jacobian1 = self.__residuals_jacobian_1note()
            residuals[k_note*Nres:(k_note+1)*Nres] = residuals1
            jacobian[k_note*Nres:(k_note+1)*Nres, :] = jacobian1
        return residuals, jacobian

    def __cost_grad_hessian_1note(self, grad_type=None, stepSize=1e-8):
        gradient = None
        hessian = None
        Nderiv = len(self.optim_params.get_active_values())

        residu = np.zeros((self._target.shape[0]*2, self._target.shape[1]))

        # Ah = self.Ah_nodiag.tocsc()
        Ah, ind_diag = self._initialize_Ah_diag()
        Lh = self.Lh.toarray()

        if grad_type == 'frechet' or grad_type == 'adjoint':
            self.__computedAH()
            gradient = np.zeros([Nderiv])
            if grad_type == 'frechet':
                hessian = np.zeros([Nderiv, Nderiv])
        for ind_freq in range(len(self.frequencies)):
            # Ah.setdiag(self.Ah_diags[:, ind_freq])
            Ah.data[ind_diag] = self.Ah_diags[:, ind_freq]
            Ahlu = splu(Ah, permc_spec='NATURAL')
            Uh = Ahlu.solve(Lh)[:, 0]
            residu[:, ind_freq] = self.__compute_residu(Uh, ind_freq)
            if self.is_unwrap:
                residu = (np.unwrap(residu*self.__target_norm())
                          / self.__target_norm())
            if grad_type == 'frechet':
                grad_temp, hess = self.__GradFrechet(residu[:, ind_freq],
                                                     Ahlu, Uh, ind_freq)
                hessian += hess
                gradient += grad_temp
            elif grad_type == 'adjoint':
                gradient += self.__GradAdjoint(residu[:, ind_freq],
                                               Ahlu, Uh, ind_freq)
        cost = self.__compute_cost(residu)
        return cost, gradient, hessian

    def get_cost_grad_hessian(self, params_values=list(), grad_type=None,
                              stepSize=1e-8):
        """
        The cost, gradient and hessian for a set of design variable.

        It computes the cost and the gradient of the problem for the specified
        value of design varaibles. Details are given in [Ernoult_FWI]_.

        The hessian :math:`H` can possibly be estimated by admitting the least
        square approximation:

            .. math ::
                H \\approx J^TJ

        with :math:`J` the jacobian of the residual.

        Parameters
        ----------
        params_values : list of float, optional
            The value of the design variable. The defautl is an empty list,
            for which the design variables keep their current values.
        grad_type : string, optional
            The way to compute the gradient. The default is None: no gradient
            is computed. It is possible to chose between:

                - 'frechet': frechet derivative is applied on the residual, \
                    giving the possibility to estimate the hessian.
                - 'adjoint': adjoint state method is used. It can be faster \
                    but avoid the estimation of the hessian.
                - 'finite diff': the gradient is estimated by finite \
                    difference. It is longer and less precise than other \
                    method. **It is not recomanded except for validation.**
        stepSize: float, optional
            Only use with 'finite diff' gradient: the step size used in the
            finite difference computation. The default is 1e-8.

        Returns
        -------
        cost: float
            The cost value
        gradient: np.array
            The gradient with respect to each design variable. None if
            `grad_type=None`
        hessian: np.array
            The hessian matrix. None except if `grad_type=frechet`

        """
        gradient = None
        hessian = None
        if len(params_values) > 0:
            self.modify_parts(params_values)
        Nderiv = len(self.optim_params.get_active_values())

        cost = 0
        if grad_type == 'frechet' or grad_type == 'adjoint':
            gradient = np.zeros([Nderiv])
            if grad_type == 'frechet':
                hessian = np.zeros([Nderiv, Nderiv])

        for note, target in zip(self.notes, self.imped_targets):
            self._target = self.__observation(target)
            self.set_note(note)
            (cost_note, gradient_note,
             hessian_note) = self.__cost_grad_hessian_1note(grad_type,
                                                            stepSize)
            cost += cost_note
            if grad_type == 'frechet' or grad_type == 'adjoint':
                gradient += gradient_note
                if grad_type == 'frechet':
                    hessian += hessian_note

        if grad_type == 'finite diff':
            gradient = self.__GradFiniteDiff(cost, stepSize=stepSize)
        return cost, gradient, hessian

    def _get_cost_grad_hessian_opt(self, x):
        cost, grad, hess = self.get_cost_grad_hessian(x, grad_type='frechet')
        self._params_evol.append(x)
        self._cost_evol.append(cost)
        return cost, grad, hess

    def _get_cost_grad_opt(self, x):
        cost, grad = self.get_cost_grad_hessian(x, grad_type='adjoint')[0:2]
        self._params_evol.append(x)
        self._cost_evol.append(cost)
        if self._iter_detail:
            print_cost(len(self._cost_evol)-1, cost, grad)
        return cost, grad

    def _update_cost_grad_hessian(self, x):
        if any(np.asarray(x) != self.optim_params.get_active_values()):
            self._cost, self._grad, self._hessian = \
                self.get_cost_grad_hessian(x, grad_type=self._grad_type)
            self._params_evol.append(x)
            self._cost_evol.append(self._cost)
            if self._iter_detail:
                print_cost(len(self._cost_evol)-1, self._cost, self._grad)

    def _get_cost(self, x):
        self._update_cost_grad_hessian(x)
        return self._cost

    def _get_grad(self, x):
        self._update_cost_grad_hessian(x)
        return self._grad

    def _get_hessian(self, x):
        self._update_cost_grad_hessian(x)
        return self._hessian

    def _update_residuals_jacobian(self, x):
        if any(np.asarray(x) != self.optim_params.get_active_values()):
            self._residuals, self._jacobian = self.residuals_jacobian(x)
            self._params_evol.append(x)
            cost = 0.5*self._residuals.dot(self._residuals)
            self._cost_evol.append(cost)
            if self._iter_detail:
                grad = self._jacobian.T.dot(self._residuals[:, np.newaxis])
                print_cost(len(self._cost_evol)-1, cost, grad)

    def _get_residuals(self, x):
        self._update_residuals_jacobian(x)
        return self._residuals

    def _get_jacobian(self, x):
        self._update_residuals_jacobian(x)
        return self._jacobian

# %% Optimization algorithms

    def optimize_freq_model(self, algorithm='default', max_iter=100,
                            minstep_cost=1e-8, tresh_grad=1e-6,
                            iter_detailed=False):
        """
        Solve the inverse problem.

        This method apply an optimization algorithm to minimize the cost
        function by respecting the specified bounds for each active design
        variables.

        Parameters
        ----------
        algorithm : string, optional
            The optimization algorithm chosen to minimize the cost function.\
            The default is "default", in this case the algorithm is chosen \
            automatically following the presence of bounds. It is possible to\
            chose between:

                - 'lm': levenberg-marquardt algorithm by using \
                    :func:`scipy.least_squares`. This is the default algorithm\
                        chosen for unconstrained problem.
                - 'trf': Trust Region Reflective algorithm, from \
                    :func:`scipy.least_squares`. This is the default algorithm\
                        chosen for bounded problem.
                - 'dogbox': dogleg algorithm from :func:`scipy.least_squares`.
                - 'Newton-CG', 'BFGS', 'SLSQP', 'L-BFGS-B' or 'trust-constr' \
                    from  :func:`scipy.minimize`.
                - 'LM', 'steepest', 'QN', 'GN': homemade algorithms
        max_iter : int, optional
            The maximum authorized umber of iterations. The default is 100.
        minstep_cost : float, optional
            The treshold on the relative variation of the cost. The default is\
            1e-8.
        tresh_grad : float, optional
            The treshold on the gradient norm to stop the algorithm. The \
            default is 1e-6
        iter_detailed : booelan
            If true, print information at each function evaluation. The \
            default is false.

        Returns
        -------
        result : :py:class:`scipy.optimize.OptimizeResult`\
        or :py:class:`HomemadeOptimizeResult<openwind.algo_optimization.HomemadeOptimizeResult>`
            A result object with at least following attributes:

                - x : the final value of the design variables
                - cost : the final value of the cost function
                - nit : the number of iteration
                - x_evol: the evolution of the design variable along the \
                    iterations
                - cost_evol : the evolution of the cost along the iterations

        """

        initial_params = self.optim_params.get_active_values()
        self._cost_evol = list()
        self._params_evol = list()
        self._iter_detail = False

        lst_algo = ['trf', 'dogbox', 'lm']
        min_algo = ['Newton-CG', 'BFGS', 'SLSQP', 'L-BFGS-B', 'trust-constr']
        home_algo = ['LM', 'steepest', 'QN', 'GN']
        if algorithm not in lst_algo + min_algo + home_algo + ['default']:
            raise ValueError("Unknown algorithm, choose between:\n {} "
                             "(least_square from scipy)\n {} (minimize from "
                             "scipy)\n {} (home-made)".format(lst_algo,
                                                              min_algo,
                                                              home_algo))

        lb, ub = tuple(zip(*self.optim_params.get_active_bounds()))
        if algorithm == 'default' and all(np.isinf(lb+ub)):
            algorithm = 'lm'
        elif algorithm == 'default':
            algorithm = 'trf'
        elif (not all(np.isinf(lb+ub)) and algorithm not
                in ['SLSQP', 'L-BFGS-B', 'trust-constr', 'trf', 'dogbox']):
            raise ValueError("Algorithm '{}' doesn't support bounds. Please "
                             "use preferentially 'trf'.".format(algorithm))
        if iter_detailed:
            print("Algortihm: '{}'".format(algorithm))
        if algorithm in lst_algo:
            result = self._least_squares_scipy(initial_params, (lb, ub),
                                               algorithm, max_iter,
                                               minstep_cost, tresh_grad,
                                               iter_detailed)
        elif algorithm in min_algo:
            result = self._minimize_scipy(initial_params, (lb, ub), algorithm,
                                          max_iter, minstep_cost, tresh_grad,
                                          iter_detailed)
        elif algorithm in home_algo:
            if algorithm in ['LM', 'GN']:
                result = homemade_minimization(self._get_cost_grad_hessian_opt,
                                               initial_params, max_iter,
                                               minstep_cost, tresh_grad,
                                               iter_detailed, algo=algorithm)
            else:
                result = homemade_minimization(self._get_cost_grad_opt,
                                               initial_params, max_iter,
                                               minstep_cost, tresh_grad,
                                               iter_detailed,
                                               algo=algorithm)

        self.get_cost_grad_hessian(result.x)
        self.solve()
        return result

    def _least_squares_scipy(self, initial_params, bounds,
                             algorithm='trf', max_iter=100,
                             minstep_cost=1e-8, tresh_grad=1e-6,
                             iter_detailed=False):
        cons = self._get_constraints(algorithm)
        (self._residuals,
         self._jacobian) = self.residuals_jacobian(initial_params)
        self._iter_detail = iter_detailed
        self._params_evol.append(initial_params)
        self._cost_evol.append(0.5*self._residuals.dot(self._residuals))
        if self._iter_detail:
            print_cost(len(self._cost_evol)-1, self._cost_evol[-1],
                       self._jacobian.T.dot(self._residuals[:, np.newaxis]))
        res = least_squares(self._get_residuals, initial_params,
                            jac=self._get_jacobian, bounds=bounds, verbose=1,
                            method=algorithm, ftol=minstep_cost,
                            max_nfev=max_iter, gtol=tresh_grad)
        res.x_evol = self._params_evol
        res.cost_evol = self._cost_evol
        res.nit = res.nfev
        return res

    def _minimize_scipy(self, initial_params, bounds, algorithm,
                        max_iter=100, minstep_cost=1e-8, tresh_grad=1e-6,
                        iter_detailed=False):
        if algorithm in ['Newton-CG', 'trust-constr']:
            self._grad_type = 'frechet'
        else:
            self._grad_type = 'adjoint'

        self._cost, self._grad, self._hessian = \
            self.get_cost_grad_hessian(initial_params,
                                       grad_type=self._grad_type)
        self._iter_detail = iter_detailed
        self._params_evol.append(initial_params)
        self._cost_evol.append(self._cost)
        if np.all(np.isinf(bounds)):
            bounds_obj = None
        else:
            bounds_obj = Bounds(bounds[0], bounds[1], keep_feasible=True)
        cons = self._get_constraints(algorithm)
        if self._iter_detail:
            print_cost(0, self._cost, self._grad)

        if algorithm in ['BFGS', 'trust-constr']:
            options = {'disp': True, 'maxiter': max_iter, 'gtol': tresh_grad}
        elif algorithm in ['L-BFGS-B', 'SLSQP']:
            options = {'disp': True, 'maxiter': max_iter, 'ftol': minstep_cost}
        else:
            options = {'disp': True, 'maxiter': max_iter}

        if algorithm in ['BFGS', 'L-BFGS-B', 'SLSQP', 'Newton-CG']:
            res = minimize(self._get_cost_grad_opt, initial_params,
                           method=algorithm, jac=True, bounds=bounds_obj,
                           options=options, constraints=cons)
        elif algorithm in ['trust-constr']:  # 'Newton-CG'
            res = minimize(self._get_cost, initial_params,
                           method=algorithm, jac=self._get_grad,
                           hess=self._get_hessian,
                           bounds=bounds_obj,
                           constraints=cons,
                           options=options)
        res.x_evol = self._params_evol
        res.cost_evol = self._cost_evol
        res.cost = res.fun
        return res

    def _get_constraints(self, algorithm):
        lin_cons = self._get_linear_constraint(algorithm)
        non_lin_cons = self._get_nonlinear_constraint(algorithm)
        return lin_cons + non_lin_cons

    def _get_linear_constraint(self, algorithm):
        A, lb, ub = self.optim_params.get_active_lin_cons()
        cons = list()
        if len(A)>0:
            if algorithm == 'trust-constr':
                cons.append( LinearConstraint(A, lb, ub, keep_feasible=True) )
            elif algorithm == 'SLSQP': # for this algo, the constraints must be scalar and organize in list of dic
                for k in range(len(lb)):
                    if lb[k]==ub[k]:
                        cons.append({'type': 'eq',
                                     'fun': lambda x, p=k:  A[p,:].dot(x) - lb[p],
                                     'jac' : lambda x, p=k: A[p,:]})
                    else:
                        if np.isfinite(lb[k]):
                            cons.append({'type': 'ineq',
                                         'fun': lambda x, p=k:  A[p,:].dot(x) - lb[p],
                                         'jac' : lambda x, p=k: A[p,:]})
                        if np.isfinite(ub[k]):
                            cons.append({'type': 'ineq',
                                         'fun': lambda x, p=k:  -A[p,:].dot(x) + ub[p],
                                         'jac' : lambda x, p=k: -A[p,:]})
            else:
                warnings.warn("Constraints are only available with 'trust-constr' or 'SLSQP' algorithms.")
        return cons

    def _get_nonlinear_constraint(self, algorithm):
        fun, jac, lb, ub = self.optim_params.get_active_nonlin_cons()
        cons = list()
        if len(lb)>0:
            if algorithm == 'trust-constr':
                cons.append( NonlinearConstraint(fun, lb, ub, jac=jac, keep_feasible=False) )
            elif algorithm == 'SLSQP': # for this algo, the constraints must be scalar and organize in list of dic
                for k in range(len(lb)):
                    if lb[k]==ub[k]:
                        cons.append({'type': 'eq',
                                     'fun': lambda x, p=k:  fun(x)[p] - lb[p],
                                     'jac' : lambda x, p=k: jac(x)[p,:]})
                    else:
                        if np.isfinite(lb[k]):
                            cons.append({'type': 'ineq',
                                         'fun': lambda x, p=k:  fun(x)[p] - lb[p],
                                         'jac' : lambda x, p=k: jac(x)[p,:]})
                        if np.isfinite(ub[k]):
                            cons.append({'type': 'ineq',
                                         'fun': lambda x, p=k:  -fun(x)[p] + ub[p],
                                         'jac' : lambda x, p=k: -jac(x)[p,:]})
            else:
                warnings.warn("Constraints are only available with 'trust-constr' or 'SLSQP' algorithms.")
        return cons

# %% Sensitivity

    def __build_window(self, window):
        if not window:
            window_pond = np.ones_like(self.frequencies)
        else:
            f0, Deltaf = window
            Deltaf_Hz = f0 * (2**(Deltaf/1200) - 1)
            window_pond = np.zeros(self.frequencies.shape, dtype=float)
            sin = 0.5 + 0.5*np.cos(np.pi*(self.frequencies - f0)/Deltaf_Hz)
            ind = np.logical_and(self.frequencies >= f0 - Deltaf_Hz,
                                 self.frequencies <= f0 + Deltaf_Hz)
            window_pond[ind] = sin[ind]
        return window_pond

    def __sensitivity_observable_1note(self, note=None, window=None,
                                       interp=False, pipes_label='main_bore',
                                       interp_grid='original'):
        """
        Sensitivity of the observable w.r. to the design parameters for 1 note.

        The sensitivity is here defined as the L2 norm of the gradient of the
        observable. This gradient is estimated by the Frechet Derivative,
        giving the possibility to have the gradients of the acoustics fields.

        Parameters
        ----------
        note : string, optional
            Name of the note idicated in the fingering chart. The default value
            is 'None' (all the holes opened)

        window: tuple, optional
            Parameters to window the observable:
                (central frequency, frequency width in cents)
            If not defined, no window is applied. The default is None.

        interp: logical, optional
            Indicates if the variation of the acoustics fields (pressure and
            flow) wrt tthe design parameters must be interpolated along the
            instrument. The default is 'False'

        pipes_label: string, optional
            The labels of the pipes on which the acoustics fields must be
            interpolated. If it is "main_bore" all the pipes of the main bore
            (all excepted chimney holes) are included. Not used if
            `interp=False`. The default is 'main_bore'

        interp_grid : {float, array(float), 'original'}
            you can give either a list of points on which to interpolate, or a
            float which is the step of the interpolation grid, or if you want
            to keep the GaussLobato grid, put 'original'. Not used if
            `interp=False`. Default is 'original'.


        Returns
        -------
        sensitivity : np.array
            The norm (along the frequency axes) of the gradient for each design
            parameters.

        grad_observation : np.array
            The gradient of the observation at each frequency for each design
            parameters

        grad_flow, grad_pressure : np.array
            The gradient of the acoustics fields at each point of the \
            interpolation grid, for each frequency and for each design \
            parameters

        """
        self.set_note(note)
        Nderiv = len(self.optim_params.get_active_values())
        Nfreq = len(self.frequencies)

        # redim_pressure = self.scaling.get_scaling_pressure()
        # redim_flow = self.scaling.get_scaling_flow()
        convention = self.source_ref.get_convention()

        grad_observation = np.zeros((Nfreq, Nderiv), dtype='complex')
        observation = np.zeros(Nfreq, dtype='complex')
        # Ah = self.Ah_nodiag.tocsc()
        Ah, ind_diag = self._initialize_Ah_diag()
        Lh = self.Lh.toarray()
        if interp:
            interpolation = FrequentialInterpolation(self, pipes_label,
                                                     interp_grid)
            self.x_interp = interpolation.x_interp
            Nx = len(self.x_interp)
            grad_flow = np.zeros((Nx, Nfreq, Nderiv), dtype='complex')
            grad_pressure = np.zeros((Nx, Nfreq, Nderiv), dtype='complex')
        else:
            grad_flow = np.array([])
            grad_pressure = np.array([])

        window_pond = self.__build_window(window)
        self.__computedAH()

        for ind_freq in range(Nfreq):
            # Ah.setdiag(self.Ah_diags[:, ind_freq])
            Ah.data[ind_diag] = self.Ah_diags[:, ind_freq]
            Ahlu = splu(Ah, permc_spec='NATURAL')
            Uh = Ahlu.solve(Lh)[:, 0]

            impedance = self.__impedance_scaled(Uh)
            observation[ind_freq] = self.__observation(impedance)

            diff_obs, diff_conj_obs = self.__diff_observation_wrU(Uh)
            diff_real_obs = 0.5*(diff_obs + diff_conj_obs)
            diff_imag_obs = -0.5j*(diff_obs - diff_conj_obs)
            dobservation_dU = vstack([diff_real_obs, 1j*diff_imag_obs])

            # sourceAdj = dobservation_dU.conj().T.dot(np.array([1., 1.]))
            # lambdaAdjconj = Ahlu.solve(-1*sourceAdj.conjugate(), 'T')
            grad = np.zeros([Nderiv], dtype='complex')
            for diff_index in range(Nderiv):
                dAhU = self.__compute_dAhU(Uh, diff_index, ind_freq)
                # grad[diff_index] = (lambdaAdjconj @ dAhU)
                dU = -1 * Ahlu.solve(dAhU)
                jacob = dobservation_dU.dot(dU).T
                grad[diff_index] = jacob.dot(np.array([1., 1.]))
                if interp:
                    diff_H1 = interpolation.interpolate_H1(dU)
                    diff_interp_H1 = (interpolation.
                                      diff_interpolate_H1(Uh, diff_index))
                    H1 = interpolation.interpolate_H1(Uh)
                    grad_H1 = (diff_H1 + diff_interp_H1)/H1

                    diff_L2 = interpolation.interpolate_L2(dU)
                    diff_interp_L2 = (interpolation.
                                      diff_interpolate_L2(Uh, diff_index))
                    L2 = interpolation.interpolate_L2(Uh)
                    grad_L2 = (diff_L2 + diff_interp_L2)/L2
                    if convention == 'PH1':
                        grad_pressure[:, ind_freq, diff_index] = grad_H1
                        grad_flow[:, ind_freq, diff_index] = grad_L2
                    elif convention == 'VH1':
                        grad_pressure[:, ind_freq, diff_index] = grad_L2
                        grad_flow[:, ind_freq, diff_index] = grad_H1
            grad_observation[ind_freq, :] = grad  # /observation

        gradient_window = grad_observation*window_pond[:, np.newaxis]
        sensitivity = np.sqrt(np.sum(gradient_window.real**2
                                     + gradient_window.imag**2, axis=0))

        sensitivity /= np.linalg.norm(observation)
        return sensitivity, grad_observation, grad_flow, grad_pressure

    def compute_sensitivity_observable(self, windows=None, interp=False,
                                       pipes_label='main_bore',
                                       interp_grid='original'):
        """
        Sensitivity of the observable w.r. to the design parameters.

        The sensitivity of each fingering :math:`n` w.r. to each parameter
        :math:`m_i` is here defined as the L2 norm of the gradient of the
        observable :math:`\\mathcal{O}(\\omega)` normalized by the norm of the
        observable along the frequency axis:

        .. math::
            \\sigma_i = \\frac{|| \\frac{\\partial \\mathcal{O}(\\omega)} \
            {\\partial m_i}||_{L_2} }{||\\mathcal{O}(\\omega)||_{L_2}}


        This gradient is estimated by the Frechet Derivative,
        giving the possibility to have the gradients of the acoustics fields.


        Parameters
        ----------
        note : string, optional
            Name of the note idicated in the fingering chart. The default value
            is 'None' (all the holes opened)

        window: tuple, optional
            Parameters to window the observable: (central frequency, frequency\
            width in cents). If not defined, no window is applied. The default\
            is None.

        interp: logical, optional
            Indicates if the variation of the acoustics fields (pressure and
            flow) wrt tthe design parameters must be interpolated along the
            instrument. The default is 'False'

        pipes_label: string, optional
            The labels of the pipes on which the acoustics fields must be
            interpolated. If it is "main_bore" all the pipes of the main bore
            (all excepted chimney holes) are included. Not used if
            `interp=False`. The default is 'main_bore'

        interp_grid : {float, array(float), 'original'}
            you can give either a list of points on which to interpolate, or a
            float which is the step of the interpolation grid, or if you want
            to keep the GaussLobato grid, put 'original'. Not used if
            `interp=False`. Default is 'original'.


        Returns
        -------
        sensitivity : np.array
            The norm (along the frequency axes) of the gradient for each design
            parameters and each note.

        grad_observation : np.array
            The gradient of the observation at each frequency for each design
            parameters

        """
        sensitivities = list()
        grad_observation = list()
        grad_flow, grad_pressure = (list(), list())
        if not windows:
            windows = [None]*len(self.notes)
        else:
            assert len(windows) == len(self.notes)

        for note, window in zip(self.notes, windows):
            sensi_note, grad_note, grad_flow_note, grad_pressure_note = \
                self.__sensitivity_observable_1note(note, window, interp,
                                                    pipes_label, interp_grid)
            sensitivities.append(sensi_note)
            grad_observation.append(grad_note)
            grad_flow.append(grad_flow_note)
            grad_pressure.append(grad_pressure_note)
        self.grad_flow = grad_flow
        self.grad_pressure = grad_pressure
        self.sensitivities = np.array(sensitivities)
        return np.array(sensitivities), grad_observation

# %% Plots
    def plot_observation(self, note=None, dbscale=True, figure=None,
                         Z_target=None, label=None, **kwargs):
        """
        Plot the observable.

        Parameters
        ----------
        note : string, optional
            The name of the note which must be plot. If None, the last one used
            is plotted. The default is None.
        dbscale : boolean, optional
            If true, plot complex observable with dBscale. The default is True.
        figure : matplotlib.Figure, optional
            The figure on which plot the observable. The default is None.
        Z_target : Array, optional
            The target impedance use to compute the target observable. If None,
            no target is plotted. The default is None.
        label : string, optional
            The label of the plot. The default is None.
        **kwargs : keywords
            Keywords given to plot.
        """

        if not figure:
            fig = plt.figure()
        else:
            fig = figure
        ax = fig.get_axes()

        if note is not None:
            self.set_note(note)
        self.solve()

        obs = self.__observation(self.imped/self.get_ZC_adim())
        if self.is_unwrap:
            obs = np.unwrap(obs)

        if Z_target is not None:
            obs_targ = self.__observation(Z_target)
            if self.is_unwrap:
                obs_targ = np.unwrap(obs_targ)

        if len(ax) < 2 and any(np.imag(obs) != 0):
            ax = [fig.add_subplot(2, 1, 1)]
            ax.append(fig.add_subplot(2, 1, 2, sharex=ax[0]))
            ax[0].grid()
            ax[1].grid()
        elif len(ax) < 1:
            ax = [fig.add_subplot(1, 1, 1)]
            ax[0].grid()

        if any(np.imag(obs) != 0):  # Plot of complex observable
            ylabel = '$||\\phi(Z/Zc)||$'
            if dbscale:
                ax[0].plot(self.frequencies, 20*np.log10(np.abs(obs)),
                           label=label, **kwargs)
                if Z_target is not None:
                    ax[0].plot(self.frequencies, 20*np.log10(np.abs(obs_targ)),
                               label='target', color='k', linewidth=.5)
                ylabel += ' (dB)'

            else:
                ax[0].plot(self.frequencies, np.abs(obs), label=label,
                           **kwargs)
                if Z_target is not None:
                    ax[0].plot(self.frequencies, np.abs(obs_targ),
                               label='target', color='k', linewidth=.5)
            ax[0].set_ylabel(ylabel)

            if len(ax) > 1:
                ax[1].plot(self.frequencies, np.angle(obs), **kwargs)
                if Z_target is not None:
                    ax[1].plot(self.frequencies, np.angle(obs_targ),
                               label='target', color='k', linewidth=.5)
                ax[1].set_xlabel("Frequency (Hz)")
                ax[1].set_ylabel("angle($\\phi(Z)$)")

        else:  # Plot of real observable
            ax[0].plot(self.frequencies, obs, label=label, **kwargs)
            if Z_target is not None:
                ax[0].plot(self.frequencies, obs_targ,
                           label='target', color='k', linewidth=.5)
            ax[0].set_ylabel('$\\phi(Z/Zc)$')
            ax[0].set_xlabel("Frequency (Hz)")

        if label:
            ax[0].legend(loc='upper right')

    def plot_sensitivities(self, logscale=False, scaling=False, relative=True,
                           text_on_map=True, param_order=None, **kwargs):
        """
        Plot a sensitivity map.

        It illustrates how much each fingering is sensitive to each active
        design parameters. The sensitivities are computed with
        :func:`compute_sensitivity_observable`.

        Parameters
        ----------
        logscale : boolean, optional
            Does the color follow a logarithmic scale. The default is False.
        scaling : boolean, optional
            If true each line is scaled (the sum of the sensitivity for one
            given parameter equals 1). The default is False.
        relative : boolean, optional
            The sensitivity is normalized by the absolute value of the design
            parameter. The default is True.
        text_on_map : boolean, optional
            Print the sensitivity value on each case. The default is True.
        param_order : list of int, optional
            Give the possibility to modify the order of the lines. The default
            is None.
        **kwargs :
            Other plotting keywords.

        Returns
        -------
        fig_sens : matplotlib.figure.Figure
            The figure.
        ax_sens : matplotlib.axes._subplots.AxesSubplot
            The axes object.
        """

        labels = np.array(self.optim_params.labels)[self.optim_params.
                                                    active].tolist()

        if relative:
            Z = self.sensitivities * np.array(self.optim_params.
                                              get_active_values())
            z_legend = 'Relative Sensitivity'
        else:
            Z = self.sensitivities
            z_legend = 'Sensitivity'

        if param_order:
            Z = Z[:, param_order]
            labels = np.array(labels)[param_order].tolist()

        if scaling:
            Z = Z / np.sum(Z, 1)[:, np.newaxis]

        if logscale:
            Z_plot = np.log10(np.abs(Z.T))
            z_legend += ' (log)'
        else:
            Z_plot = np.abs(Z.T)

        fig_sens, ax_sens = plt.subplots()
        im, cbar = heatmap(Z_plot, labels, self.notes, ax=ax_sens,
                           cbarlabel=z_legend, **kwargs)
        if text_on_map:
            cbar.remove()
            annotate_heatmap(im, data=np.abs(Z.T)*100,
                             threshold=Z_plot.max()/2)
        return fig_sens, ax_sens

    def plot_grad_acoustics_field(self, notes=None, dbscale=True,
                                  var='pressure'):
        """
        Plot either the pressure or the flow gradient w.r. to the design
        parameters for every frequency inside the entire instrument.

        Parameters
        ----------
        notes : list of string, optional
            The considered notes. The default is None (unchanged).
        dbscale : boolean, optional
            use of dB scale or not. The default is True.
        var : string, optional
            The ploted acoustic variable: 'pressure' or 'flow'. The default is
            'pressure'.

        """

        try:
            import plotly.graph_objs as go
            import plotly.offline as py
        except ImportError:  # ImportError as err:
            msg = "The 3D plot are nicer with the module plotly."
            # raise ImportError(msg) from err
            print(msg)

        X = self.x_interp
        Y = self.frequencies

        optim_labels = (np.array(self.optim_params.labels)
                        [self.optim_params.active].tolist())
        if not notes:
            notes = self.notes
        for note in notes:
            ind_note = self.notes.index(note)
            for diff_index, param_label in enumerate(optim_labels):
                if var == 'pressure':
                    Z = self.grad_pressure[ind_note][:, :, diff_index].T
                    filename = 'grad_pressure_{}_{}.html'.format(note,
                                                                 param_label)
                elif var == 'flow':
                    Z = self.grad_flow[ind_note][:, :, diff_index].T
                    filename = 'grad_flow_{}_{}.html'.format(note, param_label)
                else:
                    raise ValueError("possible values are pressure or flow")
                if dbscale:
                    Zplot = 20*np.log10(np.abs(Z))
                else:
                    Zplot = np.real(Z)
                try:
                    xaxis = dict(title='Position', autorange='reversed')
                    yaxis = dict(title='Frequency', autorange='reversed')
                    zaxis = dict(title='Relative variation')
                    layout_3D = go.Layout(scene=dict(xaxis=xaxis, yaxis=yaxis,
                                                     zaxis=zaxis))

                    x_go = go.surface.contours.X(highlightcolor="#42f462",
                                                 project=dict(x=True))
                    y_go = go.surface.contours.Y(highlightcolor="#42f462",
                                                 project=dict(y=True))
                    contours = go.surface.Contours(x=x_go, y=y_go)
                    data_u3D = [go.Surface(x=X, y=Y, z=Zplot,
                                           contours=contours)]
                    fig_u3D = go.Figure(data=data_u3D, layout=layout_3D)
                    py.plot(fig_u3D, filename=filename)
                except:
                    fig = plt.figure()
                    ax = fig.gca(projection='3d')
                    Xplot, Yplot = np.meshgrid(X, Y)
                    surf = ax.plot_surface(Xplot, Yplot, Zplot, cmap=cm.plasma,
                                           antialiased=True)
                    ax.set_xlabel('Position')
                    ax.set_ylabel('Frequency')
                    ax.set_zlabel('Relative variation')
                    fig.colorbar(surf, shrink=0.5, aspect=5)
                    fig.suptitle('{}\n{}'.format(note, param_label))
