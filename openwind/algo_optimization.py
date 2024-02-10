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


class HomemadeOptimizeResult():
    """
    Result of homemade optimization algorithm.

    It is inspired from `scipy.optimize.OptimizeResult <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.OptimizeResult.html>`_

    Attributes
    ----------
    x : array
        The solution of the optimization.
    cost : float
        Value of the cost function at the solution.
    nit : int
        Number of iterations performed by the optimizer.
    x_evol : list(array)
        Evolution of the design parameters during the optimization.
    cost_evol : list(float)
        Evolution of the cost during the optimization..
    grad : array
        Gradient of the cost function at the solution..
    status : int
            The reason for algorithm termination.
    message : str
        Verbal description of the termination reason.

    """

    def __init__(self, params_evol, cost_evol, grad, status, message):
        self.x = params_evol[-1]
        self.cost = cost_evol[-1]
        self.nit = len(cost_evol) - 1
        self.x_evol = params_evol
        self.cost_evol = cost_evol
        self.grad = grad
        self.optimality = np.linalg.norm(grad)
        self.status = status
        self.message = message
        self.succes = (status > 0)


def stop_message(iteropt, maxiter, cost, step_cost, ftol, gradient,
                 gtol):
    if np.max(np.abs(gradient)) <= gtol:
        message = ('Algorithm stops: the gradient is below the tolerance'
                   ' treshold ({:.2e})'.format(gtol))
        status = 1
    elif step_cost <= ftol:
        message = ('Algorithm stops: the cost variation is below the tolerance'
                   ' treshold ({:.2e})'.format(ftol))
        status = 2
    elif iteropt >= maxiter:
        message = ('Algorithm stops: the maximum iteration number has been '
                   'reached ({:d})'.format(maxiter))
        status = 0
    else:
        message = 'Algorithm stops for internal specific reason.'
        status = -1

    message += ('\n\tIterations:{:d} \n\tFinal cost = {:.2e} \n\tNorm of '
                'gradient = {:.2e}').format(iteropt, cost,
                                            np.linalg.norm(gradient))
    print(message)
    return status, message


def print_cost(index_iter, cost, gradient, info=('', '')):
    norm_grad = np.linalg.norm(gradient)
    if index_iter % 20 == 0:
        print('{}\t{:<14}\t{:<14}\t{:<14}'.format('Iteration', 'Cost',
                                                  'Gradient', info[0]))
    print('{:<10d}\t{:<14.8e}\t{:<14.8e}\t{}'.format(index_iter, cost,
                                                     norm_grad, info[1]))


def hessianFiniteDiff(get_cost_grad, params_values, stepSize=1e-8):
    """
    Hessian approximation by finite difference of the gradient.
    """
    Nderiv = len(params_values)
    hessFor = np.zeros((Nderiv, Nderiv))
    hessBack = np.zeros((Nderiv, Nderiv))

    _, grad_init = get_cost_grad(params_values)
    params_init = np.array(params_values, copy=True)
    params = np.array(params_values, copy=True)
    for diff_index in range(Nderiv):
        params[diff_index] = params_init[diff_index] + stepSize
        gradFor = get_cost_grad(params)[1]
        hessFor[diff_index, :] = (gradFor - grad_init) / stepSize

        params[diff_index] = params_init[diff_index] - stepSize
        gradBack = get_cost_grad(params)[1]
        hessBack[diff_index, :] = (grad_init - gradBack) / stepSize

        params[diff_index] = params_init[diff_index]
    get_cost_grad(params_init)
    return (hessFor + hessBack) / 2


def hessianBFGS(x0, x1, grad, Bk):
    """
    BFGS approximation of the inverse of the hessian.

    From [Noc.6.19]_, chap.6, eq.(6.19)

    References
    ----------
    .. [Noc.6.19] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.

    """
    # equation 6.19 du Nocedal
    sk = x1 - x0
    yk = grad[1] - grad[0]
    sk = sk[:, np.newaxis]
    yk = yk[:, np.newaxis]
    num1 = Bk.dot(sk.dot(sk.T.dot(Bk.T)))
    den1 = sk.T.dot(Bk.dot(sk))
    num2 = yk.dot(yk.T)
    den2 = yk.T.dot(sk)

    Bnew = Bk - num1/den1 + num2/den2
    return Bnew


def inv__hessianBFGS(x0, x1, grad, Hk):
    """
    BFGS approximation of the inverse of the hessian.

    From [Noc.6.17]_, chap.6, eq.(6.17)

    References
    ----------
    .. [Noc.6.17] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.

    """
    sk = x1 - x0
    yk = grad[1] - grad[0]
    sk = sk[:, np.newaxis]
    yk = yk[:, np.newaxis]
    rho = 1/(yk.T.dot(sk))
    A = np.eye(Hk.shape[0]) - rho * (sk.dot(yk.T))
    Hnew = A.dot(Hk.dot(A)) + rho * (sk.dot(sk.T))
    return Hnew


def backtracking(get_cost_grad, params_old, direction, cost_old, phi_prime):
    """
    Backtracking algorithm.

    From [Noc.3.2]_ chap.3.2, algorithm 3.1.

    References
    ----------
    .. [Noc.3.2] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.

    """
    alpha_0 = 1
    rho = 0.75
    c1 = 1e-3

    alpha = alpha_0
    kadapt = 0
    delta_f = c1 * phi_prime
    params_new = params_old + alpha * direction
    cost_new, _ = get_cost_grad(params_new)
    while cost_new > cost_old + alpha * delta_f and kadapt < 100:
        kadapt = kadapt + 1
        alpha = alpha * rho
        params_new = params_old + alpha * direction
        cost_new, _ = get_cost_grad(params_new)
    if kadapt >= 100:
        print('The backtracking process failed to find the step '
              'length in the maximal authorized iterations (100)')
    return params_new


def linesearch(get_cost_grad, params_old, direction, phi, phi_prime):
    """
    Linesearch algorithm.

    From [Noc.3.5]_ , cha. 3.5 algotithm 3.5.

    References
    ----------
    .. [Noc.3.5] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.


    """
    c1 = 1e-4  # 1e-3
    c2 = 0.9
    alpha_def = 1  # default value for QuasiNewton

    alpha_k = 0
    alpha_k = np.append(alpha_k, alpha_def)
    alpha_max = 10
    kadapt = 1
    alphaStar = []

    while not alphaStar:
        params_new = params_old + alpha_k[kadapt] * direction
        cost_new, grad_new = get_cost_grad(params_new)

        phi = np.append(phi, cost_new)
        phi_prime = np.append(phi_prime, grad_new @ direction)
        if (phi[kadapt] > phi[0] + c1*alpha_k[kadapt]*phi_prime[0] or
           (phi[kadapt] >= phi[kadapt-1] and kadapt > 1)):
            alphaStar = zoomLinesearch(get_cost_grad, kadapt-1, kadapt,
                                       alpha_k, phi, phi_prime, direction,
                                       params_old, c1, c2)
        elif np.abs(phi_prime[kadapt]) <= -c2*phi_prime[0]:
            alphaStar = alpha_k[kadapt]
        elif phi_prime[kadapt] >= 0:
            alphaStar = zoomLinesearch(get_cost_grad, kadapt, kadapt-1,
                                       alpha_k, phi, phi_prime, direction,
                                       params_old, c1, c2)
        else:
            alpha_k = np.append(alpha_k, 0.5*(alpha_k[kadapt]+alpha_max))
            kadapt = kadapt + 1

        if kadapt >= 100:
            print('The linesearch process failed to find the step '
                  'length in the maximal authorized iterations (100)')

    return params_old + alphaStar * direction


def zoomLinesearch(get_cost_grad, k_lo, k_hi, alpha_k, phi, phi_prime,
                   direction, params_optim, c1, c2):
    """
    From [Noc.3.6]_ , cha. 3.5 algotithm 3.6.

    References
    ----------
    .. [Noc.3.6] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.

    """
    a_lo = alpha_k[k_lo]
    a_hi = alpha_k[k_hi]
    phi_lo = phi[k_lo]
    phi_hi = phi[k_hi]
    phi_prime_lo = phi_prime[k_lo]
    alphaStar = []
    niter = 0
    while not alphaStar and niter < 100 and a_lo != a_hi:
        niter = niter+1
        # quadratic approach
        A = (phi_hi - phi_lo - phi_prime_lo*(a_hi - a_lo)) / (a_hi - a_lo)**2
        a_j = a_lo - phi_prime_lo/(2*A)
        # limited range
        a_sorted = sorted([.1*a_lo + .9*a_hi, .9*a_lo + .1*a_hi])
        a_j = min(max([a_sorted[0], a_j]), a_sorted[1])
        # a_j = 0.5*(a_lo + a_hi) # simply mean
        params_j = params_optim + a_j * direction
        cost_j, grad_j = get_cost_grad(params_j)

        if cost_j > phi[0] + c1*a_j*phi_prime[0] or (cost_j >= phi_lo):
            a_hi = a_j
            phi_hi = cost_j
        else:
            if np.abs(grad_j @ direction) <= -c2*phi_prime[0]:
                alphaStar = a_j
            elif (grad_j @ direction)*(a_hi - a_lo) >= 0:
                a_hi = a_lo
                phi_hi = phi_lo
            a_lo = a_j
            phi_lo = cost_j
            phi_prime_lo = grad_j @ direction
    if not alphaStar:
        alphaStar = a_j
    return alphaStar


def search_step_length(get_cost_grad, params_old, direction, cost_old,
                       gradient_old, steptype='linesearch'):
    """
    Apply the direction and search the step-length by performing backtracking
    or "linesearch" method.
    """
    phi_prime = gradient_old @ direction
    if steptype == 'backtracking':
        newparams = backtracking(get_cost_grad, params_old, direction,
                                 cost_old, phi_prime)
    else:
        newparams = linesearch(get_cost_grad, params_old, direction, cost_old,
                               phi_prime)
    return newparams


# %% Algorithms
def linesearch_algorithm(get_cost_grad_direction, get_cost_grad,
                         x0, maxiter=100, ftol=1e-8, gtol=1e-10, disp=False,
                         steptype='linesearch', BFGShessian=False):
    """
    Linesearch algorithm. Must not be used directly.

    The principle is explained in [Noc.3]_, Chap.3.

    Parameters
    ----------
    get_cost_grad_direction : callable
        Function returning cost, gradient and direction, and hessian if BFSG
        approximation is used.
    get_cost_grad : callable
        Function returning cost and gradient.
    x0 : np.array
        Initial values of x.
    maxiter : int, optional
        The maximal number of iterations. The default is 100.
    ftol : float, optional
        The stoping criterium on the relative variation of the cost function.
        The default is 1e-8.
    gtol : float, optional
        The stoping criteirum on the gradient value. The default is 1e-10.
    disp : booelan, optional
        Display informations at each iteration. The default is False.
    steptype : string, optional
        The steptype ('backtracking' ou 'linesearch') use on linesearch
        alogorithms. The default is 'linesearch'.
    BFGShessian : boolean, optional
        Use BFGS approximation for the hessian. The default is False.

    Returns
    -------
    result: HomemadeOptimizeResult
        The result of the optimization.

    References
    ----------
    .. [Noc.3] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.

    """

    step_cost = np.inf
    iteropt = 0

    if BFGShessian:
        cost, gradient, direction, Hk = get_cost_grad_direction(x0,
                                                                0., 0., 0., 0.)
    else:
        cost, gradient, direction = get_cost_grad_direction(x0)

    params_evol = [np.array(x0)]
    cost_evol = [cost]
    if disp:
        print_cost(iteropt, cost, gradient)
    while (iteropt < maxiter and step_cost > ftol
           and np.linalg.norm(gradient) > gtol):
        iteropt = iteropt + 1
        newparams = search_step_length(get_cost_grad, params_evol[iteropt-1],
                                       direction, cost, gradient,
                                       steptype=steptype)
        if BFGShessian:
            (cost, gradient, direction,
             Hk) = get_cost_grad_direction(newparams, params_evol[-1],
                                           gradient, iteropt, Hk)
        else:
            cost, gradient, direction = get_cost_grad_direction(newparams)

        params_evol.append(newparams)
        cost_evol.append(cost)
        if disp:
            print_cost(iteropt, cost, gradient)
        step_cost = (cost_evol[-2] - cost_evol[-1])/(cost_evol[-1] + ftol)

    status, message = stop_message(iteropt, maxiter, cost, step_cost,
                                   ftol, gradient, gtol)
    result = HomemadeOptimizeResult(params_evol, cost_evol, gradient, status,
                                    message)
    return result


def QuasiNewtonBFGS(get_cost_grad, x0, maxiter=100, ftol=1e-8, gtol=1e-10,
                    disp=False, steptype='linesearch'):
    """
    Quasi-Newton algorithm using BFGS hessian approximation.

    The principle is explained in [Noc.3.BFGS]_, Chap.3.

    Parameters
    ----------
    get_cost_grad : callable
        Function returning cost and gradient.
    x0 : np.array
        Initial values of x.
    maxiter : int, optional
        The maximal number of iterations. The default is 100.
    ftol : float, optional
        The stoping criterium on the relative variation of the cost function.
        The default is 1e-8.
    gtol : float, optional
        The stoping criteirum on the gradient value. The default is 1e-10.
    disp : booelan, optional
        Display informations at each iteration. The default is False.
    steptype : string, optional
        The steptype ('backtracking' ou 'linesearch') use on linesearch
        alogorithms. The default is 'linesearch'.

    Returns
    -------
    result: HomemadeOptimizeResult
        The result of the optimization.

    References
    ----------
    .. [Noc.3.BFGS] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.
    """

    def get_cost_grad_direction(x, x0, old_grad, iteropt, old_Hk):
        cost, gradient = get_cost_grad(x)
        if np.mod(iteropt, 10) == 0:
            hessian = hessianFiniteDiff(get_cost_grad, x)
            try:
                Hk = np.linalg.inv(hessian)
            except:
                Hk = np.eye(len(hessian))
        else:
            Hk = inv__hessianBFGS(x0, x, [old_grad, gradient], old_Hk)
        direction = -1 * Hk.dot(gradient)
        if (gradient @ direction) >= 0:
            direction = -1*gradient
        return cost, gradient, direction, Hk

    return linesearch_algorithm(get_cost_grad_direction, get_cost_grad,
                                x0, maxiter, ftol, gtol, disp, steptype,
                                BFGShessian=True)


def Steepest(get_cost_grad, x0, maxiter=100, ftol=1e-8,
             gtol=1e-10, disp=False, steptype='linesearch'):
    """
    Steepest descend algorithm.

    The principle is explained in [Noc.3.s]_, Chap.3.

    Parameters
    ----------
    get_cost_grad : callable
        Function returning cost and gradient.
    x0 : np.array
        Initial values of x.
    maxiter : int, optional
        The maximal number of iterations. The default is 100.
    ftol : float, optional
        The stoping criterium on the relative variation of the cost function.
        The default is 1e-8.
    gtol : float, optional
        The stoping criteirum on the gradient value. The default is 1e-10.
    disp : booelan, optional
        Display informations at each iteration. The default is False.
    steptype : string, optional
        The steptype ('backtracking' ou 'linesearch') use on linesearch
        alogorithms. The default is 'linesearch'.

    Returns
    -------
    result: HomemadeOptimizeResult
        The result of the optimization.

    References
    ----------
    .. [Noc.3.s] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.
    """

    def get_cost_grad_direction(x):
        cost, gradient = get_cost_grad(x)
        direction = -1 * gradient
        return cost, gradient, direction

    return linesearch_algorithm(get_cost_grad_direction, get_cost_grad,
                                x0, maxiter, ftol, gtol, disp, steptype)


def GaussNewton(get_cost_grad_hessian, x0, maxiter=100,
                ftol=1e-8, gtol=1e-10, disp=False):
    """
    Gauss-Newton Algorithm

    It follows the explanation given by [Noc.10]_ (chap.10.3). The used linesearch
    algorithm is :func:`linesearch`.

    Parameters
    ----------
    get_cost_grad_hessian : callable
        A method which return the cost, the gradient and the estimation of the\
        hessian.
    x0 : np.array
        Initial values of x.
    maxiter : int, optional
        The maximal number of iterations. The default is 100.
    ftol : float, optional
        The stoping criterium on the relative variation of the cost function.
        The default is 1e-8.
    gtol : float, optional
        The stoping criteirum on the gradient value. The default is 1e-10.
    disp : booelan, optional
        Display informations at each iteration. The default is False.

    Returns
    -------
    result: HomemadeOptimizeResult
        The result of the optimization.

    References
    ----------
    .. [Noc.10] Nocedal, Jorge, and Stephen J. Wright. 2006. Numerical \
        Optimization. 2nd ed. Springer Series in Operations Research. \
        New York: Springer.


    """

    def get_cost_grad(x):
        cost, gradient, _ = get_cost_grad_hessian(x)
        return cost, gradient

    def get_cost_grad_direction(x):
        cost, gradient, hessian = get_cost_grad_hessian(x)
        direction = np.linalg.solve(hessian, -1 * gradient)
        if (gradient @ direction) >= 0:
            direction = -1*gradient
        return cost, gradient, direction

    return linesearch_algorithm(get_cost_grad_direction, get_cost_grad,
                                x0, maxiter, ftol, gtol, disp)


def LevenbergMarquardt(get_cost_grad_hessian, x0, maxiter=100,
                       ftol=1e-8, gtol=1e-10,
                       disp=False, method='3'):
    """
    Levenberg Marquardt algorithm.

    This algorithm can be used only with least-square problems for which the
    cost function is writting as:

        .. math::
            F = \\frac{1}{2} ||\\mathbf{r}||_{L2}

    where :math:`\\mathbf{r}` is the residual. The three implementation
    presented here are inspired from the one proposed by H.Gavin [Gavin]_,
    [Madsen]_



    Parameters
    ----------
    get_cost_grad_hessian : callable
        A method which return the cost, the gradient and the estimation of the\
        hessian.
    x0 : np.array()
        The initial values.
    maxiter : int, optional
        The maximum iterations authorized. The default is 100.
    ftol : float, optional
        the minimal step cost authorized. The default is 1e-8.
    gtol : float, optional
        The minimal authorized value for gradient. The default is 1e-10.
    disp : boolean, optional
        Display or not the detail of each iterations. The default is False.
    method : string, optional
        The method used between {'1', '2', '3'}, see [Gavin]_. The default is '3'.
            1. the historical LM method
            2. Quadratic approximation
            3. Method proposed by [Madsen]_

    Returns
    -------
    result: HomemadeOptimizeResult
        The result of the optimization.

    References
    ----------
    .. [Gavin] http://people.duke.edu/~hpgavin/m-files/lm.pdf

    .. [Madsen] K.Madsen, H. B. Nielsen, and O. Tingleff, Methods For Non-Linear\
    Least Squares Probems, Informatics and Mathematical Modelling, Technical \
    University of Denmark. 2004.


    """

    lambda0 = 1e-2  # 1e-8  # 1e-5 #
    eps_4 = 1e-1  # 1e-1 # 0 #
    nui0 = 2
    lambdaMin = 1e-7  # ??1e-10  #
    lambdamax = 1e7  # 1e10 #
    iter_lambda_max = 100

    Lup = 11
    Ldown = 9

    # print('WARNING ARTIFICIAL MIN AND MAX')
    # p_min = -10*np.abs(x0)
    # p_max =  10*np.abs(x0)

    if method == '1':
        # cf document by H.P. Gavin method 1: method from LM
        def one_step_LM(hessian, lambdai, minusgrad):
            diag_hess = np.diag(hessian)
            step_i = np.linalg.solve(hessian + lambdai*np.diag(diag_hess),
                                     minusgrad)  # eq(13)
            params_test = params_evol[-1] + step_i
            # params_test = np.array([min(max(pi, p_min[k]), p_max[k]) for k, pi in enumerate(params_test)])
            cost, gradient, hessian = get_cost_grad_hessian(params_test)
            DeltaCost = cost_evol[-1] - cost
            expected = 0.5*(step_i @ (lambdai*diag_hess*step_i + minusgrad))
            rho = DeltaCost / min(cost_evol[-1], expected)  # eq(16)
            if rho > eps_4:
                lambdai = np.maximum(lambdai/Ldown, lambdaMin)
            else:
                lambdai = np.minimum(lambdai*Lup, lambdamax)
            return params_test, cost, gradient, hessian, rho, lambdai

    elif method == '2':
        # cf document by H.P. Gavin method 2: quadratic Method

        def one_step_LM(hessian, lambdai, minusgrad):
            direction = np.linalg.solve(hessian + lambdai*np.identity(Nparam),
                                        minusgrad)  # eq(12)
            params_test = params_evol[-1] + direction
            cost, gradient, hessian = get_cost_grad_hessian(params_test)
            alpha = minusgrad @ direction / ((cost - cost_evol[-1])/2
                                             + 2*minusgrad @ direction)
            if alpha > 0:
                step_i = alpha * direction
                params_test = params_evol[-1] + step_i
                cost, gradient, hessian = get_cost_grad_hessian(params_test)
            else:
                alpha = 1
                step_i = direction
            DeltaCost = cost_evol[-1] - cost
            expected = 0.5 * step_i @ (lambdai*step_i + minusgrad)
            rho = DeltaCost / min(cost_evol[-1], expected)  # eq(15)
            if rho > eps_4:
                lambdai = np.maximum(lambdai/(1 + alpha), lambdaMin)
            else:
                lambdai = lambdai + np.abs(DeltaCost)/(2*alpha)
            return params_test, cost, gradient, hessian, rho, lambdai

    elif method == '3':
        # cf document by H.P. Gavin method 3: from Nielsen 1999
        def one_step_LM(hessian, lambdai, minusgrad):
            step_i = np.linalg.solve(hessian + lambdai*np.identity(Nparam),
                                     minusgrad)  # eq(12)
            params_test = params_evol[-1] + step_i

            cost, gradient, hessian = get_cost_grad_hessian(params_test)

            DeltaCost = cost_evol[-1] - cost
            expected = 0.5 * step_i @ (lambdai*step_i + minusgrad)
            rho = DeltaCost / min(cost_evol[-1], expected)  # eq(15)
            if rho > eps_4:
                lambdai *= np.maximum(1/3, 1 - (2 * rho - 1)**3)
            else:
                lambdai *= nui
            return params_test, cost, gradient, hessian, rho, lambdai

    Nparam = len(x0)
    step_cost = np.inf
    rho = np.inf
    iteropt = 0
    iter_lambda = 0
    nui = nui0

    params_evol = [np.array(x0)]
    cost, gradient, hessian = get_cost_grad_hessian(params_evol[iteropt])
    cost_evol = [cost]
    minusgrad = -1 * gradient

    if method == '2' or method == '3':
        lambdai = lambda0*np.max(np.diag(hessian))
    else:
        lambdai = lambda0
    if disp:
        print_cost(iteropt, cost, gradient,
                   info=('lambda', '{:.8e}'.format(lambdai)))

    while (iteropt < maxiter and step_cost > ftol
           and (rho <= eps_4 or np.linalg.norm(gradient) > gtol)
           and iter_lambda < iter_lambda_max):

        (new_param, cost, gradient,
         hessian, rho, lambdai) = one_step_LM(hessian, lambdai, minusgrad)
        if rho > eps_4:
            iteropt += 1
            iter_lambda = 0
            nui = nui0
            minusgrad = -1 * gradient
            cost_evol.append(cost)
            params_evol.append(new_param)
            step_cost = (cost_evol[-2] - cost_evol[-1])/(cost_evol[-1] + ftol)
            if disp:
                print_cost(iteropt, cost, gradient,
                           info=('lambda', '{:.8e}'.format(lambdai)))
        else:
            if disp:
                print('\tAdapt lambda: {:.8e}'.format(lambdai))
            iter_lambda += 1
            nui *= 2

    cost, gradient = get_cost_grad_hessian(params_evol[-1])[0:2]
    if iter_lambda >= iter_lambda_max:
        print('The LM process failed to adapt the lambda in the'
              ' maximal authorized iterations ({})'.format(iter_lambda_max))
    status, message = stop_message(iteropt, maxiter, cost, step_cost,
                                   ftol, gradient, gtol)
    result = HomemadeOptimizeResult(params_evol, cost_evol, gradient, status,
                                    message)
    return result


def homemade_minimization(cost_grad_hess, x0, maxiter=100, ftol=1e-8,
                          gtol=1e-10, disp=False, steptype='linesearch',
                          algo='LM'):
    """
    Minimize a differentiable function :math:`F(x)`

    Parameters
    ----------
    cost_grad_hess : callable
        A function returning, the cost, the gradient and eventually an
        estimation of the hessian for a given vector x. The hessian is used
        only for 'LM' and 'GN' algoirthms.
    x0 : np.array
        Initial values of x.
    maxiter : int, optional
        The maximal number of iterations. The default is 100.
    ftol : float, optional
        The stoping criterium on the relative variation of the cost function.
        The default is 1e-8.
    gtol : float, optional
        The stoping criteirum on the gradient value. The default is 1e-10.
    disp : booelan, optional
        Display informations at each iteration. The default is False.
    steptype : string, optional
        The steptype ('backtracking' ou 'linesearch') use on linesearch
        alogorithms. The default is 'linesearch'.
    algo : string, optional
        The optimization algorithm used. The default is 'LM'. Chose between

        - 'LM' : Levenberg-Marquardt see: :func:`LevenbergMarquardt`
        - 'GN' : Gauss-Newton see: :func:`GaussNewton`
        - 'QN' : Quasi-Newton using BFGS hessian estimation see:\
            :func:`QuasiNewtonBFGS`
        - 'steepest' : steepest descend, see: :func:`Steepest`

    Returns
    -------
    result : HomemadeOptimizeResult
        The result of the optimization.

    """

    if algo == 'LM':
        result = LevenbergMarquardt(cost_grad_hess, x0, maxiter, ftol,
                                    gtol, disp)
    elif algo == 'steepest':
        result = Steepest(cost_grad_hess, x0, maxiter, ftol, gtol, disp,
                          steptype)
    elif algo == 'QN':
        result = QuasiNewtonBFGS(cost_grad_hess, x0, maxiter, ftol, gtol,
                                 disp, steptype)
    elif algo == 'GN':
        result = GaussNewton(cost_grad_hess, x0, maxiter, ftol, gtol,
                             disp)
    else:
        raise ValueError("Unknown algorithm, choose between:\n "
                         "{'LM', 'GN', 'steepest', 'QN'}")
    return result
