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

"""
Delayed evaluation and differentiation of parameters.

In the context of optimization, equation coefficients must be evaluated
and differentiated for various values of parameters (such as tube length,
hole position, etc.).

These classes allows delayed evaluation of the shape parameters, as well as
their differentiation with respect to the optimization variables.
"""

from abc import ABC, abstractmethod

import numpy as np


def eval_(params):
    """
    Evaluate a list of :py:class:`DesignParameter<DesignParameter>`.

    Parameters
    ----------
    params : list of :py:class:`DesignParameter<DesignParameter>`

    Returns
    -------
    list of float
        The values of each parameter.

    """
    return [p.get_value() for p in params]


def diff_(params, diff_index):
    """
    Differentiate a list of :py:class:`DesignParameter<DesignParameter>`.

    Parameters
    ----------
    params : list of :py:class:`DesignParameter<DesignParameter>`
    diff_index : int
        Index of the optimized parameter considered for the differentiation

    Returns
    -------
    list of float
        The values of the differentiation of each parameter w.r. to the
        designated optimized parameter.

    """
    return [p.get_differential(diff_index) for p in params]


class OptimizationParameters:
    """Manage the variable parameters for optimization.

    All the variable parameters possibly modified during the optimization
    process are associated to one value of a optimized parameter list.


    Attributes
    ----------
    values : list of float
        The value associated to each parameter
    labels : list of string
        The label of each parameter
    active : list of bool
        If the parameter is include (True) or not (False) in the optimization
        process
    geom_value : list of :py:meth:`get_value \
        <openwind.design.design_parameter.DesignParameter.get_value>`
        The methods used to compute the geometric value of the parameters from
        the value stored in the :py:attr:`values<OptimizationParameters.values>` list.
    bounds : list of tuple of 2 floats
        The bounds for each parameter.
    lin_cons_indices : list of list of integer
        The indices of the parameters involved in each constraint
    lin_cons_coef : list of list of float
        The coefficients of each design parameters in each linear constraint
    lin_cons_bounds : list of tuple of 2 floats
        The lower and uper bounds for each linear constraint
    non_lin_cons_fun : list of callable
        The function defining each non-linear constraint
    non_lin_cons_jac : list of callable
        The jacobian of the non-linear function for each constraint
    non_lin_cons_bounds : list of tuple of 2 floats
        The lower and uper bounds for each non-linear constraint
    non_lin_cons_label : list of string
        The label of each non-linear constraint

    """

    MARGIN = 1e-7
    """ margin added to the bounds to guarantee their respect"""

    def __init__(self):
        self.values = list()
        self.labels = list()
        self.active = list()
        self.geom_values = list()
        self.bounds = list()

        self.lin_cons_indices = list()
        self.lin_cons_coef = list()
        self.lin_cons_bounds = list()

        self.non_lin_cons_fun = list()
        self.non_lin_cons_jac = list()
        self.non_lin_cons_bounds = list()
        self.non_lin_cons_label = list()

    def __str__(self):
        msg = ('{:20s}| {:>12s} || {:>10s} < {:>12s} < {:<10s} |'
               ' {:>6s}\n'.format('Labels', 'Geom.Values', 'Min',
                                  'Optim.Values', 'Max', 'Active'))
        msg += '-'*85 + '\n'
        for k in range(len(self.labels)):
            lb, ub = self.bounds[k]
            msg += ('{:20s}| {:12.5e} || {:10.3g} < {:12.5e} < {:<10.3g} |'
                    ' {:}\n').format(self.labels[k],
                                     self.get_geometric_values()[k],
                                     lb, self.values[k], ub, self.active[k])

        if len(self.lin_cons_indices)>0:
            lin_cons_msg = '\n===== Linear Constraints =====\n'
            for indices, coef, bounds in zip(self.lin_cons_indices, self.lin_cons_coef, self.lin_cons_bounds):
                lin_cons_msg += ('{:>4.3g} \t<\t').format(bounds[0])
                for ind, c in zip(indices, coef):
                    lin_cons_msg += ('{:+2.1g}*{:s} \t').format(c, self.labels[ind])
                lin_cons_msg += ('< \t{:<.3g}\n').format(bounds[1])
            msg += lin_cons_msg

        if len(self.non_lin_cons_fun)>0:
            nnlin_cons_msg = '\n===== Non-Linear Constraints =====\n'
            for label, bounds in zip(self.non_lin_cons_label, self.non_lin_cons_bounds):
                nnlin_cons_msg += ('{:>4.3g} \t<\t {:s} \t<\t '
                                   '{:>4.3g}\n').format(bounds[0], label,
                                                        bounds[1])
            msg += nnlin_cons_msg

        return msg

    def __repr__(self):
        return "<{class_}: {labels}>".format(class_=type(self).__name__,
                                             labels=self.labels)

    def new_param(self, value, label, get_geom_values,
                  bounds=(-np.inf, np.inf)):
        """
        Add a new optimized parameter.

        Parameters
        ----------
        value : float
            The initial value associated to the new parameter.
        label : str
            The label of the new parameter.
        get_geom_values : :py:meth:`get_value \ <openwind.design.design_parameter.DesignParameter.get_value>`
            The method used to compute the geometric value of the parameter
            from its stored value.
        bounds : tuple of two floats, optional
            The bounds associated to the new parameters: (min, max).
            The default value is no bounds: (-inf, inf).

        Returns
        -------
        new_index : int
            The index at which is stored this parameters in the lists of the
            py:class:`OptimizationParameters<OptimizationParameters>` object.

        """
        new_index = len(self.values)
        self.values.append(value)
        self.labels.append(label)
        self.active.append(True)
        self.geom_values.append(get_geom_values)
        self.bounds.append(bounds)

        return new_index

    def get_geometric_values(self):
        """
        Evaluate the geometric values of the stored paramters.

        Returns
        -------
        list of float

        """
        return [get_geom() for get_geom in self.geom_values]

    def set_active_parameters(self, indices):
        """
        Include the designated parameters in the optimization process.

        It modifies the :py:attr:`active<OptimizationParameters.active>`
        attribute.

        Parameters
        ----------
        indices : 'all' or list of int
            If `"all"`, all the stored parameters are included, either only the
            parameters corresponding to the given indices are included.

        """
        if isinstance(indices, str) and indices == 'all':
            active = np.ones_like(self.active)
        else:
            active = np.zeros_like(self.active)
            active[indices] = True
        self.active = active.tolist()

    def get_active_values(self):
        """
        Return the value of the parameters included in the optimization process

        Returns
        -------
        optim_values : list of float

        """
        optim_values = [value for (value, optim) in
                        zip(self.values, self.active) if optim]
        return optim_values

    def get_active_bounds(self):
        """
        Return the bounds of the parameters included in the optim. process.

        Returns
        -------
        bounds : list of tuple of float

        """
        bounds = [(bound[0] + self.MARGIN, bound[1] - self.MARGIN)
                  for (bound, optim) in zip(self.bounds, self.active) if optim]
        return bounds

    def set_active_values(self, new_values):
        """
        Modify the value of the parameters included in the optim. process.

        It is typically done at each step of an optimization process.

        Parameters
        ----------
        new_values : list of float
            The list of the new values. Its length must correspond to the
            number of parameters included in the optimization process (True in
            :py:attr:`active<OptimizationParameters.active>`)

        """
        values = np.array(self.values)
        values[self.active] = new_values
        self.values = values.tolist()

    def get_param(self, param_index):
        """
        Get the value of the designated parameter

        Parameters
        ----------
        param_index : int
            The index at which is stored the desired parameter.

        Returns
        -------
        float

        """
        return self.values[param_index]

    def diff_param(self, param_index, diff_index):
        """
        Differentiate the parameter with respect to one parameter.

        .. math::
            \\frac{\\partial \\zeta_k}{\\partial \\zeta_l}

        The result is typically 1 or 0.

        Parameters
        ----------
        param_index : int
            The parameter which is differentiate (:math:`k`).
        diff_index : int
            The index of the parameter w.r. to which the differentiation is
            computed (:math:`l`).

        Returns
        -------
        dparam : float
            The value of the differentiate: 1 if the two parameters correspond
            (:math:`k=l`), 0 either.
        """
        indices_active = np.where(self.active)[0]
        if param_index == indices_active[diff_index]:
            dparam = 1.
        else:
            dparam = 0.
        return dparam

    def add_linear_constraint(self, indices, coef, lb, ub):
        """
        Add a linear constraint.

        The linear constraint can be written as ``lb <= A.dot(x) <= ub``
        with `x` the design parameter vector.
        Each constraint is a line of the matrix `A`.

        everything is stored in attributes

        Parameters
        ----------
        indices : list of int
            The list of indices of A which must be different than 0.
        coef : list of float
            the value of A at the specified indices.
        lb : float
            the low bound.
        ub : float
            The upward bound.

        """
        self.lin_cons_indices.append(indices)
        self.lin_cons_coef.append(coef)
        self.lin_cons_bounds.append((lb, ub))

    def get_active_lin_cons(self):
        """
        Generate the linear constraints element corresponding to the active parameters.

        Only the column of the matrix A corresponding to the active parameters are kept
        the other columns are "moved" to the bounds.

        Returns
        -------
        A : np.array
            2D matrix with 1 column per active design parameter and 1 line per constraint.
        lb : np.array
            Low bound vector 1 one scalar value per constraint.
        ub : array
            Upward bound vector 1 one scalar value per constraint.

        """
        Nx = len(self.values)
        Ncons = len(self.lin_cons_indices)
        Atot = np.zeros((Ncons, Nx))
        lb = np.zeros(Ncons)
        ub = np.zeros(Ncons)
        for k, indices, coef, bounds in zip(range(Ncons), self.lin_cons_indices, self.lin_cons_coef, self.lin_cons_bounds):
            Atot[k, indices] = coef
            if bounds[0] == bounds[1]:
                lb[k] = bounds[0]
                ub[k] = bounds[1]
            else: # add a margin if 2 bounds differents
                lb[k] = bounds[0] + self.MARGIN
                ub[k] = bounds[1] - self.MARGIN
        # Move inactive values to the bounds
        A = Atot[:, self.active]
        A_inactive = Atot[:, np.logical_not(self.active)]
        inactive_values = [value for (value, optim) in
                           zip(self.values, self.active) if not optim]
        val = A_inactive.dot(inactive_values)
        lb -= val
        ub -= val
        return A, lb, ub

    def add_nonlinear_constraint(self, fun, jac, lb, ub, label='unknown'):
        """
        Add a non linear constraint

        The non-linear constraint can be written as ``lb <= fun(x) <= ub``
        with `x` the design parameter vector.

        Everything is stored in attributes

        Parameters
        ----------
        fun : callable
            The function given a scalar value.
        jac : callable
            The jacobian of the fun. It must have the dimension of the active design parameter
        lb : float
            the low bound.
        ub : float
            The upward bound.
        label : str, optional
            A short string describing the constraint (used for __str__). The default is 'unknown'.

        """
        self.non_lin_cons_fun.append(fun)
        self.non_lin_cons_jac.append(jac)
        self.non_lin_cons_bounds.append((lb, ub))
        self.non_lin_cons_label.append(label)

    def get_active_nonlin_cons(self):
        """
        Generate the element necessary for using the non-linear constraint in minimize algo

        The problem is written as ``lb<=fun(x)<=ub``
        where lb, fun(x) and ub can be 1D-array, and x is the active design
        parameters vector.

        Returns
        -------
        global_fun: callable
            A function rendering a vector of float
        global_jac: callable
            The jacobian of `global_fun`.
        lb: array of float
            The low bounds vector.
        ub: array of float
            The upward bounds vector.

        See Also
        --------
        `scipy.optimize.minimize doc<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html>`

        """

        def global_fun(x):
            self.set_active_values(x)
            fun_tot = list()
            for fun in self.non_lin_cons_fun:
                # ajouter la liste des indices concernes par cette contraintes pour ne pas la prendre en compte s'ils sont tous desactives?
                fun_tot.append(fun())
            return np.array(fun_tot)

        def global_jac(x):
            # indices_active = np.where(self.active)[0]
            jac_tot = list()
            for jac in self.non_lin_cons_jac:
                jac_tot.append(jac())
            return np.array(jac_tot)

        lb = np.array([b[0] for b in self.non_lin_cons_bounds])
        ub = np.array([b[1] for b in self.non_lin_cons_bounds])
        return global_fun, global_jac, lb, ub




# === The Different Kinds of Parameters ===


class DesignParameter(ABC):
    """
    A geometric parameter used to design an instrument.

    It is used to parameterized a :py:class:`DesignShape<openwind.design.DesignShape>`,
    or to specify the location of a side hole on the main bore.

    Parent class of the different kinds of parameters.
    """
    UNIT_DICT = {'m': 1, 'meter': 1, 'mm': 1000, 'millimeter': 1000}
    """ Dictionary giving the coefficient to convert meter in the indicated unit"""
    DIAMETER_DICT = {'True': 2, 'true': 2, 'False': 1, 'False': 1}
    """ Dictionary giving the coefficient to convert the radius in the right format following the diameter status (true or false)"""

    @abstractmethod
    def get_value(self):
        """Current geometric value of the parameter."""

    @abstractmethod
    def set_value(self, value):
        """
        set new geometric value of the parameter.

        Parameters
        ----------
        value : float
            The new value
        """

    @abstractmethod
    def get_differential(self, diff_index):
        """
        Differentiate the parameter with respect to one optimization variable.

        Parameters
        ----------
        diff_index : int
            Index of the active parameter of the :py:class:`OptimizationParameters \
            <openwind.design.design_parameter.OptimizationParameters>`
            considered for the differentiation.

        Return
        ---------
        float
            The value of the differentiale (Typically 1 or 0).

        """

    @abstractmethod
    def is_variable(self):
        """Variable status (Fixed: False, else: True)."""

    @staticmethod
    def get_writing_coef(unit='m', diameter=False):
        """
        The coef with which multiply the value in meter to get the value
        corresponding to the option
        """
        if unit in DesignParameter.UNIT_DICT.keys():
            unit_coef = DesignParameter.UNIT_DICT[unit]
        else:
            raise ValueError('Unknown unit, chose among:{}'.format(
                list(DesignParameter.UNIT_DICT.keys())))
        if diameter in DesignParameter.DIAMETER_DICT.keys():
            diam_coef = DesignParameter.DIAMETER_DICT[diameter]
        elif type(diameter) is bool:
            if diameter:
                diam_coef = 2
            else:
                diam_coef = 1
        else:
            raise ValueError('Unknown radius/diameter option, chose among:{}'.format(
                list(DesignParameter.DIAMETER_DICT.keys())))
        return unit_coef*diam_coef

    def get_jacobian(self):
        """
        Return the jacobian of the geom value wr to the active design parameter

        Returns
        -------
        np.array
            The Jacobian vecotr

        """
        if self.is_variable():
            J = list()
            for n in range(sum(self._optim_params.active)):
                J.append(self.get_differential(n))
            return np.array(J)
        else:
            return 0

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        coef = self.get_writing_coef(unit, diameter)
        form = '{:>' + str(digit+5) + '.' + str(digit) + 'f}'
        return (form.format(self.get_value()*coef)).rstrip('0').rstrip('.')

    def __repr__(self):
        return '{label}:{class_}({value})'.format(label=self.label,
                                                  class_=type(self).__name__,
                                                  value=self.__str__())


class FixedParameter(DesignParameter):
    """
    Parameter with a constant value.

    Parameters
    -----------
    value : float
        The geometric value of the parameter
    label : str, optional
        The parameter's name. The default is ''.

    """

    def __init__(self, value, label='', unit='m', diameter=False):
        coef = self.get_writing_coef(unit, diameter)
        self._value = value/coef
        self.label = label

    def get_value(self):
        return self._value

    def set_value(self, value):
        self._value = value

    def get_differential(self, diff_index):
        return 0

    def is_variable(self):
        return False


class VariableParameter(DesignParameter):
    """
    Variable geometric design parameter.

    Parameter the value of which can be possibly modify after instanciation,
    for example during an optimization process. The use of this class is
    coupled with the :py:class:`OptimizationParameters \
    <openwind.design.design_parameter.OptimizationParameters>`.

    The geometric value equals the stored optimized value. This value is
    bounded such as:

    .. math::
        x_{min}<x<x_{max}


    Parameters
    ----------
    value : float
        The initial geometric value of the parameter
    optim_params : :py:class:`OptimizationParameters \
    <openwind.design.design_parameter.OptimizationParameters>`
        The object where is stored the variable value
    label : str, optional
        The name of the parameter. the default is ''.
    bounds : list of two float
        The boundaries :math:`x_{min}` and :math:`x_{max}` of the authorized
        range for this parameter. The default is no bounds: `(-inf, inf)`

    Attributes
    ----------
    label : str
        The parameter name.
    index : int
        The position at which is stored this parmeter in the
        :py:class:`OptimizationParameters`
    """

    def __init__(self, value, optim_params, label='',
                 bounds=(-np.inf, np.inf), unit='m', diameter=False):
        coef = self.get_writing_coef(unit, diameter)
        self._optim_params = optim_params
        self.label = label
        if bounds[0] > bounds[1]:
            raise ValueError('The low bound {} must be smaller than the upper'
                             ' bound {}'.format(bounds[0], bounds[1]))
        if (value <= bounds[0] or value >= bounds[1]):
            raise ValueError('The initial value is not inside the authorized '
                             'range.')
        self.index = optim_params.new_param(value/coef, label, self.get_value,
                                            [b/coef for b in bounds])

    def get_value(self):
        return self._optim_params.get_param(self.index)

    def set_value(self, value):
        self._optim_params.values[self.index] = value

    def get_differential(self, diff_index):
        return self._optim_params.diff_param(self.index, diff_index)

    def is_variable(self):
        return True

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        coef = self.get_writing_coef(unit, diameter)
        bounds = self._optim_params.bounds[self.index]
        form_num = '{:.' + str(digit) + 'f}'
        msg = form_num.format(self.get_value()*coef)
        if disp_optim:
            msg = '~' + msg
            if not np.isinf(bounds[0]):
                msg = form_num.format(bounds[0]*coef) + '<' + msg
            if not np.isinf(bounds[1]):
                msg += '<' + form_num.format(bounds[1]*coef)
        form_msg = '{:>' + str(digit+5) + 's}'
        return form_msg.format(msg)


class VariableHolePosition(DesignParameter):
    """
    Variable hole position defined relatively on the main bore pipe.

    The location of the hole :math:`x_{hole}` is defined as:

    .. math::
        \\begin{align}
        x_{hole} & =  (x_1 - x_0) \\zeta + x_0 \\\\
        0 & \\leq \\zeta \\leq 1
        \\end{align}

    with :math:`\\zeta` an auxiliary parameter bounded between 0 and 1,
    :math:`(x_{0},x_{1})` the boundaries of the main bore pipe
    where is placed the considered hole. It's assure that:
    :math:`x_0 \\leq x_{hole} \\leq x_1`.

    .. warning::
        The stocked and optimized value is :math:`\\zeta` and not \
        :math:`x_{hole}`

    Parameters
    ----------
    init_value : float
        The initial geometric value of the parameter
    optim_params : :py:class:`OptimizationParameters \
    <openwind.design.design_parameter.OptimizationParameters>`
        The object where is stored the variable value
    main_bore_shape : :py:class:`DesignShape <openwind.design.DesignShape>`
        The shape of the pipe where is located the hole.
    label : str, optional
        The name of the parameter. the default is ''.

    Attributes
    ----------
    label : str
        The parameter name.
    index : int
        The position at which is stored this parmeter in the
        :py:class:`OptimizationParameters \
        <openwind.design.design_parameter.OptimizationParameters>`
    """

    def __init__(self, init_value, optim_params, main_bore_shape, label='',
                 unit='m', diameter=False):
        coef = self.get_writing_coef(unit, diameter)
        self._optim_params = optim_params
        self.label = label
        self._main_bore_shape = main_bore_shape
        norm_position = main_bore_shape.get_xnorm_from_position(
            init_value/coef)
        self.index = optim_params.new_param(norm_position, label,
                                            self.get_value, (0, 1))

    def is_variable(self):
        return True

    def get_value(self):
        x_norm = self._optim_params.get_param(self.index)
        value = self._main_bore_shape.get_position_from_xnorm(x_norm)
        return value

    def set_value(self, value):
        norm_position = self._main_bore_shape.get_xnorm_from_position(value)
        self._optim_params.values[self.index] = norm_position

    def get_differential(self, diff_index):
        d_zeta = self._optim_params.diff_param(self.index, diff_index)
        zeta = self._optim_params.get_param(self.index)

        x_norm = zeta
        dx_norm = d_zeta

        Xmin, Xmax = eval_(self._main_bore_shape.get_endpoints_position())
        d_position = (self._main_bore_shape
                      .get_diff_position_from_xnorm(x_norm, diff_index))
        return dx_norm*(Xmax - Xmin) + d_position

    def create_value_nonlin_cons(self, min_value, max_value):
        r"""
        Add a Non-lin cons. in the OptimParams, such as this parameter value, respect

        .. math::
            x_{min} \leq x \leq x_{max}

        Parameters
        ----------
        min_value : float
            the x_min value.
        max_value : float
            the x_max value.
        """
        self._optim_params.add_nonlinear_constraint(self.get_value,
                                                    self.get_jacobian,
                                                    min_value,
                                                    max_value,
                                                    label='Geom. ' + self.label)

    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        coef = self.get_writing_coef(unit, diameter)
        form_num = '{:.' + str(digit) + 'f}'
        if disp_optim:
            form_num = '~' + form_num + '%'
        msg = form_num.format(self.get_value()*coef)
        form_msg = '{:>' + str(digit+5) + 's}'
        return form_msg.format(msg)


class VariableHoleRadius(DesignParameter):
    """
    Variable hole radius defined relatively to the main bore pipe radius.

    The hole raidus :math:`r_{hole}` is defined as:

    .. math::
        \\begin{align}
        r_{hole} & = r_{main} \\zeta \\\\
        0.0001 & \\leq \\zeta \\leq 1
        \\end{align}

    with :math:`\\zeta` an auxiliary parameter bounded between 0 and 1,
    :math:`r_{main}` the radius of the main bore pipe at the
    position of the considered hole. It's assure that:
    :math:`r_{hole} \\leq r_{main}`

    .. warning::
        The stocked and optimized value is :math:`\\zeta` and not \
        :math:`r_{hole}`

    Parameters
    ----------
    init_value : float
        The initial geometric value of the parameter
    optim_params : :py:class:`OptimizationParameters \
    <openwind.design.design_parameter.OptimizationParameters>`
        The object where is stored the variable value
    main_bore_shape :  :py:class:`DesignShape <openwind.design.DesignShape>`
        The shape of the pipe where is located the hole.
    hole_position : :py:class:`DesignParameter \
    <openwind.design.design_parameter.DesignParameter>`
        The position of the hole.
    label : str, optional
        The name of the parameter. the default is ''.

    Attributes
    ----------
    label : str
    index : int
        The position at which is stored this parmeter in the
        `OptimizationParameters`
    """

    def __init__(self, init_value, optim_params, main_bore_shape,
                 hole_position, label='', unit='m', diameter=False):
        coef = self.get_writing_coef(unit, diameter)
        self._optim_params = optim_params
        self.label = label
        self._hole_position = hole_position
        self._main_bore_shape = main_bore_shape

        x_norm = self.__get_x_norm()
        radius_main_pipe = main_bore_shape.get_radius_at(x_norm)
        # zeta = inv_sigmoid(init_value/radius_main_pipe)
        zeta = init_value/radius_main_pipe/coef
        self.index = optim_params.new_param(zeta, label, self.get_value,
                                            (1e-4, 1))

    def is_variable(self):
        return True

    def get_value(self):
        zeta = self._optim_params.get_param(self.index)
        x_norm = self.__get_x_norm()
        radius_main_pipe = self._main_bore_shape.get_radius_at(x_norm)
        value = radius_main_pipe * zeta
        return value

    def set_value(self, value):
        x_norm = self.__get_x_norm()
        radius_main_pipe = self._main_bore_shape.get_radius_at(x_norm)
        zeta = value/radius_main_pipe
        self._optim_params.values[self.index] = zeta

    def __get_x_norm(self):
        pos = self._hole_position.get_value()
        return self._main_bore_shape.get_xnorm_from_position(pos)

    def __get_diff_x_norm(self, diff_index):
        dXmin, dXmax = diff_(self._main_bore_shape.get_endpoints_position(),
                             diff_index)
        Xmin, Xmax = eval_(self._main_bore_shape.get_endpoints_position())
        dPos = self._hole_position.get_differential(diff_index)
        Pos = self._hole_position.get_value()
        return (((dPos - dXmin)*(Xmax - Xmin) - (Pos - Xmin)*(dXmax - dXmin))
                / (Xmax - Xmin)**2)

    def get_differential(self, diff_index):
        d_zeta = self._optim_params.diff_param(self.index, diff_index)
        zeta = self._optim_params.get_param(self.index)

        x_norm = self.__get_x_norm()
        r_pipe = self._main_bore_shape.get_radius_at(x_norm)

        dx_norm = self.__get_diff_x_norm(diff_index)
        dr_pipe_dx_norm = self._main_bore_shape.diff_radius_wr_x_norm(x_norm)
        dr_pipe_diff_index = (self._main_bore_shape
                              .get_diff_radius_at(x_norm, diff_index))
        return (d_zeta*r_pipe
                + zeta*dr_pipe_dx_norm*dx_norm
                + zeta*dr_pipe_diff_index)

    def create_value_nonlin_cons(self, min_value, max_value):
        self._optim_params.add_nonlinear_constraint(self.get_value,
                                                    self.get_jacobian,
                                                    min_value,
                                                    max_value,
                                                    label='Geom. ' + self.label)



    def __str__(self, digit=5, unit='m', diameter=False, disp_optim=True):
        coef = self.get_writing_coef(unit, diameter)
        form_num = '{:.' + str(digit) + 'f}'
        if disp_optim:
            form_num = '~' + form_num + '%'
        msg = form_num.format(self.get_value()*coef)
        form_msg = '{:>' + str(digit+5) + 's}'
        return form_msg.format(msg)
