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

"""High-level interface to run impedance computations."""

import warnings
import inspect

import numpy as np

from openwind.technical import InstrumentGeometry, Player
from openwind.continuous import InstrumentPhysics, Physics
from openwind.discretization import Mesh
from openwind.frequential import FrequentialSolver

# import timeit


class ImpedanceComputation:
    """
    Compute the input impedance of a geometry at the frequencies specified.

    This high-level class bypasses several classes, each ones having its own
    options possibly indicated here.

    This high-level class performs more or less the following steps:

    .. code-block:: python

        my_geometry = InstrumentGeometry(*files, **kwargs_geom)
        my_physics = InstrumentPhysics(my_geometry, temperature, player; **kwargs_phy)
        my_freq_model = FrequentialSolver(my_physics, frequencies, **kwargs_freq)
        my_freq_model.solve(**kwargs_solve)
        self.impedance = my_freq_model.impedance
        self.Zc = my_freq_model.get_ZC_adim()


    where `kwargs_phy`  and `kwargs_freq` are different options which can be
    specified by the user.

    See Also
    --------
    :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`
        For information concerning how the geometry of the instrument is specified
    :py:class:`InstrumentPhysics <openwind.continuous.instrument_physics.InstrumentPhysics>`
        For information concerning the graph of the instrument and the options
        `[temperature, losses, convention, nondim, radiation_category, \
         spherical_waves, discontinuity_mass, matching_volume]`
    :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`
        For information concerning the frequential domain resolution and the
        options: `[compute_method, l_ele, order, nb_sub, note]`
    :py:class:`Player <openwind.technical.player.Player>`
        For information concerning `player` option


    Parameters
    ----------
    fs : numpy.array
        Frequencies at which to compute the impedance.

    main_bore : str or list
        filename or list of data respecting the file format with the
        main bore geometry. See also : :py:class:`InstrumentGeometry \
        <openwind.technical.instrument_geometry.InstrumentGeometry>`
    holes_or_vales : str or list, optional
        filename or list of data respecting the file format, with the
        holes and/or valves geometries. The default is an empty list corresponding to
        an instrument without hole or valve. See also : :py:class:`InstrumentGeometry \
        <openwind.technical.instrument_geometry.InstrumentGeometry>`
    fingering_chart : str or list, optional
        filename or list of data respecting the file format, indicating the
        fingering chart in accordance with the holes and/or valves. The default
        is an empty list corresponding to no fingering (everything open).
        See also : :py:class:`InstrumentGeometry \
        <openwind.technical.instrument_geometry.InstrumentGeometry>`

    unit: str {'m', 'mm'}, optional
        The unit (meter or millimeter). Default is 'm' (meter)

    diameter: boolean, optional
        If True assume that diameter are given instead of radius. The default is False.

    player : :py:class:`Player <openwind.technical.player.Player>`, optional
        An object specifying how the instrument is "played". Default is `Player()`,
        corresponding to a unitary flow imposed at each frequency.

    temperature : float or callable, optional
        Temperature along the instrument in Celsius degree. Default is 25
        See also : :py:class:`InstrumentPhysics\
        <openwind.continuous.instrument_physics.InstrumentPhysics>`

    losses : bool or {'bessel', 'wl','keefe','diffrepr', 'diffrepr+'}, optional
        Whether/how to take into account viscothermal losses. Default is True.
        If 'diffrepr+', use diffusive representation with explicit additional
        variables.
        See also : :py:mod:`thermoviscous_models <openwind.continuous.thermoviscous_models>`

    radiation_category : str, tuple, dict or :py:class:`PhysicalRadiation \
        <openwind.continuous.physical_radiation.PhysicalRadiation>` , optional
        Model of radiation impedance used. The string must be one of the
        available category ('unflanged', 'infinite_flanged', ...). The use of
        dict gives the possibility to use different condition at each opening.
        Default is 'unflanged'.
        See also: :py:class:`InstrumentPhysics \
            <openwind.continuous.instrument_physics.InstrumentPhysics>`
        More details on available model names in :py:meth:`radiation_model \
        <openwind.continuous.radiation_model.radiation_model>`.

    spherical_waves : Boolean, optional
        If true, spherical waves are assumed in the pipe. The default is False.
        Option spherical_waves='spherical_area_corr' also enables the area correction
        (otherwise only the length correction is used).
        See also: :py:class:`InstrumentPhysics \
            <openwind.continuous.instrument_physics.InstrumentPhysics>`

    discontinuity_mass : Boolean, optional
        If true, acoustic mass is included in the junction between two
        pipes with different cross section. The default is True.
        See also: :py:class:`InstrumentPhysics\
        <openwind.continuous.instrument_physics.InstrumentPhysics>`,
        :py:class:`JunctionDiscontinuity\
        <openwind.continuous.junction.JunctionDiscontinuity>`

    matching_volume : boolean, optional
        Include or not the matching volume between the main and the side
        tubes in the masses of the T-joint junctions. The default is False.
        See also: :py:class:`InstrumentPhysics\
        <openwind.continuous.instrument_physics.InstrumentPhysics>`,
        :py:class:`JunctionTjoint\
        <openwind.continuous.junction.JunctionTjoint>`

    nondim : bool, optional
        Nondimensionalization mode. If activated, the physical quantities
        are nondimensionalized so that they are closer to 1. Default {False}.
        See also: :py:class:`InstrumentPhysics\
        <openwind.continuous.instrument_physics.InstrumentPhysics>`

    compute_method : {'FEM', 'TMM', 'hybrid', 'modal'}, optional
        Method chose to compute the frequency response (Default 'FEM'):

        - 'FEM' = finite elements method. See [Tour_FEM]_
        - 'TMM' = transfer matrix method
        - 'hybrid' = TMM for cylinders, FEM either
        - 'modal' = modal method based on finite element discretisation. See [Chab_Modal]_

        See also : :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`

    l_ele, order : list, optional, only used for 'FEM' or 'hybrid'
        Elements lengths and orders. Default is None: automatic meshing.
        See also : :py:class:`Mesh <openwind.discretization.mesh.Mesh>`

    nb_su : integer, optional, only used for TMM
        Number of subdivisions of each conical part. Default is 1.
        See also : :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`,
        :py:class:`FrequentialPipeTMM <openwind.frequential.frequential_pipe_tmm.FrequentialPipeTMM>`

    note : str, optional
        The note name corresponding to the right fingering, as specified in the
        given :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`.
        The default is None, corresponding to all open fingering.

    kwargs : keyword argument
        Any option wich can be used with:
        :py:class:`InstrumentPhysics<openwind.technical.instrument_geometry.InstrumentGeometry>`,
        :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`,
        :py:class:`FrequentialSolver <openwind.frequential.frequential_solver.FrequentialSolver>`,
        :py:func:`FrequentialSolver.solve() <openwind.frequential.frequential_solver.FrequentialSolver.solve()>`,


    References
    ----------
    .. [Tour_FEM] Tournemenne, R., & Chabassier, J. (2019).\
        A comparison of a one-dimensional finite element method \
            and the transfer matrix method for the computation of \
                wind music instrument impedance. Acta Acustica \
                    united with Acustica, 105(5), 838-849.
    .. [Chab_Modal] Chabassier, J., & Auvray, R. (2022).\
        Direct computation of modal parameters for musical \
            wind instruments. Journal of Sound and Vibration, 528, 116775.

    loss_factor_alpha: float, optional
        Multiply the amount of viscothermal losses by this factor. If different
        from 1, losses must be one of [True, 'bessel', 'bessel_new'].
        The default is 1.

    Attributes
    -----------
    impedance : np.array
        The complex impedance at the entrance of the instrument at each
        frequency.

    Zc : float
        The real characteristics impedance (rho c / S) at the entrance of the
        instrument, usefull to scale the input impedance.


    """


    def __init__(self, frequencies, main_bore, holes_valves=list(),
                 fingering_chart=list(),
                 # Geom options
                 unit='m', diameter=False,
                 # Physical options
                 player=Player(), temperature=None, losses=True, nondim=True,
                 radiation_category='unflanged', spherical_waves=False,
                 discontinuity_mass=True, matching_volume=False,
                 # Freq options
                 compute_method='FEM', l_ele=None, order=None, nb_sub=1, note=None,
                 use_rad1dof=False, diff_repr_vars=False,
                 # solve options
                 interp=False, interp_grid='original',
                 **kwargs # any additional keywords
                 ):

        if not temperature:
            temperature=25
            warnings.warn('The default temperature is 25 degrees Celsius.')

        if losses == 'diffrepr+': # Use Diffusive Representation with additional variables
            losses = 'diffrepr'
            diff_repr_vars = True

        kwargs_geom, kwargs_phy, kwargs_freq, kwargs_solve = self._check_kwargs(kwargs)

        geom_options = {'unit':unit, 'diameter':diameter}
        self.__instrument_geometry = InstrumentGeometry(main_bore, holes_valves,
                                                        fingering_chart, **geom_options, **kwargs_geom)

        phy_options = dict(radiation_category=radiation_category,
                           nondim=nondim,
                           spherical_waves=spherical_waves,
                           discontinuity_mass=discontinuity_mass,
                           matching_volume=matching_volume)
        self.__instru_physics = InstrumentPhysics(self.__instrument_geometry,
                                                  temperature, player=player,
                                                  losses=losses, **phy_options, **kwargs_phy)

        freq_options = {'compute_method':compute_method,
                        'use_rad1dof':use_rad1dof,
                        'diffus_repr_var':diff_repr_vars,
                        'l_ele':l_ele,
                        'order':order,
                        'note':note,
                        'nb_sub':nb_sub}
        # tic = timeit.default_timer()
        self.__freq_model = FrequentialSolver(self.__instru_physics, frequencies,
                                              **freq_options, **kwargs_freq)
        self.frequencies = self.__freq_model.frequencies
        # toc = timeit.default_timer()
        # print(f"Time spent assembling : {toc-tic:2.3f} sec")
        self.__freq_model.solve(interp=interp, interp_grid=interp_grid, **kwargs_solve)
        # tuc = timeit.default_timer()
        # print(f"Time spent solving : {tuc-toc:2.3f} sec")
        self.impedance = self.__freq_model.imped
        self.Zc = self.__freq_model.get_ZC_adim()
        # Small hack : give visibility to ALL the attributes of __freq_model
        # self.__dict__.update(self.__freq_model.__dict__)

    def __repr__(self):
        return ("<openwind.ImpedanceComputation("
                "\n{},".format(repr(self.__instru_physics)) +
                "\n{},".format(repr(self.__freq_model)) +
                "\n)>")

    def __str__(self):
        return ("{}\n\n" + 30*'*' + "\n\n{}").format(self.__instru_physics,
                                                     self.__freq_model)

    @staticmethod
    def _check_kwargs(kwargs):
        """
        Distribute additional keywords arguments between the classes.

        Parameters
        ----------
        kwargs : dict
            The additional keywords

        Raises
        ------
        TypeError
            Raise error if one of the argument does not correspond to any options.

        Returns
        -------
        kwargs_geom : dict
            The kwargs related to InstrumentGeometry.
        kwargs_phy : TYPE
            The kwargs related to InstrumentPhysics.
        kwargs_freq : TYPE
            The kwargs related to FrequentialSolver.
        kwargs_solve : TYPE
            The kwargs related to FrequentialSolver.solve().
        """
        def get_keys(my_class):
            return [s.name for s in inspect.signature(my_class).parameters.values() if s.kind is not s.VAR_KEYWORD]
        # get all the argument possible for each class/method without the **kwargs
        keys_geom = get_keys(InstrumentGeometry)
        keys_phy = get_keys(InstrumentPhysics) + get_keys(Physics)
        keys_freq = get_keys(FrequentialSolver)
        keys_mesh = get_keys(Mesh)[1:] + ['nb_sub', 'reff_tmm_losses']
        keys_solve = get_keys(FrequentialSolver.solve)[1:]

        tot_keys = keys_geom + keys_phy + keys_freq + keys_mesh + keys_solve

        if any([key not in tot_keys for key in kwargs.keys()]):
            raise TypeError('Unexpected keyword argument: {}'.format( [key for key in kwargs.keys() if key not in tot_keys]))

        kwargs_geom = { key: kwargs[key] for key in kwargs.keys() if key in keys_geom }
        kwargs_phy = { key: kwargs[key] for key in kwargs.keys() if key in keys_phy }
        kwargs_freq = { key: kwargs[key] for key in kwargs.keys() if key in keys_freq + keys_mesh }
        kwargs_solve = { key: kwargs[key] for key in kwargs.keys() if key in keys_solve }

        return kwargs_geom, kwargs_phy, kwargs_freq, kwargs_solve

    def set_note(self, note):
        """
        Update the note (fingering) apply to the instrument and compute the
        new impedance.

        See Also
        --------
        :py:meth:`FrequentialSolver.set_note() \
        <openwind.frequential.frequential_solver.FrequentialSolver.set_note>`

        Parameters
        ----------
        note : str
            The note name. It must correspond to one of the associated
            :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`.

        """
        self.__freq_model.set_note(note)
        self.__freq_model.solve()
        self.impedance = self.__freq_model.imped

    def set_frequencies(self, frequencies):
        """
        An overlay of recompute_impedance_at()

        .. deprecated:: 0.8.1
            This method will be replaced by \
            :py:meth:`solve()<FrequentialSolver.recompute_impedance_at()>` instead
        """
        warnings.warn('The method ImpedanceCompuation.set_frequencies() is deprecated,'
                      ' please use recompute_impedance_at() instead.')
        self.recompute_impedance_at(frequencies)

    def recompute_impedance_at(self, frequencies):
        """
        Recompute the impedance at the specified frequencies

        If necessary this method updates the mesh and resolve the entire problem.
        Following the compute method chosen and the difference with the precedent
        frequency range it can be a long computation.

        See Also
        --------
        :py:meth:`FrequentialSolver.recompute_impedance_at() \
        <openwind.frequential.frequential_solver.FrequentialSolver.recompute_impedance_at>`

        Parameters
        ----------
        frequencies : array of float
            The new frequency axis.

        Returns
        -------
        np.array
            The new impedance (stored also in :py:attr:`ImpedanceComputation.impedance`).

        """
        self.__freq_model.recompute_impedance_at(frequencies)
        self.frequencies = self.__freq_model.frequencies
        self.impedance = self.__freq_model.impedance
        return self.impedance

    def evaluate_impedance_at(self, freqs):
        """
        Re-evaluate the impedance at given frequencies freqs without updating the mesh

        .. warning::
            This method does not update the mesh. If you want to automatically update it use
            :meth:`ImpedanceComputation.recompute_impedance_at()` instead

        Parameters
        ----------
        freqs : array
            Frequencies, in Hz.

        Returns
        -------
        array
            The values of the impedance at the frequencies freqs (stored also in :py:attr:`ImpedanceComputation.impedance`).

        """
        return self.__freq_model.evaluate_impedance_at(freqs)

    def plot_instrument_geometry(self, figure=None, **kwargs):
        """
        Display the geometry and holes of the instrument.

        If a note name is given, also display the fingering of that note.

        Parameters
        ----------
        figure: matplotlib.figure.Figure, optional
            Which figure to use. By default opens a new figure.
        note: str, optional
            If a note name is given, closed holes are filled, whereas
            open holes are outlined.
            By default all holes are outlined.
        kwargs:
            Additional arguments are passed to the `plt.plot` function.

        """
        self.__instrument_geometry.plot_InstrumentGeometry(figure=figure,
                                                           **kwargs)

    def plot_impedance(self, **kwargs):
        """
        Plot the normalized impedance.

        It uses :py:func:`openwind.impedance_tools.plot_impedance`

        Parameters
        ----------
        **kwargs : keyword arguments
            They are transmitted to :py:func:`plot_impedance()\
            <openwind.impedance_tools.plot_impedance>`.

        """
        self.__freq_model.plot_impedance(**kwargs)

    def plot_admittance(self, **kwargs):
        """
        Plot the normalized admittance.

        It uses :py:func:`openwind.impedance_tools.plot_impedance`

        Parameters
        ----------
        **kwargs : keyword arguments
            They are transmitted to :py:func:`plot_impedance()\
            <openwind.impedance_tools.plot_impedance>`.

        """
        self.__freq_model.plot_admittance(**kwargs)

    def write_impedance(self, filename, column_sep=' ', normalize=False):
        """
        Write the impedance in a file.

        The file has the format
        "(frequency) (real part of impedance) (imaginary part of impedance)"

        See :py:func:`openwind.impedance_tools.write_impedance`

        Parameters
        ----------
        filename : string
            The name of the file in which is written the impedance (with the
            extension).
        column_sep : str, optional
            The column separator. Default is ' ' (space)
        normalize : bool, optional
            Normalize or not the impedance by the input characteristic
            impedance. The default is False.

        """
        self.__freq_model.write_impedance(filename, column_sep, normalize)

    def resonance_frequencies(self, k=5, display_warning=True):
        """
        The resonance frequencies of the impedance

        Depending of the solving method used, it uses the function :func:`openwind.impedance_tools.resonance_frequencies`

        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.
        display_warning: boolean, optional
            If false, does not display the warning relative to the method employed. Default: True

        Returns
        -------
        list of float

        """
        return self.__freq_model.resonance_frequencies(k, display_warning)

    def resonance_peaks(self,k=5, display_warning=True):
        """
        The resonance frequencies, quality factors and values of the impedance

        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.
        display_warning: boolean, optional
            If false, does not display the warning relative to the method employed. Default: True

        Returns
        -------
        tuple of 3 lists
            The resonance frequencies (float)

            The quality factors (float)

            The impedance value at the resonance frequencies (complex)

        """
        return self.__freq_model.resonance_peaks(k, display_warning)

    def antiresonance_peaks(self, k=5, display_warning=True):
        """
        The antiresonance frequencies, quality factors and values of the impedance


        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.
        display_warning: boolean, optional
            If false, does not display the warning relative to the method employed. Default: True

        Returns
        -------
        tuple of 3 lists
            The antiresonance frequencies (float)

            The quality factors (float)

            The impedance value at the resonance frequencies (complex)

        """
        return self.__freq_model.antiresonance_peaks(k, display_warning)

    def match_peaks_with_notes(self, concert_pitch_A=440, transposition = 0, k=5, display=False):
        """
        Matches resonance frequencies with notes frequencies in Hz, deviation in cents and notes names
        The user can specify a concert pitch and a transposing behavior for the instrument.

        Parameters
        ----------
        concert_pitch_A: float, optional
            Frequency of the concert A4, in Hz.
            Default value is 440 Hz.
        transposition: int or string, optional
            indicates if the instrument is transposing.
            If an integer is given, it must be the number of semitones between the played C and the actual heard C
            If a note name is given, the number of semitones will be deduced (-2 for "Bb" instrument, for instance)
            Default is 0.
        k: int, optional
            number of resonance frequencies considered
            Default is 5.
        display : boolean, optional
            If true, display the result for each mode. Default is False.

        Returns
        -------
        tuple of 3 lists
           - The closest notes frequencies (float)
           - The deviation of the resonance frequencies (float)
           - The names of the closest notes, in the given concert pitch and transposition system (string)

        """
        return self.__freq_model.match_peaks_with_notes(concert_pitch_A, transposition, k, display)


    def antiresonance_frequencies(self, k=5, display_warning=True):
        """
        The antiresonance frequencies of the impedance

        It uses the function :func:`openwind.impedance_tools.antiresonance_frequencies`

        Parameters
        ----------
        k : int, optional
            The number of resonance included. The default is 5.
        display_warning: boolean, optional
            If false, does not display the warning relative to the method employed. Default: True

        Returns
        -------
        list of float

        """
        return self.__freq_model.antiresonance_frequencies(k, display_warning)

    def discretization_infos(self):
        """
        Information of the total mesh used to solve the problem.

        See Also
        --------
        :py:mod:`discretization <openwind.discretization>`

        :py:class:`Mesh <openwind.discretization.mesh.Mesh>`

        Returns
        -------
        str
        """
        self.__freq_model.discretization_infos()

    def technical_infos(self):
        """
        Print technical information on the instrument geometry and the player.

        See Also
        --------
        :py:mod:`technical <openwind.technical>`

        :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`

        :py:class:`Player <openwind.technical.player.Player>`

        """
        print(self.__instrument_geometry)
        self.__instru_physics.player.display()

    def get_instrument_geometry(self):
        """
        The instrument geometry object.

        Returns
        -------
        :py:class:`InstrumentGeometry <openwind.technical.instrument_geometry.InstrumentGeometry>`

        """

        return self.__instrument_geometry

    def get_all_notes(self):
        """
        Return all the notes specified in the fingering chart

        Returns
        -------
        list[string]
            The list of the notes names
        """
        return self.__instrument_geometry.fingering_chart.all_notes()

    def get_entry_coefs(self, *labels):
        """
        Obtain values of physical properties of air at the entry point.

        Parameters
        ----------
        *labels : string...
            the names of the coefficients to take.
            See :py:class:`Physics <openwind.continuous.physics.Physics>`

        Returns
        -------
        values : tuple of (float or array-like)

        Example
        -------

        .. code-block:: python

           result.get_entry_coefs('Cp', 'c', 'gamma', 'kappa', 'mu', 'rho', 'temp', 'T')

        """
        return self.__instru_physics.get_entry_coefs(*labels)

    def get_nb_dof(self):
        """
        The total number of degrees of freedom (dof) used to solve the problem.

        Returns
        -------
        int

        """
        return self.__freq_model.n_tot


    def get_pressure_flow(self):
        """
        Return x, pressure and flow for all frequencies

        This fields are estimated only if the "interp" option is set to True

        Returns
        -------
        x : 1D array
            The positions at which are estimated the pressure and flow
        pressure : 2D array
            The pressure at each position and for each frequency.
        flow : 2D array
            The flow at each position and for each frequency.

        """
        if hasattr(self.__freq_model, 'x_interp'):
            return self.__freq_model.x_interp, self.__freq_model.pressure, self.__freq_model.flow
        else:
            raise ValueError('The pressure and flow are not computed. To have access to them, please set "interp" option to True.')


    def plot_ac_field(self, dbscale=True, var='pressure', with_plotly=False, **kwargs):
        """
        Plot one acoustic field in the instrument for every  frequency on a surface.

        Parameters
        ----------
        dbscale : bool, optional
            Plot the fields with a dB scale or not. The default is True.
        var : 'pressure' or 'flow', optional
            Which field must be plotted. The default is 'pressure'.
        with_plotly : boolean, optional
            If True and plotly installed, use plotly instead of matplotlib
        **kwargs : key word arguments
            Passed to `plt.pcolor() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.pcolor.html>`_.

        """
        if hasattr(self.__freq_model, 'x_interp'):
            self.__freq_model.plot_var3D(dbscale=dbscale, var=var, with_plotly=with_plotly, **kwargs)
        else:
            raise ValueError('The pressure and flow are not computed. To have access to them, please set "interp" option to True.')
