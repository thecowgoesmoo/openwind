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

import warnings

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.sparse import csr_matrix, SparseEfficiencyWarning
from scipy.sparse.linalg import spsolve
import scipy.sparse as ssp
import numpy.linalg as LA

import openwind.impedance_tools as tools
from openwind.design import Cone
from openwind.technical import Fingering
from openwind.continuous import (PhysicalRadiation, Excitator, Flow, Flute,
                                 JunctionTjoint, JunctionSimple, JunctionSwitch,
                                 ThermoviscousLossless, JunctionDiscontinuity,
                                 ThermoviscousDiffusiveRepresentation,
                                 ThermoviscousBessel,
                                 SphericalHarmonics,
                                 WebsterLokshin,
                                 RadiationPerfectlyOpen, RadiationPade)
from openwind.frequential import (FrequentialPipeFEM, FrequentialRadiation,
                                  FrequentialRadiation1DOF,
                                  FrequentialJunctionTjoint,
                                  FrequentialJunctionSimple,
                                  FrequentialJunctionDiscontinuity,
                                  FrequentialJunctionSwitch,
                                  FrequentialSource, FrequentialFluteSource,
                                  FrequentialInterpolation,
                                  FrequentialPipeDiffusiveRepresentation,
                                  FrequentialPipeTMM,
                                  FrequentialPressureCondition,
                                  FrequentialComponent)
from openwind.tracker import SimulationTracker

import timeit


class FrequentialSolver:
    """
    Solve equations in the frequential domain.

    In the frequential domain, only the wave propagation into the instrument
    is solved. It allows the computation of the acoustic fields into the
    entire instrument and the impedance.

    Parameters
    ----------
    instru_physics : :py:class:`InstrumentPhysics<openwind.continuous.instrument_physics.InstrumentPhysics>`
        The object describing the physics of the instruments.
    frequencies : numpy.array
        Frequencies at which to compute the impedance.
    diffus_repr_var: bool, optional
        Whether to use additional variables when computing the diffusive
        representation. The default is False.
    note : str, optional
        The name of the note corresponding to the fingering which must be
        applied. The default is None and correspond to all holes open.
    compute_method: str in {'FEM', 'TMM', 'hybrid'}, optional
        Which method must be used, the default is FEM:

        - 'FEM': finite element method, usable with any geometry. See [Tour_FEM_a]_
        - 'TMM': Transfer Matrix Method, usable only with conical pipes
        - 'hybrid': mix between FEM and TMM: the cylinders are computed with\
        TMM (exact solution) and the other pipes with FEM
        - 'modal' = modal method based on finite element discretisation. See [Chab_Modal_a]_
    use_rad1dof : boolean, optional
        If True, add 1 dof for the radiation condition (necessary for modal computation).
        See :py:class:`FrequentialRadiation1DOF <openwind.frequential.frequential_radiation1dof.FrequentialRadiation1DOF>`.
        Default is False.
    discr_params : keyword arguments
        Discretization parameters. See: :py:class:`Mesh <openwind.discretization.mesh.Mesh>`

    References
    ----------
    .. [Tour_FEM_a] Tournemenne, R., & Chabassier, J. (2019).\
        A comparison of a one-dimensional finite element method \
            and the transfer matrix method for the computation of \
                wind music instrument impedance. Acta Acustica \
                    united with Acustica, 105(5), 838-849.
    .. [Chab_Modal_a] Chabassier, J., & Auvray, R. (2022).\
        Direct computation of modal parameters for musical \
            wind instruments. Journal of Sound and Vibration, 528, 116775.

    Attributes
    ----------
    f_pipes, f_connectors: list of :py:class:`FrequentialComponent\
        <openwind.frequential.frequential_component.FrequentialComponent>`
        The list of the pipes and connectors in their frequential format.

    impedance : array of float
        The impedance at the entrance of the instrument (need to solve the \
        equations with :py:meth:`solve()<FrequentialSolver.solve()>` before)

    pressure, flow : array of float
        The pressure and flow along the instrument (need to solve the equations\
        with :py:meth:`solve(interp=True)<FrequentialSolver.solve()>` before)

    dpressure : array of float
        The spatial gradient of the pressure along the instrument (need to \
        solve the equations with
        :py:meth:`solve(interp_grad=True)<FrequentialSolver.solve()>` before)


    """

    FMIN_disc = 2000.0 # provides a mesh adapted to at least FMIN_disc Hz
    """
    float
    The minimal frequency in Hz, for which the mesh is adapted. The mesh is
    adapted to the frequency max([frequencies, FMIN_disc])
    """

    def __init__(self, instru_physics, frequencies, diffus_repr_var=False,
                 note=None, compute_method='FEM',
                 use_rad1dof=False,
                 **discr_params):
        self.netlist = instru_physics.netlist
        self.source_label = instru_physics.source_label

        # option
        self.discr_params = discr_params
        self.diffus_repr_var = diffus_repr_var   # When using 'diffrepr+'
        self.compute_method = compute_method     # FEM, TMM, modal or hybrid?
        if compute_method=='modal' and not use_rad1dof: # force use of radiation model with additional dof
            warnings.warn('With modal computation, the option "use_rad1dof=True" is necessary. It has been change automatically.')
            use_rad1dof=True
        self._use_rad1dof = use_rad1dof           # Additional dof for radiation: necessary for modal

        # set frequencial axis and options for automatic meshing
        frequencies = self._check_frequencies(frequencies, compute_method)
        self._update_shortestLbd(frequencies)
        self.frequencies = frequencies

        # discretize the netlist components and set their frequencial properties
        self._convert_frequential_components()

        # organize the components in the big matrix
        self._organize_components()

        # construct the matrix of the pipes (independant of the fingering)
        self._construct_matrices_pipes()

        # set the fingering
        self.note = note
        self._apply_note()

        # construct the matrix of the connectors
        self._construct_matrices_connectors()

    def __repr__(self):
        if len(self.frequencies) > 7:
            freq = ("array([{:.2e}, {:.2e}, ..., {:.2e}, "
                    "{:.2e}])".format(self.frequencies[0], self.frequencies[1],
                                     self.frequencies[-2], self.frequencies[-1]))
        else:
            freq = repr(self.frequencies)

        tmm_pipes = len([p for p in self.f_pipes
                         if p.__class__==FrequentialPipeTMM])
        fem_pipes = len(self.f_pipes) - tmm_pipes
        return ("<openwind.frequential.FrequentialSolver("
                "\n\tfrequencies={},".format(freq) +
                "\n\tnetlist={},".format(repr(self.netlist)) +
                "\n\tcompute_method='{:s}',".format(self.compute_method) +
                "\n\tnote='{}',".format(self.netlist.get_fingering_chart().get_current_note()) +
                "\n\tmesh_info:{{{} dof, {} TMM-pipes, {} FEM-pipes,"
                " elements/FEM-pipe:{}, "
                "order/element:{}}}\n)>".format(self.n_tot, tmm_pipes, fem_pipes,
                                          self.get_elements_mesh(),
                                          self.get_orders_mesh()))

    def __str__(self):
        if len(self.frequencies) > 7:
            freq = ("array([{:.2e}, {:.2e}, ..., {:.2e}, "
                    "{:.2e}])".format(self.frequencies[0], self.frequencies[1],
                                     self.frequencies[-2], self.frequencies[-1]))
        else:
            freq = repr(self.frequencies)
        return ("FrequentialSolver:\n" + "="*20 +
                "\nFrequencies:{}\n".format(freq) +"="*20 +
                "\n{}\n".format(self.netlist) + "="*20 +
                "\nCompute Method: '{:s}'\n".format(self.compute_method) + "="*20 +
                "\nCurrent Note: '{}'\n".format(self.netlist.get_fingering_chart().get_current_note()) + "="*20 +
                "\n" + self.__get_mesh_info())

    @property
    def imped(self):
        """
        The impedance, equivalent to :py:attr:`impedance<FrequentialSolver.impedance>`
        """
        return self.impedance

    def _update_shortestLbd(self, frequencies):
        """
        Update the value of the discretization parameter 'shortestLbd': the
        shortest wavelength to considered in case of automatic meshing, with
        respect to the value of the highest frequency
        """
        con, end = self.netlist.get_connector_and_ends(self.source_label)
        phy_entrance = end[0].pipe.get_physics()
        c_entrance = phy_entrance.get_coefs(0, 'c')[0]
        FMAX = np.max([np.max(frequencies), FrequentialSolver.FMIN_disc])
        if 'shortestLbd' not in self.discr_params.keys():
            self.discr_params['shortestLbd'] = c_entrance / FMAX
            is_updated = True
        elif FMAX > c_entrance/self.discr_params['shortestLbd']:
            self.discr_params['shortestLbd'] = c_entrance / FMAX
            warnings.warn('The shortest wave length used for automatic meshing'
                          ' has been changed to fit the highest frequency: '
                          'lambda = {:.2e} m'.format(self.discr_params['shortestLbd']))
            is_updated = True
        else:
            is_updated = False
        return is_updated

    @staticmethod
    def _check_frequencies(frequencies, compute_method):
        """
        Check that all the frequencies are positive and that it is an array

        In case of frequencies is a float or an int it is converted in array

        Returns
        -------
        np.array
        """
        if isinstance(frequencies, int) or isinstance(frequencies, float):
            frequencies = [frequencies]
        frequencies = np.array(frequencies)
        if np.any(frequencies <= 0):
            raise ValueError('The frequencies must be strictly positive!')
        if compute_method != 'TMM' and max(frequencies)/min(frequencies)>1000:
            warnings.warn('The frequency range is too big (fmax/fmin={:.0f}, advised: <1000). With FEM the use of'
                          'the same mesh for such range of frequencies can '
                          'induce numerical issue.'.format(max(frequencies)/min(frequencies)))
        return frequencies

    def _apply_note(self):
        if not self.note:
            return
        if isinstance(self.note, str):
            fc = self.netlist.get_fingering_chart()
            fingering = fc.fingering_of(self.note)
        elif isinstance(self.note, Fingering):
            fingering = self.note
        fingering.apply_to(self.f_components)

    def set_note(self, note):
        """
        Update the note (fingering) apply to the instrument.

        Parameters
        ----------
        note : str
            The note name. It must correspond to one of the associated
            :py:class:`FingeringChart<openwind.technical.fingering_chart.FingeringChart>`.

        """
        self.note = note
        self._apply_note()
        # Since solve() assumes the matrices are constructed,
        # update the matrices.
        self._construct_matrices_connectors()

    def set_frequencies(self, frequencies):
        """
        An overlay of update_frequencies_and_mesh()

        .. deprecated:: 0.8.1
            This method will be replaced by \
            :py:meth:`solve()<FrequentialSolver.update_frequencies_and_mesh()>` instead
        """
        warnings.warn('The method FrequentialSolver.set_frequencies() is deprecated,'
                      ' please use update_frequencies_and_mesh() instead.')
        self.update_frequencies_and_mesh(frequencies)


    def update_frequencies_and_mesh(self, frequencies):
        """
        Update the frequency axis and update the mesh if necessary

        Parameters
        ----------
        frequencies : array of float
            The new frequency axis.

        """
        self.frequencies = self._check_frequencies(frequencies, self.compute_method)
        need_update_mesh = self._update_shortestLbd(frequencies)
        # update mesh if the shortest wavelength has been modified
        if need_update_mesh:
            self._convert_frequential_components()
            self._organize_components()
            self._apply_note()
        if need_update_mesh or self.compute_method!='modal': # for FEM and TMM it is allways necessary to recompute matrices
            self._construct_matrices_pipes()
            self._construct_matrices_connectors()
        return need_update_mesh

    def _convert_pipe(self, pipe):
        """
        Construct an appropriate instance of (a subclass of) FrequentialPipe.

        Parameters
        ----------
        pipe : `Pipe <openwind.continuous.pipe.Pipe>`
            Continuous model of the pipe.
        **discr_params : keyword arguments
            Discretization parameters, passed to the FPipe initializer.

        Returns
        -------
        `FrequentialPipeFEM <openwind.frequential.frequential_pipe_fem.FrequentialPipeFEM>`
        OR `FrequentialPipeTMM <openwind.frequential.frequential_pipe_tmm.FrequentialPipeTMM>`

        """
        # only give to each pipe its corresponding disc value
        tmm_keys = {'nb_sub', 'reff_tmm_losses'}
        disc_keys = set(list(self.discr_params.keys()))
        params_fem = {k: self.discr_params[k] for k in disc_keys - tmm_keys}
        params_tmm = {k: self.discr_params[k] for k in disc_keys
                      if k in tmm_keys}
        if ('l_ele' in params_fem and isinstance(params_fem['l_ele'], dict)):
            dict_l_ele = params_fem['l_ele']
            params_fem['l_ele'] = dict_l_ele[pipe.label]
        if ('order' in params_fem and isinstance(params_fem['order'], dict)):
            # only give to each pipe its corresponding disc value
            dict_order = params_fem['order']
            params_fem['order'] = dict_order[pipe.label]
        if self.compute_method == 'FEM':
            if (self.diffus_repr_var and
                isinstance(pipe.get_losses(), ThermoviscousDiffusiveRepresentation)):
                return FrequentialPipeDiffusiveRepresentation(pipe, **params_fem)
            return FrequentialPipeFEM(pipe, **params_fem)
        elif self.compute_method == 'modal':
            # verify that the chosen model is compatible for modal
            # TODO Add test
            if(not pipe.is_compatible_for_modal()):
                raise ValueError(f"{pipe.get_losses()} model is not compatible with modal method. Consider using lossless (losses=False) or diffusive representation of Zwikker-Kosten model (losses=diffrepr+) instead.")
            if(isinstance(pipe.get_losses(), ThermoviscousDiffusiveRepresentation)):
                # forcing to extended diffusive representation formulation
                self.diffus_repr_var = True
                return FrequentialPipeDiffusiveRepresentation(pipe, **params_fem)
            return FrequentialPipeFEM(pipe, **params_fem)
        elif self.compute_method == 'TMM':
            return FrequentialPipeTMM(pipe, **params_tmm)
        elif self.compute_method == 'hybrid':
            # Use TMM when it is exact,
            # i.e. if the pipe is a cylinder,
            # or a lossless cone.
            # TODO Add test
            shape = pipe.get_shape()
            lossless = isinstance(pipe.get_losses(), ThermoviscousLossless)
            if isinstance(shape, Cone) and \
                (shape.is_cylinder() or lossless):
                return FrequentialPipeTMM(pipe, **params_tmm)
            return FrequentialPipeFEM(pipe, **params_fem)

        raise ValueError("compute_method must be in {'FEM', 'TMM', 'hybrid', 'modal'}")

    def _convert_connector(self, connector, ends):
        """
        Construct the appropriate frequential version of a connector.

        Parameters
        ----------
        connector : :py:class:`NetlistConnector<openwind.continuous.netlist.NetlistConnector>`
            Continuous model for radiation, junction, or source.
        ends : list(`FPipeEnd <openwind.frequential.frequential_pipe_fem.FPipeEnd>`)
            The list of all `FPipeEnd`s this component connects to.

        Returns
        -------
        `FrequentialComponent <openwind.frequential.frequential_component.FrequentialComponent>`

        """
        if isinstance(connector, Excitator):
            # verify that source is a flow
            if isinstance(connector, Flow):
                f_source = FrequentialSource(connector, ends)
            elif isinstance(connector, Flute):
                f_source = FrequentialFluteSource(connector, ends)
            else:
                raise ValueError('The input type of player must be flow for frequential computation')

            # Register the source to know on which d.o.f. to measure impedance
            if (hasattr(self, 'source_ref') and
                f_source.source.label != self.source_ref.source.label):
                raise ValueError('Instrument has several Sources (instead of one).')
            else:
                self.source_ref = f_source
            return f_source
        #   return(FrequentialSource(connector, ends))
        elif isinstance(connector, PhysicalRadiation):
            if isinstance(connector, RadiationPerfectlyOpen):
                return FrequentialPressureCondition(0, ends)
            elif isinstance(connector, RadiationPade) and self._use_rad1dof:
                return FrequentialRadiation1DOF(connector, ends)
            else:
                return FrequentialRadiation(connector, ends)
        elif isinstance(connector, JunctionTjoint):
            return FrequentialJunctionTjoint(connector, ends)
        elif isinstance(connector, JunctionSimple):
            return FrequentialJunctionSimple(connector, ends)
        elif isinstance(connector, JunctionDiscontinuity):
            return FrequentialJunctionDiscontinuity(connector, ends)
        elif isinstance(connector, JunctionSwitch):
            return FrequentialJunctionSwitch(connector, ends)

        raise ValueError("Could not convert connector %s" % str(connector))


    def _convert_frequential_components(self):

        self.f_pipes, self.f_connectors = \
            self.netlist.convert_with_structure(self._convert_pipe,
                                                self._convert_connector)

        self.f_components = self.f_pipes + self.f_connectors
        assert all([isinstance(f_comp, FrequentialComponent)
                    for f_comp in self.f_components])

        if(self.compute_method == 'modal'):
            # check if components are compatible
            compatib = [f_comp.is_compatible_for_modal()
                        for f_comp in self.f_components]
            if not all(compatib):
                [print(f"Chosen {f_comp} is not compatible for modal computation") for f_comp in self.f_components if not f_comp.is_compatible_for_modal()]
                raise ValueError()

        if not hasattr(self, 'source_ref'):
            raise ValueError('The input emplacement is not identified: '
                             'it is impossible to compute the impedance.')
        self.scaling = self.source_ref.get_scaling()


    def _organize_components(self):
        n_dof_cmpnts = self.get_dof_of_components()
        self.n_tot = sum(n_dof_cmpnts)
        # place the components
        beginning_index = np.zeros_like(self.f_components)
        beginning_index[1:] = np.cumsum(n_dof_cmpnts[:-1])
        for k, f_comp in enumerate(self.f_components):
            f_comp.set_first_index(beginning_index[k])
            f_comp.set_total_degrees_of_freedom(self.n_tot)

    def _construct_matrices_Mh_and_Kh_of(self, components):
        # initiate matrices
        n_tot = self.n_tot

        Mh = np.zeros(n_tot, dtype='float64')
        Kh_diag_row = []
        Kh_diag_col = []
        Kh_diag_data = []
        Eh_row = list()
        Eh_data = list()
        # fill the matrices
        for f_comp in components:
            ind_f, data_f = f_comp.get_contrib_Mh()
            Mh[ind_f] = data_f
            row, col, data = f_comp.get_contrib_Kh()
            Kh_diag_row=np.append(Kh_diag_row,row)
            Kh_diag_col=np.append(Kh_diag_col,col)
            Kh_diag_data=np.append(Kh_diag_data,data)
            source_row, source_data = f_comp.get_contrib_source()
            Eh_row.append(source_row)
            Eh_data.append(source_data)


        Eh_row_array = np.concatenate(Eh_row)
        Eh_data_array = np.concatenate(Eh_data)


        Kh = csr_matrix((Kh_diag_data, (Kh_diag_row, Kh_diag_col)),
                                    shape=(n_tot, n_tot), dtype='float64')


        Eh = csr_matrix((Eh_data_array, (Eh_row_array, np.zeros_like(Eh_row_array))),
                             shape = (n_tot, 1), dtype='float64')
        return Mh, Kh, Eh

    def _construct_matrices_of(self, components):
        omegas_scaled = 2*np.pi*self.frequencies * self.scaling.get_time()
        # initiate matrices and list of col/row indices and data for sparse matrices
        n_tot = self.n_tot
        no_diag_row = list()
        no_diag_col = list()
        no_diag_data = list()
        Lh_row = list()
        Lh_data = list()
        Ah_comp_diags = np.zeros((n_tot, len(omegas_scaled)), dtype='complex128')

        # fill the matrices
        for f_comp in components:
            row, col, data = f_comp.get_contrib_indep_freq()
            no_diag_row.append(row)
            no_diag_col.append(col)
            no_diag_data.append(data)
            ind_f, data_f = f_comp.get_contrib_freq(omegas_scaled)
            Ah_comp_diags[ind_f, :] = data_f

            source_row, source_data = f_comp.get_contrib_source()
            Lh_row.append(source_row)
            Lh_data.append(source_data)

        nodiag_row_array = np.concatenate(no_diag_row)
        nodiag_col_array = np.concatenate(no_diag_col)
        nodiag_data_array = np.concatenate(no_diag_data)
        Lh_row_array = np.concatenate(Lh_row)
        Lh_data_array = np.concatenate(Lh_data)
        # construct sparse matrices from indices and data
        Ah_comp_nodiag = csr_matrix((nodiag_data_array, (nodiag_row_array, nodiag_col_array)),
                                    shape=(n_tot, n_tot), dtype='complex128')
        Lh_comp = csr_matrix((Lh_data_array, (Lh_row_array, np.zeros_like(Lh_row_array))),
                             shape = (n_tot, 1), dtype='complex128')
        # Transfer the diagonal of Ah_nodiag onto Ah_diags
        # so that the diagonal data of Ah_nodiag can be replaced
        # by each column of Ah_diags
        Ah_comp_diags[:, :] += Ah_comp_nodiag.diagonal()[:, np.newaxis]
        return Ah_comp_nodiag, Ah_comp_diags, Lh_comp

    def _construct_matrices_pipes(self):
        if(self.compute_method=='modal'):
            Mh, Kh, Lh = self._construct_matrices_Mh_and_Kh_of(self.f_pipes)
            self.Mh_pipes = Mh
            self.Kh_pipes = Kh
            self.Eh_pipes = Lh
        else:
            nodiag, diag, Lh = self._construct_matrices_of(self.f_pipes)
            self.Ah_pipes_nodiag = nodiag
            self.Ah_pipes_diags = diag
            self.Lh_pipes = Lh



    def _construct_matrices_connectors(self):
        if(self.compute_method=='modal'):
            Mh_co, Kh_co, Eh_co = self._construct_matrices_Mh_and_Kh_of(self.f_connectors)
            self.Mh = self.Mh_pipes + Mh_co
            self.Kh = self.Kh_pipes + Kh_co
            self.Eh = self.Eh_pipes + Eh_co
        else:
            (Ah_conect_nodiag, Ah_conect_diags,
             Lh_conect) = self._construct_matrices_of(self.f_connectors)
            self.Ah_nodiag = self.Ah_pipes_nodiag + Ah_conect_nodiag
            self.Ah_diags = self.Ah_pipes_diags + Ah_conect_diags
            self.Lh = self.Lh_pipes + Lh_conect



    def get_dof_of_components(self):
        """
        The degree of freedom of each component constituing the frequential graph

        Returns
        -------
        n_dof_cmpts : list of int

        """
        n_dof_cmpts = np.array([f_cmpnt.get_number_dof()
                                for f_cmpnt in self.f_components], dtype='int')
        return n_dof_cmpts

    def solve_FEM(self, interp=False, pipes_label='main_bore', interp_grad=False,
                  interp_grid='original', observe_radiation=False,
                  enable_tracker_display=False):
        """
        An overlay of solve()

        .. deprecated:: 0.5
            This method will be replaced by \
            :py:meth:`solve()<FrequentialSolver.solve()>` instead
        """
        warnings.warn('The method FrequentialSolver.solve_FEM() is deprecated,'
                      ' please use solve() instead.')
        self.solve(interp, pipes_label, interp_grad, interp_grid,
                   observe_radiation, enable_tracker_display)


    def solve_with_method_modal(self):
        """
        Solve the acoustic equations with the modal solver.

        Parameters
        ----------

        None

        """

        Mh =  self.Mh
        Kh = -self.Kh
        Eh =  self.Eh

        # static elimination first
        # indice of the lines that must be tackled
        elimination_todo = False
        Mh_ = np.array(np.diag(Mh))
        ind_zero_mass = np.where(Mh==0)[0]
        continuons = len(ind_zero_mass)
        if(continuons>0):
            # print(f"Mass matrix vanishes at indices {ind_zero_mass}, trying static elimination")
            elimination_todo = True

            K_elim = ssp.csr_matrix(Kh)
            M_elim = ssp.csr_matrix(Mh_)
            E_elim = Eh

        while continuons:
            n_row, n_col = K_elim.shape
            K_elim = K_elim.tolil()
            M_elim = M_elim.tolil()
            i = ind_zero_mass[0]
            Kii = K_elim[i,i]
            if(Kii!=0):
                # if Kii non zero we eliminate the aux variable Xi and that's it.
                ligne_K_memory = K_elim[i,:].copy()
                ligne_M_memory = M_elim[i,:].copy()
                Ei = E_elim[i]
                for j in np.arange(0,n_row):
                    M_elim[j,:] = Kii*M_elim[j,:] - K_elim[j,i]*ligne_M_memory
                    E_elim[j]   = Kii*E_elim[j]   - K_elim[j,i]*Ei
                    K_elim[j,:] = Kii*K_elim[j,:] - K_elim[j,i]*ligne_K_memory
                K_elim = delete_from_csr(K_elim, rows=[i], cols=[i])
                M_elim = delete_from_csr(M_elim, rows=[i], cols=[i])
                Mh = np.delete(Mh,(i),axis=0)
                E_elim = np.delete(E_elim,(i),axis=0)
                ind_zero_mass = np.where(Mh==0)[0]
                continuons = len(ind_zero_mass)

            else:
                # toc = timeit.default_timer()
                #print(f"saddle point, we eliminate two ddl : the LM and the last ddl appearing in the constraint")
                col_i = K_elim[:,i]
                non_zero_col_number = col_i.transpose().rows[0]
                Nlambda = non_zero_col_number[-1]
                #print(f"we eliminate the LM : X_{i}")
                KNlambdai = K_elim[Nlambda,i]
                ligne_K_memory = K_elim[Nlambda,:].copy()
                ligne_M_memory = M_elim[Nlambda,:].copy()
                ENlambda = E_elim[Nlambda]
                colonne = ssp.coo_matrix(K_elim[:,i])
                for j in colonne.row:
                    M_elim[j,:] = KNlambdai*M_elim[j,:] - K_elim[j,i]*ligne_M_memory
                    E_elim[j]   = KNlambdai*E_elim[j]   - K_elim[j,i]*ENlambda
                    K_elim[j,:] = KNlambdai*K_elim[j,:] - K_elim[j,i]*ligne_K_memory
                Li = K_elim[i,:]
                non_zero_col_number = Li.rows[0]
                Ni = non_zero_col_number[-1]
                #print(f"we eliminate the last ddl appearing in the constraint : X_{Ni}")
                MiNi = M_elim[i,Ni]
                KiNi = K_elim[i,Ni]
                ligne_K_memory = K_elim[i,:].copy()
                ligne_M_memory = M_elim[i,:].copy()
                Ei = E_elim[i]
                colonne = ssp.coo_matrix(K_elim[:,Ni])
                for j in colonne.row:
                    # this line uses the time derivative of the constraint equation
                    M_elim[j,:] = KiNi*M_elim[j,:] - M_elim[j,Ni]*ligne_K_memory
                    # no mistake : we really add the K line to M, hence the cross term of constants to eliminate the variable X_Ni
                    E_elim[j]   = KiNi*E_elim[j]   - K_elim[j,Ni]*Ei
                    K_elim[j,:] = KiNi*K_elim[j,:] - K_elim[j,Ni]*ligne_K_memory
                #print(f"we eliminate the obsolete lines and columns")
                K_elim = delete_from_csr(K_elim, rows=[i,Ni], cols=[i,Ni])
                M_elim = delete_from_csr(M_elim, rows=[i,Ni], cols=[i,Ni])

                Mh = np.delete(Mh,(i),axis=0)
                Mh = np.delete(Mh,(Ni),axis=0)
                E_elim = delete_from_csr(E_elim,rows=[i,Ni])
                ind_zero_mass = np.where(Mh==0)[0]
                #print(f"mass is zero at indices {ind_zero_mass}")
                continuons = len(ind_zero_mass)

        if(elimination_todo):
            Mh_ = M_elim.toarray() # for use of LA.inv
            Kh = K_elim
            Eh = E_elim
            # tuc = timeit.default_timer()
            # print(f"Time spent eliminating : {tuc-toc:2.3f} sec")


        iMh = LA.inv(Mh_)
        eigenval, eigenvect = LA.eig(iMh@Kh)
        iPh = LA.inv(eigenvect)
        Fh = np.dot(iPh,iMh@Eh)
        Ah = Eh.T@eigenvect
        # pointwise product
        C = np.multiply(Ah,Fh.T)
        omegas = 2*np.pi*self.frequencies * self.scaling.get_time()
        self.impedance = self.scaling.get_impedance() * np.array([np.sum(C/(1j*omega - eigenval)) for omega in omegas])
        self._C = C
        self._eigenval = eigenval

        # for output: keeping only positive resonance frequencies above 1Hz
        ind = eigenval.imag>(2*np.pi*self.scaling.get_time())
        vp  = eigenval[ind]
        omega = np.abs(eigenval[ind])
        dissip = -eigenval[ind].real
        # Q and freqs as defined in [Chab_Modal]
        Q = omega/(2*dissip)
        freqs = omega/(2*np.pi)
        coefs = C[0,ind]

        sort_index = np.argsort(freqs)
        self.eigen_frequencies     = freqs[sort_index] / self.scaling.get_time()
        self.eigen_quality_factors = Q[sort_index]
        omegas = 2*np.pi*self.eigen_frequencies * self.scaling.get_time()
        self.Zstars = self.scaling.get_impedance() * np.array([np.sum(C/(1j*omega - eigenval)) for omega in omegas])


    def solve_with_method_direct(self, interp=False, pipes_label='main_bore', interp_grad=False,
              interp_grid='original', enable_tracker_display=False):
        """
        Solve the acoustic equations with the direct solver.

        It gives access to :py:attr:`impedance<FrequentialSolver.impedance>`

        Parameters
        ----------
        interp : bool, optional
            to interpolate the acoustic fields on some given points along the \
            instrument (necessary to have access to \
            :py:attr:`pressure<FrequentialSolver.pressure>` and \
            :py:attr:`flow<FrequentialSolver.flow>`). Default is False.
        pipes_label : str or list of str, optional
            The label of the pipes on which interpolate. The default is "main_bore"
            and correspond to an interpolation on all the pipes of the main bore.
            Used only if interp=True or interp_grad=True.
        interp_grad : bool, optional
            to interpolate the gradient of pressure along the instrument
            (necessary to have access to
            :py:attr:`dpressure<FrequentialSolver.dpressure>`). Default is False
        interp_grid : {float, array(float), 'original', 'radiation'}, optional
            Determine the point on which are computed the interpolated data.
            you can give either

            - a list of points on which to interpolate
            - a float which is the step of the interpolation grid
            - 'original'{Default}, if you want to keep the GaussLobato grid
            - 'radiation' if you want to observe acoustic flow at the radiation opening

        enable_tracker_display: bool, optional
            Display simulation tracker to give some indication on the resting
            computation time. Defaults to False

        """
        ind_source = self.source_ref.get_source_index()
        entrance_H1 = np.empty(self.frequencies.shape, dtype=np.complex128)

        tracker = SimulationTracker(len(self.frequencies),
                                    display_enabled=enable_tracker_display)
        # init save data
        if interp or interp_grad:
            interpolation = FrequentialInterpolation(self, pipes_label, interp_grid)
            self.x_interp = interpolation.x_interp
            interp_H1 = list()
            interp_L2 = list()
            interp_gradH1 = list()
        Ah, ind_diag = self._initialize_Ah_diag()
        # the loop on the frequency
        for cpt in range(len(self.frequencies)):
            # the frequency dependant part of Ah is only on its diagonal
            # Ah.setdiag(self.Ah_diags[:, cpt])
            Ah.data[ind_diag] = self.Ah_diags[:, cpt]
            # solve the problem
            Uh = spsolve(Ah, self.Lh, permc_spec='NATURAL')
            tracker.update()
            # save the right data
            entrance_H1[cpt] = Uh[ind_source]
            if interp:
                interp_H1.append(interpolation.interpolate_H1(Uh))
                interp_L2.append(interpolation.interpolate_L2(Uh))
            if interp_grad:
                interp_gradH1.append(interpolation.interpolate_gradH1(Uh))

        # rescale data
        convention = self.source_ref.get_convention()
        if convention == 'PH1':
            if self.source_ref.is_flute_like(): # for flute like, Z=1/u, and "ind_source" corresponds to the Lagr. coeff
                self.impedance = self.scaling.get_impedance()/entrance_H1
            else:
                self.impedance = self.scaling.get_impedance() * entrance_H1
            if interp:
                self.pressure = np.array(interp_H1) * self.scaling.get_scaling_pressure()
                self.flow = np.array(interp_L2) * self.scaling.get_scaling_flow()
            if interp_grad:
                self.dpressure = np.array(interp_gradH1) * self.scaling.get_scaling_pressure()
        elif convention == 'VH1':
            self.impedance = self.scaling.get_impedance() / entrance_H1
            if interp:
                self.pressure = np.array(interp_L2) * self.scaling.get_scaling_pressure()
                self.flow = np.array(interp_H1) * self.scaling.get_scaling_flow()

    def solve(self, interp=False, pipes_label='main_bore', interp_grad=False,
              interp_grid='original', enable_tracker_display=False):
        """
        Solve the acoustic equations.

        It gives access to :py:attr:`impedance<FrequentialSolver.impedance>`

        Parameters
        ----------
        interp : bool, optional
            to interpolate the acoustic fields on some given points along the \
            instrument (necessary to have access to \
            :py:attr:`pressure<FrequentialSolver.pressure>` and \
            :py:attr:`flow<FrequentialSolver.flow>`). Default is False.
        pipes_label : str or list of str, optional
            The label of the pipes on which interpolate. The default is "main_bore"
            and correspond to an interpolation on all the pipes of the main bore.
            Used only if interp=True or interp_grad=True.
        interp_grad : bool, optional
            to interpolate the gradient of pressure along the instrument
            (necessary to have access to
            :py:attr:`dpressure<FrequentialSolver.dpressure>`). Default is False
        interp_grid : {float, array(float), 'original', 'radiation'}, optional
            Determine the point on which are computed the interpolated data.
            you can give either

            - a list of points on which to interpolate
            - a float which is the step of the interpolation grid
            - 'original'{Default}, if you want to keep the GaussLobato grid
            - 'radiation' if you want to observe acoustic flow at the radiation opening

        enable_tracker_display: bool, optional
            Display simulation tracker to give some indication on the resting
            computation time. Defaults to False

        """


        if(self.compute_method == 'modal'):
            self.solve_with_method_modal()
            if interp:
                warnings.warn('The interpolation is not yet implemented for modal computation.')
            if enable_tracker_display:
                warnings.warn('Tracker display, is not available for modal computation.')
        else:
            self.solve_with_method_direct(interp=interp, pipes_label=pipes_label,
                                         interp_grad=interp_grad, interp_grid=interp_grid,
                                         enable_tracker_display=enable_tracker_display)

    def _initialize_Ah_diag(self):
        """
        Initialize diag of the sparse matrix Ah and return the corresponding indices

        for numerical purpose, it is interesting to initiate diag values and
        get the corresponding index in the "data" vector

        Returns
        -------
        Ah : csr sparse matrix
            The matrix Ah with abritrary values in diagonal
        ind_diag : array of int
            The indices corresponding to the diagonal values in the "data" attributes
            of the sparse matrix.
        """
        if not all(np.isfinite(self.Ah_nodiag.data)):
            raise ValueError('The matrix Ah contains non-finite value(s) (inf or NaN).')
        Ah = self.Ah_nodiag.tocsc()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SparseEfficiencyWarning)
            Ah.setdiag(np.NaN)
        ind_diag = np.where(np.isnan(Ah.data))[0]

        self.Ah_diags = np.array(self.Ah_diags) # Make sure it is an array (and not a matrix)
        return Ah, ind_diag


    # %% several notes

    def get_flow_pressure_several_notes(self, notes, f_interest, interp_grid='original'):
        """
        Compute the acoustic fields for several notes.

        Parameters
        ----------
        notes : list of str
            The note names.
        f_interest : list of array
            The interesting frequencies for each notes (the same number of
            frequencies must be given for each note)
        interp_grid : {float, array(float), 'original'}, optional
            Determine the point on which are computed the interpolated data.
            you can give either

            - a list of points on which to interpolate
            - a float which is the step of the interpolation grid
            - 'original'{Default}, if you want to keep the GaussLobato grid
            - 'radiation' if you want to observe acoustic flow at the radiation opening

        Returns
        -------
        flow_notes , pressure_notes : 3D-array
            The flow and pressure for each frequency, for each interpolation \
            point and each note

        """
        assert len(notes) == len(f_interest)
        flow_notes = list()
        pressure_notes = list()

        for note, freq in zip(notes, f_interest):
            self.update_frequencies_and_mesh(freq)
            self.set_note(note)
            self.solve(interp=True, interp_grid=interp_grid)
            flow_notes.append(self.flow)
            pressure_notes.append(self.pressure)

        self.flow_notes = np.array(flow_notes)
        self.pressure_notes = np.array(pressure_notes)
        return flow_notes, pressure_notes

    def impedance_several_notes(self, notes):
        """
        Compute the impedance for several notes

        Parameters
        ----------
        notes : list of str
            The note names.

        Returns
        -------
        impedances : list of array
            The list of the impedance corresponding to each note.

        """
        impedances = list()
        for note in notes:
            self.set_note(note)
            self.solve()
            impedances.append(self.impedance)
        return impedances


    # %% output
    def get_ZC_adim(self):
        """
        The caracteristic impedance at the entrance of the instrument

        .. math::
            Z_c = \\frac{\\rho c}{S_0}

        Returns
        -------
        float

        """
        return self.source_ref.get_Zc0()


    # %% --- Plotting functions ---

    def plot_flow_at_freq(self, freq, **kwargs):
        """
        Plot the acoustic flow for a given frequency inside the instrument.

        It correspond to the interpolation grid used in
        :py:meth:`solve()<FrequentialSolver.solve()>`.

        Parameters
        ----------
        freq : float
            The expected frequency (it uses the first higher frequency computed).
        **kwargs : key word arguments
            Passed to `plt.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.

        """
        ifreq = np.where(self.frequencies >= freq)[0][0]
        plt.plot(self.x_interp, np.real(self.flow[ifreq]), **kwargs)
        plt.xlabel('Position (m)')
        plt.ylabel('Flow (m/s)')
        plt.legend()

    def plot_pressure_at_freq(self, freq, **kwargs):
        """
        Plot the acoustic pressure for a given frequency inside the instrument.

        It correspond to the interpolation grid used in
        :py:meth:`solve()<FrequentialSolver.solve()>`.

        Parameters
        ----------
        freq : float
            The expected frequency (it uses the first higher frequency computed).
        **kwargs : key word arguments
            Passed to `plt.plot() <https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html>`_.

        """
        ifreq = np.where(self.frequencies >= freq)[0][0]
        plt.plot(self.x_interp, np.real(self.pressure[ifreq]), **kwargs)
        plt.xlabel('Position (m)')
        plt.ylabel('Pressure (Pa)')
        plt.legend()

    def plot_impedance(self, normalize=True, **kwargs):
        """
        Plot the normalized impedance.

        It uses :py:func:`openwind.impedance_tools.plot_impedance`

        Parameters
        ----------
        normalize : bool, optional
            Normalize or not the impedance by the input characteristic
            impedance. The default is True.
        **kwargs : keyword arguments
            They are transmitted to :py:func:`plot_impedance()\
            <openwind.impedance_tools.plot_impedance>`.

        """
        if normalize:
            Zc = self.get_ZC_adim()
        else:
            Zc=1

        tools.plot_impedance(self.frequencies, self.impedance,
                             Zc, **kwargs)

    def plot_admittance(self, normalize=True, **kwargs):
        """
        Plot the normalized admittance.

        It uses :py:func:`openwind.impedance_tools.plot_impedance`

        Parameters
        ----------
        normalize : bool, optional
            Normalize or not the admittance by the input characteristic
            impedance. The default is True.
        **kwargs : keyword arguments
            They are transmitted to :py:func:`plot_impedance()\
            <openwind.impedance_tools.plot_impedance>`.

        """
        if normalize:
            Zc = self.get_ZC_adim()
        else:
            Zc=1

        tools.plot_impedance(self.frequencies, self.impedance, Zc, admittance=True,
                             **kwargs)

    def plot_var3D(self, dbscale=True, var='pressure', with_plotly=False, **kwargs):
        """
        Plot one acoustic field in the instrument for every
        frequency on a surface.

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
        X = self.x_interp
        Y = self.frequencies
        if var == 'pressure':
            Z = self.pressure

        elif var == 'flow':
            Z = self.flow

        else:
            raise ValueError("possible values are pressure or flow")
        if dbscale:
            Zplot = 20*np.log10(np.abs(Z))
            label = '{} [dB]'.format(var)
        else:
            Zplot = np.real(Z)
            label = 'Real({})'.format(var)

        if with_plotly:
            self._plot_var3d_plotly(X, Y, Zplot, label)
        else:
            self._plot_var3d_pcolor(X, Y, Zplot, label, **kwargs)

    @staticmethod
    def _plot_var3d_plotly(X, Y, Z, label):
        """Plot with a 3D surface using plotly"""

        filename = label + '_3D.html'
        try:
            import plotly.graph_objs as go
            import plotly.offline as py
        except ImportError as err:
            msg = "option with_plotly requires plotly."
            raise ImportError(msg) from err
        try:
            layout_3D = go.Layout(scene=dict(xaxis=dict(title='Position (m)',
                                                        autorange='reversed'),
                                             yaxis=dict(title='Frequency (Hz)',
                                                        autorange='reversed'),
                                             zaxis=dict(title=label)))

            data_u3D = [go.Surface(x=X, y=Y, z=Z,
                                   contours=go.surface.Contours(
                                           x=go.surface.contours.X(
                                                   highlightcolor="#42f462",
                                                   project=dict(x=True)),
                                           y=go.surface.contours.Y(
                                                   highlightcolor="#42f462",
                                                   project=dict(y=True))
                                                                 )
                                   )
                        ]
            fig_u3D = go.Figure(data=data_u3D, layout=layout_3D)
            py.plot(fig_u3D, filename=filename)
        except:
            print('Impossible to load plotly: no 3D plot')


    @staticmethod
    def _plot_var3d_pcolor(X, Y, Z, label, **kwargs):
        plt.figure()
        im = plt.pcolor(X, Y, Z, shading='auto', **kwargs)
        plt.colorbar(im, label=label)
        plt.xlabel('Position (m)')
        plt.ylabel('Frequency (Hz)')

    def plot_norm_ac_fields_at_notes(self, notes, variable='power', logscale=False,
                                     scaled=False, **kwargs):
        """
        A map plotting the norm of an acoustic quantity at each interpolation
        point for each note.

        The acoustic quantity can be the flow, the pressure or the power (flow
        time pressure).

        Parameters
        ----------
        notes : list of str
            the notes name.
        variable : {'flow', 'pressure', 'power'}, optional
            The acoustic quantity which is plotted. The default is 'power'.
        logscale : bool, optional
            If true the color scale is logarithmic. The default is False.
        scaled : bool, optional
            If true, the acoustic quantity is scaled fingering, by fingering.
            The default is False.
        **kwargs : keyword arguments
            Keyword givent to :py:meth:`matplotlib.pyplot.imshow`

        """
        if variable == 'flow':
            ac_field = np.linalg.norm(self.flow_notes, axis=1)
            title = 'Acoustic flow'
        elif variable == 'pressure':
            ac_field = np.linalg.norm(self.pressure_notes, axis=1)
            title = 'Acoustic pressure'
        elif variable == 'power':
            ac_field = np.linalg.norm(self.flow_notes*self.pressure_notes,
                                      axis=1)
            title = 'Acoustic power'

        if scaled:
            ac_field = (np.abs(ac_field)
                        / np.sum(np.abs(ac_field), 1)[:, np.newaxis])

        if logscale:
            Z = np.log10(ac_field.T)

        else:
            Z = ac_field.T

        fig_test, ax_test = plt.subplots()
        im = ax_test.imshow(Z, **kwargs)


        ax_test.set_xticks(np.arange(0, len(notes), 1))
        ax_test.set_xticks(np.arange(-.5, len(notes)+.5, 1), minor=True)
        ax_test.set_xlim(-.5, len(notes)-.5)
        ax_test.set_xticklabels(notes)

        fig_test.suptitle(title)
        ax_test.xaxis.tick_top()
        ax_test.grid(which='minor', color='k', linestyle='-', linewidth=1.5)

        divider = make_axes_locatable(ax_test)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        fig_test.colorbar(im, cax=cax)


    # %% --- Post-processing functions ---

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
        if normalize:
            impedance = self.impedance/self.get_ZC_adim()
        else:
            impedance = self.impedance
        tools.write_impedance(self.frequencies, impedance, filename, column_sep)

    def resonance_frequencies(self, k=5, display_warning=True):
        """
        The resonance frequencies of the impedance

        Depending of the solving method used, it uses the function :func:`openwind.impedance_tools.resonance_frequencies`

        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.

        Returns
        -------
        list of float

        """
        if(self.compute_method=='modal'):
            nb = np.min([len(self.eigen_frequencies),k])
            return self.eigen_frequencies[0:nb]
        else:
            return tools.resonance_frequencies(self.frequencies, self.impedance, k, display_warning)

    def resonance_peaks(self, k=5, display_warning=True):
        """
        The resonance frequencies, quality factors and values of the impedance

        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.

        Returns
        -------
        tuple of 3 lists
            The resonance frequencies (float)

            The quality factors (float)

            The impedance value at the resonance frequencies (complex)

        """
        if(self.compute_method=='modal'):
            nb = np.min([len(self.eigen_frequencies),k])
            return (self.eigen_frequencies[0:nb], self.eigen_quality_factors[0:nb], self.Zstars[0:nb])
        else:
            if display_warning:
                warnings.warn("With FEM and TMM, these quantities are estimated a posteriori. For finer results, please  consider solving with compute_method = 'modal'.")
            return tools.resonance_peaks_from_phase(self.frequencies, self.impedance, k, display_warning)

    def antiresonance_frequencies(self, k=5, display_warning=True):
        """
        The antiresonance frequencies of the impedance

        It uses the function :func:`openwind.impedance_tools.antiresonance_frequencies`

        Parameters
        ----------
        k : int, optional
            The number of resonance included. The default is 5.

        Returns
        -------
        list of float

        """
        return tools.antiresonance_frequencies(self.frequencies, self.impedance, k, display_warning)

    def antiresonance_peaks(self, k=5, display_warning=True):
        """
        The frequencies, quality factors and values of the impedance at the antiresonances


        Parameters
        ----------
        k : int, optional
            The number of resonances included. The default is 5.

        Returns
        -------
        tuple of 3 lists
            The antiresonance frequencies (float)

            The quality factors (float)

            The impedance value at the antiresonance frequencies (complex)

        """
        return tools.antiresonance_peaks_from_phase(self.frequencies, self.impedance, k, display_warning)


    def match_peaks_with_notes(self,concert_pitch_A=440, transposition = 0, k=5, display=False):
        """
        Matches resonance frequencies with notes frequencies in Hz, deviation in cents and notes names
        The user can specify a concert pitch and a transposing behavior for the instrument.

        Parameters
        ----------
        concert_pitch_A: float, optional
            Frequency of the concert A4, in Hz. Default value is 440 Hz.
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
           - The closest notes frequencies (float);
           - The deviation of the resonance frequencies (float)
           - The names of the closest notes, in the given concert pitch and transposition system (string)

        """
        f_ = self.resonance_frequencies(k)

        return tools.match_freqs_with_notes(f_,concert_pitch_A, transposition, display)

    def evaluate_impedance_at(self, freqs):
        """
        Re-evaluate the impedance at given frequencies freqs without updating the mesh

        .. warning::
            This method does not update the mesh. If you want to automatically update it use
            :meth:`FrequentialSolver.recompute_impedance_at()` instead

        Parameters
        ----------
        freqs : array
            Frequencies, in Hz.

        Returns
        -------
        array
            The values of the impedance at the frequencies freqs.

        """
        if max(freqs)>max(self.frequencies):
            warnings.warn('You specify higher frequencies than the original {:.2f}>{:.2f}'
                          'frequency range, for which the spatial discretisation was adjuted'
                          'to ensure a converged numerical solution. it can induce numerical issue.'.format( max(freqs), max(self.frequencies)))
        self.frequencies = self._check_frequencies(freqs, self.compute_method)
        if(self.compute_method == 'modal'):
            omegas = 2*np.pi*freqs
            self.impedance = self.scaling.get_impedance() * np.array([np.sum(self._C/(1j*omega - self._eigenval)) for omega in omegas])
            return self.impedance
        raise ValueError("Option not available for FEM and TMM computation methods. "
                         "Please use FrequentialSolver.recompute_impedance_at() or consider solving with compute_method = 'modal'.")

    def recompute_impedance_at(self, frequencies):
        """
        Recompute the impedance at the specified frequencies

        If necessary this method updates the mesh and resolve the entire problem.
        Following the compute method chosen and the difference with the precedent
        frequency range it can be a long computation.

        Parameters
        ----------
        frequencies : np.array
            The new frequency axis.

        Returns
        -------
        np.array
            The new impedance (stored also in :py:attr:`FrequentialSolver.impedance`).

        """
        updated_mesh = self.update_frequencies_and_mesh(frequencies)
        if updated_mesh or self.compute_method!='modal':
            self.solve()
        else:
            self.evaluate_impedance_at(frequencies)
        return self.impedance

    def get_lengths_pipes(self):
        """
        Returns
        -------
        list of float
            The length of each pipe (in meter)
        """
        return [f_pipe.pipe.get_length() for f_pipe in self.f_pipes]

    def get_orders_mesh(self):
        """
        Returns
        -------
        list of list of int
            The order of each elements of each pipe
        """
        return [f_pipe.mesh.get_orders().tolist() for f_pipe in self.f_pipes
                if f_pipe.__class__ != FrequentialPipeTMM]

    def get_elements_mesh(self):
        """
        Returns
        -------
        list of int
            The number of elements on each pipe.
        """
        return [len(x) for x in self.get_orders_mesh()]

    def __get_mesh_info(self):
        msg = "Mesh info:"
        msg += '\n\t{:d} degrees of freedom'.format(self.n_tot)
        msg += "\n\tpipes type: {}".format([t for t in self.f_pipes])
        lengths = self.get_lengths_pipes()
        msg += "\n\t{:d} pipes of length: {}".format(len(lengths), lengths)

        # Orders contains one sub-list for each pipe.
        orders = self.get_orders_mesh()
        elem_per_pipe = self.get_elements_mesh()
        msg += ('\n\t{} elements distributed on FEM-pipes '
                'as: {}'.format(sum(elem_per_pipe), elem_per_pipe))
        msg += '\n\tOrders on each element: {}'.format(orders)
        return msg

    def discretization_infos(self):
        """
        Information of the total mesh used to solve the problem.

        See Also
        --------
        :py:class:`Mesh <openwind.discretization.mesh.Mesh>`

        Returns
        -------
        str
        """
        print(self.__get_mesh_info())

def delete_from_csr(mat, rows=[], cols=[]):
    """
    Remove the rows and columns from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    mat = mat.tocsr()

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat
