
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

"""Module for the exitator classes"""

from abc import ABC, abstractmethod
import warnings
import numpy as np

from openwind.continuous import NetlistConnector, RadiationPade, radiation_model


def create_excitator(player, label, scaling, convention):
    """
    Instanciate the right excitator according to the player information

    Parameters
    ----------
    player : :py:class:`Player<openwind.technical.player.Player>`
        The object with the excitator information.
    label : str
        The label of the connector.
    scaling : :py:class: `Scaling <openwind.continuous.scaling.Scaling>`
        Nondimensionalization coefficients.
    convention : str, optional
        Must be one out of {'PH1', 'VH1'} \
        The convention used in this component. The default is 'PH1'.

    Returns
    -------
    :py:class:`Excitator<Excitator>`
        The rigth excitator object.

    """
    exc_type = player.excitator_type.lower()
    available_exc = [clss.__name__.lower() for clss in Excitator.__subclasses__()]
    if exc_type in available_exc:
        exc_class = [clss for clss in Excitator.__subclasses__() if clss.__name__.lower()==exc_type][0]
        return exc_class(player.control_parameters, label, scaling, convention)
    # if exc_type == "Flow":
    #     return Flow(player.control_parameters, label, scaling, convention)
    # elif exc_type == "Reed1dof":
    #     return Reed1dof(player.control_parameters, label, scaling, convention)
    # elif exc_type == "Reed1dof_scaled":
    #     return Reed1dof_Scaled(player.control_parameters, label, scaling, convention)
    # elif exc_type == "Flute":
    #     return Flute(player.control_parameters, label, scaling, convention)
    else:
        raise ValueError("Could not convert excitator type '{:s}'; please chose"
                         " between {}".format(exc_type, available_exc))

class ExcitatorParameter(ABC):
    """
    Abstract class for excitator parameter

    An excitator parameter can be any parameter asociated to the model of
    the excitator. It can be stationnary or evolving with respect to time.

    """
    @abstractmethod
    def get_value(self,t):
        """
        Method which returns the curve value for a given time t

        Parameters
        ----------
        t : float
            The time at which we want to get the curve value

        Returns
        -------
        float
            the value of the parameter at the time t
        """
        pass


class VariableExcitatorParameter(ExcitatorParameter):
    """
    Class for any excitator parameter which varies with respect to time.

    Parameters
    ----------
    curve : callable
        time dependant function. For example, you can use a curve from the
        module :py:mod:`Temporal Curve <openwind.technical.temporal_curves>`.

    """
    def __init__(self,curve):
        self._variable_value = curve

    def get_value(self,t):
        return self._variable_value(t)


class FixedExcitatorParameter(ExcitatorParameter):
    """
    Class for stationnary excitator parameter

    Parameters
    ----------
    curve : float
        Fixed value for the parameter
    """
    def __init__(self, curve):
        self._fixed_value = curve


    def get_value(self,t):
        return self._fixed_value


class Excitator(ABC, NetlistConnector):
    """
    One excitator, a source of the acoustic oscillations.

    It a component of a :py:class:`Netlist<openwind.continuous.netlist.Netlist>`
    which interacts with a :py:class:`PipeEnd<openwind.continuous.netlist.PipeEnd>`.

    The excitator is typically an oscillator coupled with the acoustic fields.
    It brings the energy into the acoustical system. It is generally described
    trough a set of ODE, the coefficients of which can vary with time.

    Parameters
    ----------
    control_parameters : dict
        Associate each controle parameter (coefficient of the ODE) to its value
        and the evolution with respect to time. It must have all the keys
        necessary to instanciate the right excitator.
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.
    """

    NEEDED_PARAMS = list()
    """list of str: Needed parameters to instanciate the excitator"""

    POSSIBLE_PARAMS = list()

    def __init__(self, control_parameters, label, scaling, convention):
        super().__init__(label, scaling, convention)
        self.check_needed_params(control_parameters)
        self._update_fields(control_parameters)

    @classmethod
    def check_needed_params(cls, control_parameters):
        if not set(cls.NEEDED_PARAMS) <= set(control_parameters):
            missing = list(set(cls.NEEDED_PARAMS) - set(control_parameters))
            raise ValueError(f'The following parameters are missing in "Player": {missing}')

    @abstractmethod
    def _update_fields(self, control_parameters):
        """
        This method update all the attributes according to the
        control_parameters fields attributes

        Parameters
        ----------
        control_parameters : dict
            Associate each controle parameter (coefficient of the ODE) to its value
            and the evolution with respect to time. It must have all the keys
            necessary to instanciate the right excitator.
        """

    @staticmethod
    def _create_parameter(curve):
        if callable(curve):
            return VariableExcitatorParameter(curve)
        else:
            return FixedExcitatorParameter(curve)


class Flow(Excitator):
    """
    Flow excitator: imposes the value of the flow at the connected pipe-end.

    This excitator simply imposes the flow at the pipe-end connected to it. The
    only parameter is the flow in m^3/s

    See Also
    --------
    :py:class:`FrequentialSource\
        <openwind.frequential.frequential_source.FrequentialSource>`
        The frequential version of this excitator (only has a sense for dirac flow)
    :py:class:`TemporalFlowCondition\
        <openwind.temporal.tflow_condition.TemporalFlowCondition>`
        The temporal version of this excitator

    Parameters
    ----------
    control_parameters : dict
        Associate each controle parameter (coefficient of the ODE) to its value
        and the evolution with respect to time. It must have only the key
        "input_flow".
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.

    Attributes
    ----------
    input_flow : float or callable
        The value in m^3/s of the flow at the entrance of the instrument or its
        evolution with time
    """
    NEEDED_PARAMS = ["input_flow"]
    POSSIBLE_PARAMS = NEEDED_PARAMS

    def _update_fields(self, control_parameters):
        self.input_flow = self._create_parameter(control_parameters["input_flow"])


class Flute(Excitator):
    """
    Flute-like excitator.

    The "jet-drive" model is used for flute-like instruments (recorders, organ
    pipes, transverse flutes, etc). The eqautions associated to this model
    can be found for examples in [auvray-jetdrive]_.
    This model needs some parameters values:

    - "jet_velocity": the velocity of the jet (often `Uj` in litterature) in [m/s],
      which can be directly deduced from the supply pressure using Bernoulli law
    - "width": the distance between the channel and the edge in [m] (`W` in lit.)
    - "channel_height": the height of the channel in[m] (`h` in lit.)
    - "convection": the ratio between the convection velocity of perturbation along the
      jet and the jet velocity (`Cp/Uj` in lit.)
    - "section": the cross section area of the embouchure opening (window) (`WH` in lit.)
    - "edge_offset": the vertical offset between the center of the channel and\
        the tip of the edge in [m] (`y0` in lit.)
    - "vena_contracta": the magnitude of the vena contracta phenomena\
      (`\\alpha_{vc}` in lit.), directly linked to the magnitude of the non-linear losses.
    - "gain": a global gain added to the source term. (by default, you can indicate 1).

    It is also necessary to specify the radiation impedance of the embouchure
    opening, witht the keyword "radiation_category". In addition to the classical
    ones with :py:func:`radiation_model() <openwind.continuous.radiation_model.radiation_model>`,
    it is possible to use a radiation specific to recorder-like instruments. In
    this case, the impedance is a `RadiationPade <openwind.continuous.physical_radiation.RadiationPade>`
    with coefficients computed with the model from [ernoult_window]_. The
    following geometric parameters are needed:

    - "edge_angle": The edge angle in degree.
    - "wall_thickness": the length of the "ears" or wall thickness aside the \
    edge in [m] (`le` in lit.).

    One other optional parameters can be specified:

    - "noise_level" : The noise added to the jet position (necessary to \
      iniciate the osciallation). The default is 1e-4.


    The other parameters of the jet-drive models are computed using default expression
    from the other control parameters (amplification rate of the jet, distance
    between dipole sources, jet height at the edge). They only influence the global gain of the
    source term which can be tuned trough the "gain" parameters. They are computed
    in :py:meth:`Flute.get_hidden_parameters()<.get_hidden_parameters>`

    Parameters
    ----------
    control_parameters : dict
        Associate each controle parameter to its value
        and the evolution with respect to time. It must have the keys
        `["jet_velocity", "width", "channel_height", "convection", "section",
        'edge_offset', "vena_contracta", "edge_angle", "wall_thickness"]`.
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.

    See Also
    --------
    :py:class:`FrequentialFluteSource <openwind.frequential.frequential_source.FrequentialFluteSource>`
        The version of this excitator in the frequency domain.
    :py:class:`TemporalFlute <openwind.temporal.tflute.TemporalFlute>`
        The version of this excitator in the time domain.

    References
    ----------
    .. [auvray-jetdrive] R. Auvray, A. Ernoult, B. Fabre and P.-Y. Lagrée 2014.
        "Time-domain simulation of flute-like instruments: Comparison of
        jet-drive and discrete-vortex models". JASA 136(1), p.389–400.
        https://hal.science/hal-01426971/document.

    .. [ernoult_window] A. Ernoult and B. Fabre 2017. "Window Impedance of
        Recorder-Like Instruments". Acta Acustica united with Acustica. 103(1),
        p.106–116. https://hal.sorbonne-universite.fr/hal-01430654.

    Attributes
    ----------
    jet_velocity : :py:class:`VariableExcitatorParameter`
        the velocity of the jet (often `Uj` in litterature) in [m/s], which can
        be directly deduced from the supply pressure using Bernoulli law
    width: :py:class:`FixedExcitatorParameter`
        the distance between the channel and the edge in [m] (`W` in lit.)
    channel_height : :py:class:`FixedExcitatorParameter`
        the height of the channel in[m] (`h` in lit.)
    convection : :py:class:`FixedExcitatorParameter`
        the ratio between the convection velocity of perturbation along the
        jet and the jet velocity (`Cp/Uj` in lit.)
    section : :py:class:`FixedExcitatorParameter`
        the cross section area of the embouchure opening (window) (`WH` in lit.)
    edge_offset : :py:class:`FixedExcitatorParameter`
        the vertical offset between the center of the channel and the tip of
        the edge in [m] (`y0` in lit.)
    loss_mag : :py:class:`FixedExcitatorParameter`
        the magnitude of the non-linear losses (vena contracta & vortex
        shedding effects)
    edge_angle : :py:class:`FixedExcitatorParameter`
        The edge angle in degree.
    wall_thickness : :py:class:`FixedExcitatorParameter`
        the length of the "ears" or wall thickness aside the edge in [m] (`le` in lit.).
    noise_level : :py:class:`FixedExcitatorParameter`
        The noise added to the jet position (necessary to iniciate the
        oscillation). The default is 1e-4.
    gain : :py:class:`FixedExcitatorParameter`
        a global gain added to the source term. Default is 1.
    radiation_category : string
        The radiation category used to model the radiation at the embouchure opening

    """

    NEEDED_PARAMS = ["jet_velocity", "width", "channel_height", "convection",
                     "section", 'edge_offset', "loss_mag", "gain",
                     "radiation_category"]
    POSSIBLE_PARAMS = NEEDED_PARAMS + ['edge_angle', 'wall_thickness', 'noise_level']

    def _update_fields(self, control_parameters):
        self.jet_velocity = self._create_parameter(control_parameters["jet_velocity"])

        # currently only constant value are accepted for other parameters
        # the numerical schemes are not compatible with varaible coefficients.
        self.width = self._create_cst_param(control_parameters["width"])
        self.channel_height = self._create_cst_param(control_parameters["channel_height"])
        self.convection = self._create_cst_param(control_parameters["convection"])
        self.section = self._create_cst_param(control_parameters["section"])
        self.edge_offset = self._create_cst_param(control_parameters["edge_offset"])
        self.loss_mag = self._create_cst_param(control_parameters["loss_mag"])
        self.gain = self._create_cst_param(control_parameters["gain"])

        self.radiation_category = control_parameters["radiation_category"]
        if self.radiation_category=="window":
            if "edge_angle" and "wall_thickness" in control_parameters:
                self.edge_angle = self._create_cst_param(control_parameters["edge_angle"])
                self.wall_thickness = self._create_cst_param(control_parameters["wall_thickness"])
                # self.face_angle = self._create_cst_param(control_parameters["face_angle"])
            else:
                raise ValueError('With the radiation type "window", it is '
                                 'necessary to specify the "edge_anlge" and the "wall_thickness"')
        else:
            self.edge_angle = None
            self.wall_thickness = None

        if "noise_level" in control_parameters:
            self.noise_level = self._create_parameter(control_parameters["noise_level"])
        else:
            # warnings.warn('By default, the noise level added to the jet position'
            #               ' (necessary to initiate the oscillation) is set to 1e-4'
            #               ' (relatively to the jet height).')
            self.noise_level = self._create_parameter(1e-4)

    @staticmethod
    def _create_cst_param(curve):
        if callable(curve):
            raise ValueError("Except 'jet_velocity', evolving parameters of "
                             "Flute Excitator are not supported yet, must be a"
                             " constant value")
        else:
            return FixedExcitatorParameter(curve)

    def get_hidden_parameters(self):
        r"""
        Compute the "hidden" parameters needed in jet drive model

        These parameters  are :math:`\alpha_i, b, \delta_d, \delta_j`. They
        are computed from control parameters with the following expressions:

        .. math::
            \begin{align}
            \alpha_i & = \frac{0.3}{h} \\
            b & = \frac{2}{5} h \\
            \delta_d & = \frac{4}{\pi} \sqrt{2hW} \\
            \delta_j & =1
            \end{align}


        Returns
        -------
        alpha_i : float
            the amplification rate of the jet.
        b : float
            The jet width at the edge location assuming a Bickley profile.
        delta_d : float
            Th distance between the flow sources of the dipole.
        delta_j : float, 1.
            eventual corrective factor for the jet receptivity.

        """
        h = self.channel_height.get_value(0)
        W = self.width.get_value(0)

        alpha_i = .3/h # amplification rate of the jet
        b = 2*h/5 # jet width at the edge location assuming a Bickley profile
        delta_d = 4/np.pi*np.sqrt(2*h*W) # distance between the flow sources of the dipole
        delta_j = 1 # corrective factor for the jet receptivity
        alpha_vc = .6
        return alpha_i, b, delta_d, delta_j, alpha_vc

    def get_scaled_model_parameters(self, celerity):
        r"""
        Compute coefficients for scaled jet-drive model

        .. math::
            \begin{align}
            A_{scale} & = \exp(\alpha_i W) \frac{\delta_j h}{b} \\
            G_j & = G A_{scale}  \frac{\delta_d b}{c_0 W} \\
            G_l & = \frac{1}{(2 c_0 A_{scale}\alpha_{vc}^2)} \\
            C & = \frac{W}{\gamma} = \frac{W U_j}{c_p}  \\
            y_0 & = \frac{\hat{y_0} }{ b }
            \end{align}

        with :math:`c_0` the speed of sound and :math:`G` the optional additional gain.

        Parameters
        ----------
        celerity : float
            The speed of sound.

        Returns
        -------
        Ascale, Gj, Gl, y0 : float
            The coefficients for dimensionless equations.

        """
        alpha_i, b, delta_d, delta_j, alpha_vc = self.get_hidden_parameters()
        h = self.channel_height.get_value(0)
        W = self.width.get_value(0)
        loss_mag = self.loss_mag.get_value(0)
        gain = self.gain.get_value(0)

        # exp_aiW = np.exp(alpha_i*W)

        Ascale = np.exp(alpha_i*W)*delta_j*h/b
        Gj = gain * Ascale*delta_d*b/(celerity*W)#gain * delta_d*delta_j*h/(celerity*W) * exp_aiW
        Gl = loss_mag/(2*celerity*Ascale*alpha_vc**2)#b/(2*celerity*delta_j*h*alpha_vc**2 * exp_aiW)

        C = W / self.convection.get_value(0)

        y0 = self.edge_offset.get_value(0) / b

        return Ascale, Gj, Gl, C, y0

    def get_equivalent_radius(self):
        r"""
        Compute the equaivalent radius of the opening

        :math:`r_w = \sqrt(S_w/\pi) = \sqrt{WH/\pi}`

        Returns
        -------
        float
            The equivalent radius

        """
        Sw = self.section.get_value(0)
        return np.sqrt(Sw/np.pi)

    def get_Uj_dUj(self, t, dt):
        r"""
        The jet velocity and an esimation of its time derivative at a given t.

        The time derivative is estimated using a centered finite difference

        .. math::
            \frac{dU_j}{dt} \approx \frac{U_j(t+\Delta t) - U_j(t-\Delta t) }{2\Delta t}

        Parameters
        ----------
        t : float
            The given time in s.
        dt : float
            The step time (eventually scaled, without dimension).

        Returns
        -------
        Uj : float
            The jet velocity
        dUj : float
            The time derivative of the jet velocity
        next_Uj : float
            The jet velocity at the next time step.

        """
        Uj = self.jet_velocity.get_value(t)
        dt_dim = dt*self.scaling.get_time()
        next_Uj = self.jet_velocity.get_value(t+dt_dim)
        last_Uj = self.jet_velocity.get_value(t-dt_dim)
        dUj = (next_Uj - last_Uj)/(2*dt_dim)
        return Uj, dUj, next_Uj

    def get_rad_model_window(self, Sp):
        """
        Generate the radiation model of the embouchure opening (window).

        Parameters
        ----------
        Sp : float
            The cross section are of the main pipe at the location of the embouchure, in [m].

        Returns
        -------
        :py:class:`RadiationPade <openwind.continuous.physical_radiation.RadiationPade>`
            The radiation object associated to the radiation of the window.

        """
        if self.radiation_category == "window":
            coefs = self.coefs_window_impedance(Sp)
            rad = RadiationPade(coefs, 'window_radiation', self.scaling,
                                convention=self.convention)
        else:
            rad = radiation_model(self.radiation_category, label="embouchure_radiation",
                                  scaling=self.scaling, convention=self.convention)
        return rad


    def coefs_window_impedance(self, Sp):
        r"""
        Pade dev. coefficients for radiation impedance of recorder's window

        The relationships between geoemtrical parameters and the coefficients come
        from [ernoult_window_meth]_
        The influence of the flutist face could be modelled thanks to [ernoult_inclined]_.

        The coefficients delta_c is related to the inertance by
        :math:`\delta_w = S_w/(\rho R_w) M_w`. The Taylor development coefficients are
        converted into Padé dev.

        Parameters
        ----------
        Sp : float
            The cross-section area of the main pipe in m².

        Returns
        -------
        alpha : float
            First coefficient of the Pade dev.
        beta : float
            Second coefficient of the Pade dev.

        References
        ----------
        .. [ernoult_window_meth] A. Ernoult and B. Fabre 2017. "Window Impedance of
            Recorder-Like Instruments". Acta Acustica united with Acustica. 103(1),
            p.106–116. https://hal.sorbonne-universite.fr/hal-01430654.

        .. [ernoult_inclined] A. Ernoult, P. de la Cuadra and B. Fabre 2018.
            "An Inclined Plane: A Simple Model for the Acoustic Influence of
            the Flutist’s Face". Acta Acustica united with Acustica. 104(3),
            p.496–508. https://hal.science/hal-01943166.

        """
        W = self.width.get_value(0)
        Sw = self.section.get_value(0)
        alpha = self.edge_angle.get_value(0)*np.pi/180
        le = self.wall_thickness.get_value(0)
        # face_angle = self.face_angle.get_value(0)*np.pi/180

        delta = Sw/Sp
        H = Sw/W
        Rw = np.sqrt(Sw/np.pi)

        delta_ra = 2*W/(np.pi*Rw)*(np.log((1 + delta**2/4)/(2*delta)) # right angle inertance
                                + 2/delta*(1-delta**2/4)*np.arctan(delta/2))
        delta_alpha = W/Rw*(np.tan(alpha)*np.log(1 + le/(W * np.tan(alpha)))
                         + 0.85/32**alpha * np.sqrt(le/H) ) #inertance due to angle
        delta_rad = 0.695  #+ .25*(1/np.tan(face_angle/4) - 1)  # radiation correction

        # Zc = rho*c/Sw

        delta_w = delta_ra + delta_alpha + delta_rad
        beta_w = .5 #np.pi/(2*face_angle)

        return 1/delta_w, beta_w/(delta_w**2)


class Reed1dof(Excitator):
    """
    Reed excitator (cane or lips): non-linear excitator

    A reed excitator is used to model both brass (lips-reeds) and woodwind
    (cane-reed) instruments. It can be, following [Fletcher]_ nomeclature, an:

    - outwards valve (lips-reeds) for brass: the reed opens when the supply \
    pressure is higher than the pressure inside the instrument
    - inwards valve (cane-reed) for simple or double reed instruments: the reed \
    opens when the supply pressure is smaller than the pressure inside the\
    instrument

    The reed is modelled by a 1D damped mass-spring oscillator following
    [Bilbao]_. The contact at closure is modelled through a penalizating term.

    .. math::
        \\begin{align}
        &\\ddot{y} + g \\dot{y} + \\omega_0^2 (y - y_0) - \
            \\frac{\\omega_1^{\\alpha+1}}{y_0^{\\alpha-1}}(|[y]^{-}|)^{\\alpha}\
            =  \\epsilon \\frac{S_r \\Delta p}{M_r} \\\\
        &\\Delta p = p_m - p \\\\
        & u = w [y]^{+} \\sqrt{\\frac{2 |\\Delta p|}{\\rho}} sign(\\Delta p) \
            \\epsilon S_r \\dot{y}
        \\end{align}

    With :math:`y` the position of the reed, :math:`[y]^{\\pm}` the positive
    or negative part of :math:`y`, :math:`p,u` the acoustic fields inside the
    instrument and :math:`\\rho` the air density.

    The other coefficients must be specified by the user. They are:

    - "opening" :math:`y_0` : the resting opening height of the reed (in m)
    - "mass" :math:`M_r` : the effective mass of the reed (in kg)
    - "section" :math:`S_r` : the effective vibrating surface of the reed  \
    (and not the opening) (in m²)
    - "pulsation" :math:`\\omega_0` : the resonant angular frequency of the reed
    - "dissip" :math:`g` : the damping coefficient
    - "width" :math:`w` : the effective width of the reed channel opening (in m)
    - "mouth_pressure" :math:`p_m` : the supply pressure (in Pa)
    - "model" : string coefficient that specifies if it is a "cane" (inwards)\
        or "lips" (inwards) reed. It fixes the value of :math:`\\epsilon` to \
        1 (cane) or -1 (lips)
    - "contact_pulsation" :math:`\\omega_1` : the angular frequency associated\
        to the contact law when the reed is closed.
    - "contact_exponent" :math:`\\alpha` : the exponent associated to the \
        contact law when the reed is closed.


    .. warning::
        This excitator can be used only in temporal domain

    See Also
    --------
    :py:class:`TemporalReed1dof\
        <openwind.temporal.treed.TemporalReed1dof>`
        The temporal version of this excitator

    References
    ----------
    .. [Bilbao] Bilbao, S. (2009). Direct simulation of reed wind instruments.\
        Computer Music Journal, 33(4), 43-55.
    .. [Fletcher] Fletcher, N H. 1979. “Excitation Mechanisms in Woodwind and \
        Brass Instruments.” Acustica 43: 10.


    Parameters
    ----------
    control_parameters : dict
        Associate each controle parameter (coefficient of the ODE) to its value
        and the evolution with respect to time. It must have the keys
        `["opening", "mass", "section", "pulsation", "dissip", "width",
        "mouth_pressure", "model", "contact_pulsation", "contact_exponent"]`.
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.

    Attributes
    ----------
    opening : float or callable
        :math:`y_0` : the resting opening height of the reed (in m)
    mass : float or callable
        :math:`M_r` : the effective mass of the reed (in kg)
    section : float or callable
        :math:`S_r` : the effective vibrating surface of the reed
        (and not the opening) (in m²)
    pulsation : float or callable
        :math:`\\omega_0` : the resonant angular frequency of the reed
    dissip : float or callable
        :math:`\\sigma` : the damping coefficient
    width : float or callable
        :math:`w` : the effective width of the reed channel opening (in m)
    mouth_pressure : float or callable
        :math:`p_m` : the supply pressure (in Pa)
    model : -1 or 1
        :math:`\\epsilon` the opening sens of the reed: 1 (cane,\
        inwards) or -1 (lips, outwards)
    contact_pulsation : float or callable
        :math:`\\omega_1` : the angular frequency associated to the contact \
        law when the reed is closed.
    contact_exponent : float or callable
        :math:`\\alpha` : the exponent associated to the contact law when the\
        reed is closed.
    """

    NEEDED_PARAMS = ["mouth_pressure", "opening",  "mass", "section",
                     "pulsation", "dissip", "width", "contact_pulsation",
                     "contact_exponent", "model"]
    POSSIBLE_PARAMS = NEEDED_PARAMS

    def _update_fields(self, control_parameters):

        self.mouth_pressure = self._create_parameter(control_parameters["mouth_pressure"])

        # currently only constant value are accepted for other parameters
        # the numerical schemes are not compatible with varaible coefficients.
        self.opening = self._create_cst_param(control_parameters["opening"])
        self.mass = self._create_cst_param(control_parameters["mass"])
        self.section = self._create_cst_param(control_parameters["section"])
        self.pulsation = self._create_cst_param(control_parameters["pulsation"])
        self.dissip = self._create_cst_param(control_parameters["dissip"])
        self.width = self._create_cst_param(control_parameters["width"])
        self.contact_pulsation =  self._create_cst_param(control_parameters["contact_pulsation"])
        self.contact_exponent =  self._create_cst_param(control_parameters["contact_exponent"])

        # the value of epsilon is set by interpreting the keyword in "model"
        model = control_parameters['model']
        if model in ["lips","outwards"]:
            model_value = 1
        elif model in ["cane","inwards"]:
            model_value= -1
        else:
            warnings.warn("WARNING : your model is %s, but it must "
                  "be in {'lips', 'outwards', 'cane', 'inwards'}, it "
                  "will be set to default (lips)"
                  %model)
            model_value = 1
        self.model = FixedExcitatorParameter(model_value)


    @staticmethod
    def _create_cst_param(curve):
        if callable(curve):
            raise ValueError("Except 'mouth_pressure', evolving parameters of "
                             "Reed1dof Excitator are not supported yet, must be a"
                             " constant value")
        else:
            return FixedExcitatorParameter(curve)


    def get_dimensionfull_values(self, t, Zc=1, rho=1):
        """
        Method that returns the values of all the dimensionfull parameters at a given time

        Parameters
        ----------
        t : int
            The given time we want to get the curves values from
        Zc : float, UNUSED
            The characteristic (real) impedance at the entry of the instrument.
        rho : float, UNUSED
            the air density.

        Returns
        -------
        Sr : float
            the effective cross section area of the reed (in m²)
        Mr: float
            the effecitve mass of the read (in kg)
        g: float
            the damping coefficient
        omega02: float
             the squared resonant angular frequency
        w: float
            the effective width of the reed channel opening (in m)
        y0: float
            the resting opening of the reed (in m)
        epsilon: float
            1 (for lips/outwards) or -1 (cane/inwards) following the model
        pm: float
            the supply pressure (in Pa)
        """
        Sr = self.section.get_value(t)
        Mr = self.mass.get_value(t)
        g = self.dissip.get_value(t)
        omega02 = self.pulsation.get_value(t)**2
        w = self.width.get_value(t)
        y0 = self.opening.get_value(t)
        epsilon = self.model.get_value(t)
        pm= self.mouth_pressure.get_value(t)

        omega_c = self.contact_pulsation.get_value(t)
        alpha_c = self.contact_exponent.get_value(t)

        return (Sr, Mr, g, omega02, w, y0, epsilon, pm, omega_c, alpha_c)

    def get_dimensionless_values(self, t, Zc, rho):
        """
        Return the values of all dimensionless parameter at a given time:

        Parameters
        ----------
        t : float
            The given time we want to get the curves values from.
        Zc : float
            The characteristic (real) impedance at the entry of the instrument.
        rho : float
            the air density.

        Returns
        -------
        gamma : float
            Dimensionless parameter relative to the supply pressure.
        zeta : float
            Dimensionless parameter relative to the reed opening.
        kappa : float
            The dimensionless parameter relative to the "reed flow".
        Qr : float
            the quality factor of the ree.
        omegar : float
            The resonant angular frequency of the reed in rad/s.
        Kc : float
            The contact force stifness.
        alpha_c : TYPE
            DESCRIPTION.
        epsilon : float
            1 (for lips/outwards) or -1 (cane/inwards) following the model.

        """
        (Sr, Mr, g, omega02, w, y0, epsilon, pm, omega_nl, alpha_c) = self.get_dimensionfull_values(t)

        omegar = np.sqrt(omega02)
        Kc = omega_nl**(alpha_c+1)/omega02
        Qr = omegar/g

        Pclosed = self.get_Pclosed(t)

        gamma = pm/Pclosed
        zeta = Zc*y0*w*np.sqrt(2/(rho*Pclosed))
        kappa = y0*Zc*Sr*omegar/Pclosed
        return gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon

    def get_Pclosed(self, t):
        r"""
        Return the closing pressure of the reed.

        This value does not modify the response of the system. It is only necessary
        to rescale the results of the equations.

        .. math::
            P_{\text{closed}} = \frac{K_r }{S_r}  y_0 =  \omega_r^2 \frac{M_r}{S_r} y_0

        Parameters
        ----------
        t : float
            The given time we want to get the curves values from.

        Returns
        -------
        Pclosed : float
            The closing pressure in Pa.

        """
        Sr = self.section.get_value(t)
        Mr = self.mass.get_value(t)
        omegar = self.pulsation.get_value(t)
        y0 = self.opening.get_value(t)
        Kr = omegar**2*Mr # the reed stiffness
        Pclosed = Kr*y0/Sr
        return Pclosed



class Reed1dof_Scaled(Excitator):
    r"""
    Reed excitator (cane or lips): non-linear excitator with dimensionless parameters

    A reed excitator is used to model both brass (lips-reeds) and woodwind
    (cane-reed) instruments. It can be, following [Fletcher_sclaled]_ nomenclature, an:

    - outwards valve (lips-reeds) for brass: the reed opens when the supply\
      pressure is higher than the pressure inside the instrument
    - inwards valve (cane-reed) for simple or double reed instruments: the reed\
      opens when the supply pressure is smaller than the pressure inside the instrument

    The reed is modelled by a 1D damped mass-spring oscillator following
    [Bilbao_sclaled]_. The contact at closure is modelled through a penalizating term.
    The equations are scaled to make appear dimensionless coefficients and
    variables (e.g. [Chabassier_scaling]_):

    .. math::
        \begin{align}
        &\frac{1}{\omega_r^2}\ddot{y} + \frac{1}{\omega_r Q_r} \dot{y} +  y
        - K_c \left\vert \left[ y \right]^{-} \right\vert^{\alpha}   =  1 +\epsilon  \Delta p \\
        &p  = \gamma - \Delta p  \\
        &u = \zeta [y]^{+} \text{sign}(\Delta p) \sqrt{ \vert \Delta p \vert}
        + \epsilon \kappa \frac{1}{\omega_r} \dot{y}
        \end{align}

    With :math:`y` the dimensionless position of the reed, :math:`[y]^{\\pm}` the positive
    or negative part of :math:`y`, :math:`p,u` the dimensionless acoustic fields inside the
    instrument.

    The other coefficients must be specified by the user. They are:

    - "gamma" :math:`\gamma` : the first dimensionless parameter relative to the supply pressure
    - "zeta" :math:`\zeta` : the second dimensionless parameter relative to the "reed opening"
    - "kappa" :math:`\kappa` : the third dimensionless parameter relative to the "reed flow"
    - "pulsation" :math:`\omega_r` : the resonant angular frequency of the reed (in rad/s)
    - "qfactor" :math:`Q_r` : the quality factor of the reed
    - "model" : string coefficient that specifies if it is a "cane" (inwards) \
      or "lips" (inwards) reed. It fixes the value of :math:`\\epsilon` to 1 (cane) or -1 (lips)
    - "contact_stifness" :math:`K_c` : the stifness associated to the contact law when the reed is closed.
    - "contact_exponent" :math:`\\alpha` : the exponent associated to the contact law when the reed is closed.

    In order to rescale the variables, two other parameters can be given.
    They do not influence the result of the simulation, only the magnitude of the signals.

    - "closing_pressure" :math:`P_{closed}` the minimal pressure for which the reed is closed (in Pa)
    - "opening" :math:`y_0` : the resting opening height of the reed (in m)

    The "real" variables with dimensions are:

    - :math:`y \times y_0`
    - :math:`p \times P_{closed}`
    - :math:`u \times P_{closed} /Z_c` (with :math: `Z_c` the caracteristic impedance of the pipe)

    .. warning::
        This excitator can be used only in temporal domain

    See Also
    --------
    :py:class:`TemporalReed1dof <openwind.temporal.treed.TemporalReed1dof>`
        The temporal version of this excitator

    References
    ----------
    .. [Bilbao_sclaled] Bilbao, S. (2009). Direct simulation of reed wind instruments.\
        Computer Music Journal, 33(4), 43-55.
    .. [Fletcher_sclaled] Fletcher, N H. 1979. “Excitation Mechanisms in Woodwind and \
        Brass Instruments.” Acustica 43: 10.
    .. [Chabassier_scaling] J. Chabassier and R. Auvray 2022. "Control Parameters\
        for Reed Wind Instruments or Organ Pipes with Reed Induced Flow". 25th \
        International Conference on Digital Audio Effects (Vienna, Austria).



    Parameters
    ----------
    control_parameters : dict
        Associate each controle parameter (coefficient of the ODE) to its value\
        and the evolution with respect to time. It must have the keys\
        `["gamma", "zeta", "kappa", "pulsation", "qfactor", "model",\
        "contact_stifness", "contact_exponent", "closing_pressure", "opening"]`.
    label : str
        the label of the connector
    scaling : :py:class:`Scaling<openwind.continuous.scaling.Scaling>`
        object which knows the value of the coefficient used to scale the
        equations
    convention : {'PH1', 'VH1'}
        The basis functions for our finite elements must be of regularity
        H1 for one variable, and L2 for the other.
        Regularity L2 means that some degrees of freedom are duplicated
        between elements, whereas they are merged in H1.
        Convention chooses whether P (pressure) or V (flow) is the H1
        variable.

    Attributes
    ----------
    gamma: float or callable
        :math:`\gamma` : the first dimensionless parameter relative to the supply pressure
    zeta: float or callable
        :math:`\zeta` : the second dimensionless parameter relative to the "reed opening"
    kappa: float or callable
        :math:`\kappa` : the third dimensionless parameter relative to the "reed flow"
    pulsation: float or callable
        :math:`\omega_r` : the resonant angular frequency of the reed (in rad/s)
    qfactor: float or callable
        :math:`Q_r` : the quality factor of the reed
    model : -1 or 1
        :math:`\\epsilon` the opening sens of the reed: 1 (cane,\
        inwards) or -1 (lips, outwards)
    contact_stifness : float or callable
        :math:`Kc` : the stifness associated to the contact law when the reed is closed.
    contact_exponent : float or callable
        :math:`\\alpha` : the exponent associated to the contact law when the\
        reed is closed.
    opening : float or callable
        :math:`y_0` : the resting opening height of the reed (in m)
    closing_pressure: float or callable
        :math:`P_{closed}` the minimal pressure for which the reed is closed (in Pa)
    """

    NEEDED_PARAMS = ["gamma", "kappa",  "zeta", "pulsation", "qfactor",
                     "contact_stifness", "contact_exponent", "model"]
    POSSIBLE_PARAMS = NEEDED_PARAMS + ["opening", "closing_pressure"]

    def _update_fields(self, control_parameters):

        self.gamma = self._create_parameter(control_parameters["gamma"])

        # currently only constant value are accepted for other parameters
        # the numerical schemes are not compatible with varaible coefficients.

        self.zeta = self._create_cst_param(control_parameters["zeta"])
        self.kappa = self._create_cst_param(control_parameters["kappa"])
        self.pulsation = self._create_cst_param(control_parameters["pulsation"])
        self.qfactor = self._create_cst_param(control_parameters["qfactor"])
        self.contact_stifness =  self._create_cst_param(control_parameters["contact_stifness"])
        self.contact_exponent =  self._create_cst_param(control_parameters["contact_exponent"])

        # "opening" and "closing_pressure" are otpional as they do not influence the solution (excepting the scaling)
        if "opening" in control_parameters:
            self.opening = self._create_cst_param(control_parameters["opening"])
        else:
            self.opening = 1
        if "closing_pressure" in control_parameters:
            self.closing_pressure = self._create_cst_param(control_parameters["closing_pressure"])
        else:
            self.closing_pressure = 1

        # the value of epsilon is set by interpreting the keyword in "model"
        model = control_parameters['model']
        if model in ["lips","outwards"]:
            model_value = 1
        elif model in ["cane","inwards"]:
            model_value= -1
        else:
            warnings.warn("WARNING : your model is %s, but it must "
                  "be in {'lips', 'outwards', 'cane', 'inwards'}, it "
                  "will be set to default (lips)"
                  %model)
            model_value = 1
        self.model = FixedExcitatorParameter(model_value)


    @staticmethod
    def _create_cst_param(curve):
        if callable(curve):
            raise ValueError("Except 'gamma', evolving parameters of "
                             "Reed1dof Excitator are not supported yet, must be a"
                             " constant value")
        else:
            return FixedExcitatorParameter(curve)

    def get_dimensionfull_values(self, t, Zc, rho):
        """
        Return the values of all dimensionfull parameters at a given time:

        Parameters
        ----------
        t : float
            The given time we want to get the curves values from.
        Zc : float
            The characteristic (real) impedance at the entry of the instrument.
        rho : float
            the air density.

        Returns
        -------
        Sr : float
            the effective cross section area of the reed (in m²)
        Mr: float
            the effecitve mass of the read (in kg)
        g: float
            the damping coefficient
        omega02: float
             the squared resonant angular frequency
        w: float
            the effective width of the reed channel opening (in m)
        y0: float
            the resting opening of the reed (in m)
        epsilon: float
            1 (for lips/outwards) or -1 (cane/inwards) following the model
        pm: float
            the supply pressure (in Pa)
        """
        gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon = self.get_dimensionless_values(t, Zc, rho)
        Pclosed = self.get_Pclosed(t)
        y0 = self.opening.get_value(t)

        pm = gamma*Pclosed
        w = zeta*np.sqrt(.5*rho*Pclosed)/(Zc*y0)
        Sr = kappa*Pclosed/(omegar*Zc*y0)

        Mr = Pclosed*Sr/(omegar**2 * y0)

        g = omegar/Qr

        omega02 = omegar**2

        omega_c = (Kc*omegar**2)**(1/(alpha_c + 1))
        return (Sr, Mr, g, omega02, w, y0, epsilon, pm, omega_c, alpha_c)


    def get_dimensionless_values(self, t, Zc=1, rho=1):
        """
        Return the values of all dimensionless parameters at a given time:

        Parameters
        ----------
        t : float
            The given time we want to get the curves values from.
        Zc : float, UNUSED
            The characteristic (real) impedance at the entry of the instrument.
        rho : float, UNUSED
            the air density.

        Returns
        -------
        gamma : float
            Dimensionless parameter relative to the supply pressure.
        zeta : float
            Dimensionless parameter relative to the reed opening.
        kappa : float
            The dimensionless parameter relative to the "reed flow".
        Qr : float
            the quality factor of the ree.
        omegar : float
            The resonant angular frequency of the reed in rad/s.
        Kc : float
            The contact force stifness.
        alpha_c : TYPE
            DESCRIPTION.
        epsilon : float
            1 (for lips/outwards) or -1 (cane/inwards) following the model.

        """
        gamma = self.gamma.get_value(t)
        zeta = self.zeta.get_value(t)
        kappa = self.kappa.get_value(t)
        omegar = self.pulsation.get_value(t)
        Qr = self.qfactor.get_value(t)
        Kc = self.contact_stifness.get_value(t)
        alpha_c = self.contact_exponent.get_value(t)
        epsilon = self.model.get_value(t)
        return gamma, zeta, kappa, Qr, omegar, Kc, alpha_c, epsilon

    def get_Pclosed(self, t):
        r"""
        Return the closing pressure of the reed.

        This value does not modify the response of the system. It is only necessary
        to rescale the results of the equations.

        .. math::
            P_{\text{closed}} = \frac{K_r }{S_r}  y_0 =  \omega_r^2 \frac{M_r}{S_r} y_0

        Parameters
        ----------
        t : float
            The given time we want to get the curves values from.

        Returns
        -------
        Pclosed : float
            The closing pressure in Pa.

        """
        return self.closing_pressure.get_value(t)
