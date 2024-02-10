
Ex. 11: Compute Resonance Frequencies
=====================================

This example presents how to accurately estimate the resonance frequencies related
to the eigenvalues of the underlying modal formulation as described in [1].

The different available models and their implementations are described in

* [1] Chabassier, J., & Auvray, R. (2022). Direct computation of modal parameters for musical wind instruments. Journal of Sound and Vibration, 528, 116775.
* [2] Thibault, A., & Chabassier, J. (2021). Dissipative time-domain one-dimensional model for viscothermal acoustic propagation in wind instruments. The Journal of the Acoustical Society of America, 150(2), 1165-1175.


Simple Cylinder
---------------

Use the keyword compute_method='modal' to activate the modal formulation
The matrix assembly is faster than for direct method, but an eigenvalue
problem must be solved, which can be longer.

.. code-block:: python

   fs= np.linspace(20, 10000,10000)
   temperature = 25
   (R0, L) = (1e-2, .189)
   file = [[0, R0], [L, R0]]
   result = ImpedanceComputation(fs, file, compute_method='modal', losses='diffrepr', nondim = True, temperature=temperature)

for comparison, also compute with usual direct method

.. code-block:: python

   result_direct = ImpedanceComputation(fs, file, compute_method='FEM', losses='diffrepr', nondim = True, temperature=temperature)

Call method resonance_peaks

* f are the resonance frequencies in Hz
* Q are the associated quality factors
* Z are the associated complex amplitudes

.. code-block:: python

   N = 15 # maximal number of printed frequencies

With "modal" these values are obtained directly from the eigenvalues

.. code-block:: python

   f, Q, Z = result.resonance_peaks(N)

With "FEM" or "TMM" they are estimated a posteriori from the impedance by looking around the 0-crossing of the phase

.. code-block:: python

   f_fem, Q_fem, Z_fem  = result_direct.resonance_peaks(N)
   nb = np.min([len(f_fem),len(f),N])

Display the frequencies and quality factors

.. code-block:: python

   print('Frequencies:')
   for i in range(0,nb):
       print(f"mode {i+1:2d} - Estimated (direct) : {f_fem[i]:4.2f} Hz - Resonance (modal) : {f[i]:4.2f} Hz")
   print('\nQuality factors')
   for i in range(0,nb):
       print(f"mode {i+1:2d} - Estimated (direct) : {Q_fem[i]:3.2f} - Resonance (modal) : {Q[i]:3.2f}")

   print('\nScaled magnitudes')
   for i in range(0,nb):
       print(f"mode {i+1:2d} - Estimated (direct) : {np.abs(Z_fem[i])/result.Zc:3.2f} -  Modal : {np.abs(Z[i])/result.Zc:3.2f}")

Be careful with the resonance frequencies greater than the highest prescribed
frequency vector: the spatial discretization is not sufficient anymore, those modes
are likely to be polluted with numerical error.

.. code-block:: python

   print(f"\nRelative error between modal and direct computation on Z: {np.linalg.norm(result_direct.impedance - result.impedance)/np.linalg.norm(result.impedance):3.3e}")

Print obtained impedances

.. code-block:: python

   fig=plt.figure(1), plt.clf()
   result.plot_instrument_geometry(figure=plt.gcf())
   fig=plt.figure(2)
   plt.clf()
   result.plot_impedance(figure=fig)
   result_direct.plot_impedance(figure=fig)
   ax=fig.get_axes()
   ax[0].plot(f, 20*np.log10(np.abs(Z/result.Zc)),'r+')
   ax[0].legend(['modal computation','direct computation','modal estimation'])
   plt.xlim([fs[0],fs[-1]/2])
   ax[1].plot(f, np.angle(Z/result.Zc),'r+')

Recompute the impedance on a new set of frequencies does not require
a new system solving. Be careful not to specify higher frequencies than the
original frequency range, for which the spatial discretisation was adjuted
to ensure a converged numerical solution

.. code-block:: python

   fs_supplementary = np.linspace(50, 3000,300)
   Z_supplementary = result.evaluate_impedance_at(fs_supplementary)

Match the resonance frequencies with the equally tempered scale to assess if
the instrument is in tune. This cylinder could be considered as a clarinet
tuned at the default pitch : 440 Hz. Use k= to specify how many frequencies (default is 5)

.. code-block:: python

   print(f"\n----------------- Matching of the resonance frequencies at 440 Hz : -----------------")
   (pitches,devs,names) = result.match_peaks_with_notes(k=7, display=True)

If the player says "C" when playing the fundamental note, the note names can be adjusted to match this vocabulary

.. code-block:: python

   print(f"----------------- Matching of the resonance frequencies at 440 Hz for a transposing instrumentist : -----------------")
   (pitches,devs,names) = result.match_peaks_with_notes(transposition="A", display=True)

The same instrument can also be seen as a clarinet in Bb tuned at 415 Hz

.. code-block:: python

   print(f"----------------- Matching of the resonance frequencies at 415 Hz for a transposing instrumentist : -----------------")
   (pitches,devs,names) = result.match_peaks_with_notes(concert_pitch_A=415, transposition="Bb", display=True)

This option also works with the estimated frequencies of the direct method

.. code-block:: python

   print(f"----------------- Matching of the estimated frequencies at 440 Hz : -----------------")
   (pitches,devs,names) = result_direct.match_peaks_with_notes(display=True)

Complex Instrument
------------------

Compute the resonance frequencies on a brass instrument with valves

.. code-block:: python

   geom = [[0,  .1, 5e-3, 3e-3, 'linear'],
                     [.1, 1.3, 5e-3, 5e-2, 'bessel', .4]]
   holes = [['variety',  'label',    'position', 'reconnection', 'radius',   'length'],
                   ['valve',    'piston1',   0.1,       .125,            3e-3,       0.11],
                   ['valve',    'piston2',  0.15,        .155,           5e-3,       0.07],
                   ['valve',    'piston3',  0.29,       .32,            2e-3,       0.22],]
   fing_chart = [['label',   'note0', 'note1', 'note2', 'note3', 'strange'],
                 ['piston1', 'o',      'x',    'x',       'o',   'o'],
                 ['piston2', 'o',      'x',    'o',       'x',   '0.5'],
                 ['piston3', 'o',      'x',    'x',       'x',   'x']]
   f1, f2 = 50, 3000
   fs= np.linspace(f1,f2,10000)
   result = ImpedanceComputation(fs, geom, holes,fing_chart, note='strange', compute_method='modal', losses='diffrepr', nondim = True, temperature=temperature)

Display resonance frequencies

.. code-block:: python

   N = 30 # maximal number of printed frequencies
   f, Q, Z = result.resonance_peaks(N)
   nb = len(f)
   for i in range(0,nb):
       print(f"mode {i+1:2d} - Resonance frequency : {f[i]:4.2f} Hz, Quality factor : {Q[i]:3.2f}, Scaled amplitude : {np.abs(Z[i])/result.Zc:3.2f}")

Print obtained impedances

.. code-block:: python

   fig=plt.figure(1), plt.clf()
   result.plot_instrument_geometry(figure=plt.gcf())
   fig=plt.figure(2)
   plt.clf()
   result.plot_impedance(figure=fig)
   ax=fig.get_axes()
   ax[0].plot(f, 20*np.log10(np.abs(Z/result.Zc)),'r+')
   ax[0].legend(['modal computation','modal estimation'])
   plt.xlim([fs[0],fs[-1]/2])
   ax[1].plot(f, np.angle(Z/result.Zc),'r+')

Available Options With This Computation Method
----------------------------------------------

Not all models implemented in openwind are compatible with the modal method.
Let us demonstrate them on a simple cone

.. code-block:: python

   shape = [[0,1e-3],[0.2,5e-3]]
   fs = [2000]

The following radiation categories are compatible

.. code-block:: python

   rad_cats = ['planar_piston', 'unflanged', 'infinite_flanged',
               'closed', 'perfectly_open']
   for rad_cat in rad_cats:
       res_direct = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category=rad_cat, losses='diffrepr', compute_method='FEM', temperature=temperature)
       res_modal  = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category=rad_cat, losses='diffrepr', compute_method='modal', temperature=temperature)
       err = np.abs(res_direct.impedance - res_modal.impedance) / np.abs(res_direct.impedance)
       # modal and direct method should give the same result
       assert(np.max(err)< 1e-10)

Not all viscothermous models are compatible.
Lossless is one of them.

.. code-block:: python

   res_direct = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category='unflanged', losses=False, compute_method='FEM', temperature=temperature)
   res_modal  = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category='unflanged', losses=False, compute_method='modal', temperature=temperature)
   err = np.abs(res_direct.impedance - res_modal.impedance) / np.abs(res_direct.impedance)
   assert(np.max(err)< 1e-10)

Only diffusive representations of losses are compatible.
They need to be written in expensive form, with all auxiliary variables as
unknowns of the system. This can be activated with losses='diffrepr+' but
calling losses='diffrepr' automatically activates the auxiliary variables.

.. code-block:: python

   res_direct = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category='unflanged', losses='diffrepr', compute_method='FEM', temperature=temperature)
   res_modal  = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category='unflanged', losses='diffrepr', compute_method='modal', temperature=temperature)
   err = np.abs(res_direct.impedance - res_modal.impedance) / np.abs(res_direct.impedance)
   assert(np.max(err)< 1e-10)

The diffusive representation of losses are an approximation of Zwikker-Kosten
model of losses, so be aware that there is a small error with a direct computation of ZK:

.. code-block:: python

   res_direct = ImpedanceComputation(fs, shape, l_ele=0.1, order=5, radiation_category='unflanged', losses=True, compute_method='FEM', temperature=temperature)
   err = np.abs(res_direct.impedance - res_modal.impedance) / np.abs(res_direct.impedance)
   print(f"\nError between ZK and diffrepr : {err[0]:e}")

Other Options
-------------

Otherwise, you can enjoy all the wonderful functionnalities of openwind as
spherical waves, arbitrary shape, discontinuities, lateral holes and valves,
varying temperature, automatic meshing and so on...

.. code-block:: python

   fs= [5000]
   (R0, L) = (1e-3, .2)
   file = [[0, 3*R0], [L/2, R0], [L/2, 3*R0], [L, 3*R0]]
   result        = ImpedanceComputation(fs, file, compute_method='modal', losses='diffrepr', nondim = True,
                                        discontinuity_mass=True, matching_volume=True, spherical_waves=True, temperature=temperature)
   result_direct = ImpedanceComputation(fs, file, compute_method='FEM', losses='diffrepr', nondim = True,
                                        discontinuity_mass=True, matching_volume=True, spherical_waves=True, temperature=temperature)
   err = np.abs(result_direct.impedance - result.impedance) / np.abs(result_direct.impedance)
   N = 5 # maximal number of printed frequencies
   f, Q, Z = result.resonance_peaks(N)

Call method resonance_frequencies, available for all solving methods, based on peak finding for FEM and TMM.
Display the frequencies and quality factors

.. code-block:: python

   for i in range(0,len(f)):
       print(f"mode {i+1:2d} - Resonance frequency (modal) : {f[i]:4.2f} Hz, Quality factor : {Q[i]:3.2f}, Scaled amplitude : {np.abs(Z[i])/result.Zc:3.2f}")
   print(f"\nRelative error between modal and direct computation on Z: {np.linalg.norm(result_direct.impedance - result.impedance)/np.linalg.norm(result_direct.impedance):3.3e}")
