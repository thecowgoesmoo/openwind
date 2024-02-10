
Ex. 5: Convergence curves of a simple reed instrument.
======================================================

How to generate convergence curves.

This example uses the :py:meth:`simulate <openwind.temporal_simulation.simulate>` method

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   import simpleaudio as sa
   import scipy.signal as signal
   from openwind import Player, simulate
   from openwind.temporal.utils import export_mono
   from openwind.technical.temporal_curves import constant_with_initial_ramp

Definition
----------

A cylinder

.. code-block:: python

   instrument = [[0.0, 1e-2],
                 [0.3, 1e-2]]

with no hole

.. code-block:: python

   holes = []

a reed model for the oscillator

.. code-block:: python

   player = Player('CLARINET')

other parameters:


* the series of time refinements
  .. code-block:: python

     fact = [1,2,4,8,16,32,64]
     #fact = [4]

* the series of mouth pressure targets
  .. code-block:: python

     mouth_pressures = [1900,1950,2000,2050,2100, 2200,2400,2600,2800,3000]
     #mouth_pressures = [2200]

* simulation time in seconds
  .. code-block:: python

     duration = 1

* first simulation is run with 17221 time steps in each second
  .. code-block:: python

     n_time_step_base = 17221 * duration

Loop on the target mouth_pressures
----------------------------------

.. code-block:: python

   outputs = {}
   for jj in mouth_pressures:
       # loop on the refinements
       for ii in fact:

           n_time_step = ii * n_time_step_base #51661  # power of 2 * n_time_step_base
           # launch simulation
           rec , _ = simulate(duration,
                          instrument,
                          holes,
                          player = player,
                          losses='diffrepr',
                          temperature=20,
                          l_ele=0.25, order=4,  # Discretization parameters
                          n_steps=n_time_step
                          )

           # signal = rec.values['bell_radiation_pressure']
           outputs['rec_{}_pm_{}'.format(ii, jj)] = rec



   output_values = {}
   for i in outputs:
       output_values[i] = outputs[i].values

   np.save('output_values.npy', output_values)


   read_outputs = np.load('output_values.npy',allow_pickle=True).item()

Compute the consecutive errors
------------------------------

.. code-block:: python


   abs_norm = []
   for jj in mouth_pressures:
       pm_norm = []
       for ii in fact:
           new = read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure'][::ii]
           ref = read_outputs['rec_{}_pm_{}'.format(fact[-1], jj)]['bell_radiation_pressure'][::fact[-1]]

           pm_norm.append(np.max(np.abs(new - ref)) / np.max(np.abs(ref)))

       abs_norm.append(pm_norm)



   mouth_pressures.reverse()

Convergence curves
------------------

.. code-block:: python

   plt.figure(2)
   for jj in range(len(mouth_pressures)):
       plt.loglog([duration/(ii*n_time_step_base) for ii in fact[:-1]], abs_norm[jj][:-1], 'o-')
   plt.xlabel('Dt [s]')
   plt.ylabel('sup norm of the difference')
   plt.title('convergence')
   plt.grid(True, 'major', linewidth=2)
   plt.grid(True, 'minor', linewidth=0.5)
   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])

Plots and Sounds
----------------

plot output sound

.. code-block:: python

   plt.figure(1)
   for ii in [fact[0]]:
       for jj in [2200]:
           plt.title('Bell Radiation pressure ; tuyau L = 30cm, r = 1cm')
           plt.plot(np.linspace(0, duration, ii * n_time_step_base),
                    read_outputs['rec_{}_pm_{}'.format(ii, jj)]['bell_radiation_pressure'],
                    linewidth=1)

   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
   plt.xlabel('temps [s]')

export sounds

.. code-block:: python


   NP = fact[0]
   PM = mouth_pressures[0]
   sound = outputs['rec_{}_pm_{}'.format(NP, PM)].values['bell_radiation_pressure']

   sound = np.interp(np.linspace(0, duration, int(44100 * duration)),
                     np.linspace(0, duration, int(NP * n_time_step_base)),
                     sound)

   audio = sound * (2**15 - 1) / np.max(np.abs(sound))
   audio = audio.astype(np.int16)

       # Start playback
   play_obj = sa.play_buffer(audio, 1, 2, 44100)

       # Wait for playback to finish before exiting
   play_obj.wait_done()

the bell radiation and y-reed

.. code-block:: python

   plt.figure(3)
   for ii in [1]:
       for jj in mouth_pressures[:]:
           ax1 = plt.subplot(2,1,1)
           plt.ylabel('Delta p')
           t = np.linspace(0, duration, ii * n_time_step_base)
           loc_plot = (constant_with_initial_ramp(jj)(t) -
                    read_outputs['rec_{}_pm_{}'.format(ii, jj)]['source_pressure'])
           plt.plot(t,
                    loc_plot,
                    linewidth=1)
           plt.fill_between(t,
                            loc_plot,
                            0,
                            loc_plot<0)

           ax2 = plt.subplot(2,1,2, sharex=ax1)
           plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
                read_outputs['rec_2_pm_{}'.format(jj)]['source_y'])
           plt.grid(True)
   plt.ylabel('y anche')

   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))],
              loc='lower right')
   plt.xlabel('temps [s]')

y-reed

.. code-block:: python

   plt.figure(33)
   for jj in mouth_pressures:
       plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
                read_outputs['rec_2_pm_{}'.format(jj)]['source_y'])
   plt.xlabel('temps')
   plt.ylabel('y anche')
   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
   plt.grid(True)

delta p

.. code-block:: python

   plt.figure(4)
   for ii in [2]:
       for jj in mouth_pressures:
           t = np.linspace(0, duration, ii * n_time_step_base)
           loc_plot = (constant_with_initial_ramp(jj)(t) -
                    read_outputs['rec_{}_pm_{}'.format(ii, jj)]['source_pressure'])
           plt.plot(t,
                    loc_plot,
                    linewidth=1)
           plt.fill_between(t,
                            loc_plot,
                            0,
                            loc_plot<0)
   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])
   plt.xlabel('temps')
   plt.ylabel('Delta p')
   plt.grid(True,'major')

d/dt delta p

.. code-block:: python

   plt.figure(4)
   for ii in [fact[-1]]:
       for jj in [0]:
           jjj = mouth_pressures[jj]
           t = np.linspace(0, duration, ii * n_time_step_base)
           loc_plot = np.gradient((constant_with_initial_ramp(jjj)(t) -
                    read_outputs['rec_{}_pm_{}'.format(ii, jjj)]['source_pressure']))
           plt.plot(t,
                    loc_plot,
                    linewidth=1)
           plt.xlabel('temps')
           plt.ylabel('d/dt Delta p ; pm = {}'.format(jjj))


   # plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])

RAMP MOUTH PRESSURE

.. code-block:: python

   plt.figure(5)
   for jj in mouth_pressures:
       plt.plot(np.linspace(0, duration, 2 * n_time_step_base),
                constant_with_initial_ramp(ii)(np.linspace(0, duration, 2 * n_time_step_base)))

   plt.legend(['pm = '+str(mouth_pressures[kk])+ 'Pa' for kk in range(len(mouth_pressures))])





   t = np.linspace(0, duration, fact[-1] * n_time_step_base)
   loc_plot = np.gradient((constant_with_initial_ramp(PM)(t) -
                           read_outputs['rec_{}_pm_{}'.format(fact[-1], PM)]['source_pressure']))
   plt.plot(t,
            loc_plot,
            linewidth=1)
   plt.xlabel('temps [s]')
   plt.ylabel('Delta p [Pa]')
   plt.title(f'pm = {PM} Pa ; Delta P')
