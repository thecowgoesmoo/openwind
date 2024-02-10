# Changelog

All notable changes to this project will be documented in this file.


## [0.11.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.11.1) - 2024-01-10 

### Changed:
- fix issue with octave number in `match_freqs_with_notes`


## [0.11.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.11.0) - 2024-01-09 

### Added:
- Gives the possibility to use dimensionless equations in time domain (option `nondim=True`), using scaled time or not (option `is_scaled_time`)
- Admittance computation of flute-like instrument, including the radiation of the embouchure hole (window)
- Sound simulation of flute like-instrument using jet-drive model
- Simulation of transverse instrument by locating the sound-source at a side-hole opening using the key-word "source_location"
- If a long instrument is given, a warning (L>3m) and an error (L>25m) are raised to avoid unit mistake (confusion between mm and m)


### Changed:
-


## [0.10.3](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.10.3) - 2023-10-19 

### Changed:
- update requirements with `h5py` (in anaconda config and "requirements.txt")


## [0.10.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.10.2) - 2023-10-18 

### Added:
- The transfer matrix for cone using the model proposed by C.J.Nederveen and reformulated by T.Grothe JASA 2023.
- Scripts used for generating figures for Alexis Thibault Thesis 
- Roughness parametric model 
- Explicit/Implicit time discretisation for tonehole 
- Space interpolation for time simulation 

### Changed:
- New parameters for time simulation export (HDF5, ...)

## [10.0.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v10.0.2) - 2023-10-18 

### Added:
- The transfer matrix for cone using the model proposed by C.J.Nederveen and reformulated by T.Grothe JASA 2023.
- Scripts used for generating figures for Alexis Thibault Thesis 
- Roughness parametric model 
- Explicit/Implicit time discretisation for tonehole 
- Space interpolation for time simulation 

### Changed:
- New parameters for time simulation export (HDF5, ...)

## [0.10.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.10.1) - 2023-08-25 

### Changed:
- try to fix CI to automatic generation of anaconda package


## [0.10.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.10.0) - 2023-08-25 

### Added:
- Reed/lips model with dimensionless control parameters (zeta, gamma, kappa)
- `antiresonance_peaks` method in `ImpedanceComputation`
- effect of air composition (humidity and CO2 rates) on the acoustic propagation with the keywords *"humidity"* and *"carbon"*
- Some geometric constrains can now be added to the inverse problem directly from the `InstrumentGeometry` (length, spline nodes distance, conicity, hole radius and position and distance between holes)
- An example (inversion/Ex8...) has been added to illustrate this feature
- option `reff_tmm_losses` in `tmm_tools` to chose the effective radius used to compute the losses in cones for TMM

### Changed:
- reorganize the keyword arguments between `ImpedanceComputation` and `FrequentialSolver`
- Example 3 of "frequential" to include the effect of air composition
- the numeric scheme used for the reed has been rewritten with dimensionless control parameters


## [0.9.3](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.9.3) - 2023-02-08 

### Added:
- `antiresonance_peaks` method in `ImpedanceComputation`


## [0.9.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.9.2) - 2023-01-03 

### Added:
- give the possibility to use geometry with conical holes or valves (deviation pipes) by adding a column "radius_out" in the side component file (holes/valves)
- scripts to generate the figures of the article: Alexis Thibault, Juliette Chabassier, Henri Boutin, Thomas Hélie. _Transmission line coefficients for viscothermal acoustics in conical tubes._ Journal of Sound and Vibrations, 2022.
- possibility to write/load a single file 'filename.ow' including the data for the main bore, the holes/valves and the fingering chart.
- possibility to add holes/valves *a posteriori* with `InstrumentGeometry.add_side_components()`
- possibility to print file in (x,r) format for the main bore
- give access to pressure and flow field from `ImpedanceComputation`

### Changed:
- Improve performance by changing LU factorization method


## [0.9.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.9.0) - 2022-09-02 

### Added:
- computation of resonance frequencies with modal method
- matching of frequencies with name notes
- Parsing tools to use RESONANS and Pafi files with Openwind
- Module `compute_transfer_matrix` allowing the computation with FEM of the transfer matrix coefficients for any geometry


### Changed:
- fix issue with method `FrequentialSolver.update_frequencies_and_mesh()`: the indicated fingering was not maintained


## [0.8.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.8.1) - 2022-05-23 

### Added:

- Possibility to use millimeter and diameter for geometries.
- Possibility to write the geometry files in millimeter and diameter
- Example `technical/Ex7_...` explaining how to use millimter and/or diameter
- Test relative to this new functionality

### Changed:

- Profile matrix construction in frequential domain
- Put the purely parsing functions outside `InstrumentGeometry`, in `openwind.technical.parser` module


## [0.8.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.8.0) - 2022-04-07 

### Added:

- Possibility to get the value of the physical quantities at the entrance of the instrument with the method `InstrumentPhysics.get_entry_coefs`
- Possibility to plot only the modulus of the impedance (and not the phase angle), with the option `modulus_only=True` in the methods `plot_impedance()`
- the main classes `FrequentialSolver` and `TemporalSolver` can now be imported directly as `openwind.FrequentialSolver`
- One example in inversion (Ex7) on how to ensure the continuity of the radius at the junction of two pipes during an optimization, or inversely to let the possibility to have a discontinuity of section
- method to convert RESONANS files into openwind compatible files

### Removed:

- plotly from dependencies for conda & pip

### Changed:

- update frequential tests
- update examples with the new import of `FrequentialSolver` and `TemporalSolver`
- improve performance of temporal simulations
- Move the computation of the shortest wavelength for automatic meshing from `ImpedanceComputation` to `FrequentialSolver`
- Method `FrequantialSolver.plot_var3D()` render by default a 2D surface with matplotlib instead of a 3D surface with plotly.  It can still be plotted with the keyword `with_plotly`
- The error messages `InstrumentGeometry` are now more explicit


## [0.7.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.7.2) - 2022-02-08 

### Changed:

- Corrected conda package generation & removed plotly from dependencies


## [0.7.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.7.1) - 2022-01-28 

### Added:

- Warning if target impedance is not normalized in inversion
- `plt.show()` in example files

### Removed:

- Images from documentation

### Changed:

- Correct repeated label issue for `InstrumentGeometry.plot_InstrumentGeometry()`
- Correct test issues with random numbers


## [0.7.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.7.0) - 2021-12-14 

### Updated:

- Documentation layout & generation updated

### Added:

- Give the possibility to use valves (for brass instruments):
  + add the new classes :
    * `openwind.technical.instrument_geometry.BrassValve`
    * `openwind.continuous.junction.JunctionSwitch`
    * `openwind.frequential.frequential_junction_switch.FrequentialJunctionSwitch `
    * `openwind.temporal.tjunction.TemporalJunctionSwitch`
  + complete the `InstrumentGeometry` parser to reed files with valves
  + add the plot of valves in `InstrumentGeometry`
  + complete `InstrumentPhysics` to create `Netlist` with valves
  + add an example illustrating this new functionality `example\module\Ex2bis_Brass_Valves.py`
  + add tests relative to these new functionalities
  + add the doc relative to this new functionality
- Possibility to use "bell" label in the fingering chart to open/closed the bell opening
- Geometry Manipulation:
    - The possibility to slice the geometry with the method `InstrumentGeometry.extract(start, stop)`
    - The possibility to concatenate several `InstrumentGeometry` by adding them
    - One example to illustrate this two functionalities
 - New models of viscothermal losses `'bessel_new'` and `'sh'` corresponding to
 modified loss coefficients to account for conicity. Article to be published :
  Thibault, Alexis and Chabassier, Juliette and Boutin, Henri and Hélie, Thomas.
  A low reduced frequency model for viscothermal wave propagation in conical
  tubes of arbitrary cross-section.

### Changed:

- Precise the input parameters of `InstrumentGeometry` to be more explicit

## [0.6.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.6.1) - 2021-06-23 

### Added:
- Documention on "clone project"

### Changed:

- fix some typo in documentation
- clarify navigation bar in documentation
- hyperlinks to images in documention
- fix problem of keywords arguments in `ImpedanceComputation.write_impedance()`


## [0.6.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.6.0) - 2021-05-27 

### Updated:


- Documentation (organisation and cross reference)
- Docstring format
- Conda package documentation link
- Fixed gitlab-ci to update documentation
- Build doc when publishing a release
- Fixed gitlab-ci to update pip, conda & doc
- Radiation example (frequential, ex.4)

### Added

- Possibility to use different radiation categories for the bell and the holes
- Possibility to use different radiation categories for the bell and each holes
- Possibility to use radiation impedance from data
- Radiation model from *Silva et al., JSV, 2009* (both causal and non-causal) can be used in frequential domain and bore reconstruction
- Gradient computation with spherical waves
- Code structure page in Reference (documentation)
- Error message when inversion is tried with non-uniform temperature or variable input radius
- Temporal version of the junction between to pipes with mass due to cross-section discontinuity

### Changed

- Group the class `RadiationModel` and `PhysicalRadiation` in a unique  `PhysicalRadiation` class
- The angle of the 'pulsating_sphere' condition is now computed from the slope of the radiating pipe in `PhysicalInstrument`
- The radiation class in `openwind.continuous` have been rewritten
- `PipeEnd` class definition moved out of the `Netlist` class
- `End` classes definition moved out of `FrequentialPipeFEM`, `FrequentialPipeTMM` and `TemporalPipe`
- Uniformize the name convention concerning `Component`, `Connector` and `Pipe` in the graph
- Rename the excitator `Valve` into `Reed1dof`
- Player attributes moved to a dictionnary with "excitator_type" as a key
- Inversion example files from FA2020, replaced by file related to submission to Acta Acustica.
- Modify method names: `InstrumentGeometry.print_files()` to `InstrumentGeometry.write_files()`

### Removed

- `ReedModel` was obsolete and removed


## [0.5.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.5.2) - 2021-03-09

### Updated:

- Fixed gitlab-ci to update pip, conda & doc


## [0.5.1](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.5.1) - 2021-03-09

### Updated:

- Fixed gitlab-ci to update pip, conda & doc


## [0.5.0](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.5.0) - 2021-03-08

### Added

- Add latest_changelog to releases description
- Add FrequentialPressureCondition (usable with `radiation_category='closed'`)
- Add option `enable_tracker_display` to FrequentialSolver
- Add a method 'is_TMM_compatible' in DesignShape
- CI support for documentation automatic update
- Conda package update from CI
- Install on Windows in the documentation
- Script to automatically parse examples to documentation
- CONTRIBUTORS & LICENCE files
- Link to previous releases in README (v0.1, 0.2 & 0.3)


### Update
- Install with conda tutorials in documentation


### Changed
- Correct continuity assertion in instrument geometry to allow small deviation due to numerical approximation
- Correct equation for matching volume
- Correction of radius in radiation pulsatin sphere
- correct Netlist update during inversion
- Correct LM algorithm
- Added player in InstrumentPhysics arguments
- imports in examples & test files
- update changelog : python script instead of shell
- changed test/ into tests/
- headers metadata -> licence
- bumpversion manager : using tbump instead of bump2version (more complete)
- minimal requirements.txt file for installation


### Removed
- all pictures from doc


## [0.4.2](https://gitlab.inria.fr/openwind/openwind/-/releases/v0.4.2) - 2021-02-17

### Added

- add a method 'is_TMM_compatible' in DesignShape
- add changelog (this file)
- add requirements.txt
- add bumversion.cfg
- add Release and Pip_Update stages to gitlab-ci.yml

### Fixed

- Correct a typo in LM algorithm and modify the tests consequently
- Correct the linesearch algorithm with a quadratic approach
- Coreect netlist update during inversion
- correct inversion examples using Player in InstrumentPhysics parameters
- Correct continuity assertion in instrument geometry to allow small deviation due to numerical approximation
- Correct matching volume equation
- Correct RadiationPulsatingSphere
