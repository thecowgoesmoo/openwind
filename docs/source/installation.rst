############
Installation
############

1. Quick install
################

In your `python virtual environment <https://docs.python.org/3/library/venv.html>`_, do :

.. code-block:: shell

  pip install openwind


Installation was successful ? You can now try the :any:`beginner's tutorial <basic_tutorial>`

.. note:: If it didn't work (and *only in this case*), you may check other procedures to install ``openwind`` on :ref:`Linux/MacOS <linux-macos>` or on :ref:`Windows <windows>`

.. _linux-macos :

2. Linux/MacOS
##############

.. hint:: Don't do the following steps if you already installed `openwind` successfully with pip. Just choose your favorite installation method out of :ref:`a. conda <conda-install>`, :ref:`b. cloning <clone-project>` or :ref:`c. sources <install-from-source>`. If you don't know which one to chose, we recommand to :ref:`install from sources <install-from-source>`.


.. _conda-install:

a. Conda install
****************

A conda package is available in our channel :

.. code-block :: python

  conda install -c openwind openwind


.. _clone-project:

b. Clone project
****************

.. code-block:: shell

  git clone git@gitlab.inria.fr:openwind/openwind.git .

Then add openwind to your :ref:`PythonPATH <configure-your-pythonpath>`

.. _install-from-source:

c. Install from source
**********************


If you rather want to install this software from the latest release, you can download the sources `here <https://gitlab.inria.fr/openwind/openwind/-/releases>`_, then extract it somewhere permanent on your computer, and add ``openwind`` to your :ref:`PythonPATH <configure-your-pythonpath>`.


.. _previous-versions:

d. Previous versions
********************

If you are interested in early development versions, you can download them from `our previous repository <https://gitlab.inria.fr/openwind/release/-/releases>`_. Don't forget to add ``openwind`` to your :ref:`PythonPATH <configure-your-pythonpath>`


.. _configure-your-pythonpath:

e. Configure your PythonPATH
****************************

.. warning:: Don't do this if you already installed `openwind` successfully with pip.


Make sure that ``openwind`` is in your ``PYTHONPATH`` (otherwise the line ``import openwind`` will fail).

Add the following line to your ``~/.bashrc``. (or ``~/.bash_profile`` if you are using MacOS), replacing ``/path/to/release`` with the correct path.

.. code-block:: shell

  export PYTHONPATH="${PYTHONPATH}:/path/to/release/"

(don't forget to reopen your terminal, or do ``source ~/.bashrc`` )

Otherwise, you can always change your ``PYTHONPATH`` at the top of your scripts
that use ``openwind``, with:

.. code-block:: python

  import sys
  sys.path.append("/path/to/release/")

.. _windows:

3. Windows
##########

.. warning:: Don't do this if you already installed `openwind` successfully with pip.

We are going to use Anaconda to create a virtual environment, which has build in Spyder `IDE <https://en.wikipedia.org/wiki/Integrated_development_environment>`_ and is the easiest way to get started with virtual environments on windows.

If you're already familiar with Anaconda or any other virtual environment manager, go to :ref:`command-line`  OR :ref:`install openwind from source <windows-install-from-source>`

a. Installation
***************

Anaconda Navigator
==================

If you are new to ``conda``, this will be the easiest way to install ``openwind``

All instructions to install Anaconda are given here: `docs.anaconda.com/anaconda/install/windows/ <https://docs.anaconda.com/anaconda/install/windows/>`_

**Once you have installed Anaconda**, open the Anaconda Navigator from the windows Start Menu :

.. image :: https://files.inria.fr/openwind/pictures/start.png
  :width: 750
  :align: center


Create a new environment:

1. Select the "Environment" tab in the left panel

2. Click on "create" at the bottom of the environment panel

3. Name it  ``openwind-env`` for example, and click on ``Create`` .

.. image :: https://files.inria.fr/openwind/pictures/anaconda1.png
  :width: 750
  :align: center

4. Add ``openwind`` to openwind channels

.. image :: https://files.inria.fr/openwind/pictures/anaconda2.png
  :width: 300
  :align: center

.. image :: https://files.inria.fr/openwind/pictures/anaconda3.png
  :width: 300
  :align: center

.. image :: https://files.inria.fr/openwind/pictures/anaconda4.png
  :width: 300
  :align: center


5. Once this is done, be sure that the package search is set to all, then search for "openwind" in the conda `search package` field and install it

.. image :: https://files.inria.fr/openwind/pictures/anaconda-all.png
  :width: 300
  :align: center

Select the `openwind` package, then click on Apply


.. image :: https://files.inria.fr/openwind/pictures/anaconda-apply.png
  :width: 300
  :align: center


You will be asked to install some dependencies as well, click on Apply.


.. image :: https://files.inria.fr/openwind/pictures/anaconda-installpackage.png
  :width: 300
  :align: center


When it is installed, go back to the main menu of Anaconda, select the ``openwind-env`` environment, and then launch Spyder (install it if it isn't already installed)


.. image :: https://files.inria.fr/openwind/pictures/spyder.png
  :width: 750
  :align: center

In Spyder console (on the bottom left), try this command :

.. code-block:: python

  [1]: import openwind


If nothing happens, it means that the installation is perfect, you can now start to use ``openwind``


.. _command-line:

Command line
============

**Install**


In your ``conda environment``, install ``openwind`` package with this command:

.. code-block:: shell

  conda install -c openwind openwind

**Update**


To update the package, type:

.. code-block:: shell

  conda update


.. _windows-install-from-source:

Install from source
===================


If you rather want to install this software from the latest release, you can download the sources `here <https://gitlab.inria.fr/openwind/openwind/-/releases>`_, then extract it somewhere permanent on your computer, and add openwind to your Pythonpath. If you are using Spyder, you can go to :ref:`spyder_pythonpath`.

.. _spyder_pythonpath:

b. Configure Pythonpath in Spyder
*********************************

.. caution:: This step is unessecary if you have installed `openwind` with pip, conda or anaconda. Do this only when you installed from :ref:`sources <windows-install-from-source>` or :ref:`previous releases <previous-versions>`

Open Spyder (not necessarly within the Anaconda Environment Manager) then click on the ``Python`` icon in the top bar:


.. image :: https://files.inria.fr/openwind/pictures/path.png
  :width: 750
  :align: center



A window will open, you can then navigate to the openwind-master directory (or wherever you have unzipped the source files) in the file explorer. Make sure that the folder name is the root directory of the project (openwind-master in our case), then click on "select folder" :


.. image :: https://files.inria.fr/openwind/pictures/path2.png
  :width: 750
  :align: center


In Spyder console (on the bottom left), try this command :

.. code-block:: python

  [1]: import openwind


If nothing happens, it means that the installation is perfect, you can now start to use ``openwind``


If you are not using Spyder, you can check `this link <https://stackoverflow.com/a/4855685>`_


c. Get started by running some examples
***************************************

You can choose to open a file from the "File -> Open" or by clicking on the shortcut in the top bar, then navigate within the examples subfolders, select a file to open and click on "Open"


.. image :: https://files.inria.fr/openwind/pictures/open_example.png
  :width: 750
  :align: center



Now you can press "F5" or click on the green triangle in the top bar to Run the file. The results will be shown in the console in the bottom-right panel, and you can see some figures on the top-right panel.


.. image :: https://files.inria.fr/openwind/pictures/run.png
  :width: 750
  :align: center


.. note:: you may see some warnings about "reloading packages", if you wish to deactivate them you can go to the parameters (the wrench icon in the top bar), the in the "Python Interpreter" tab unselect the "Show Reloaded modules list", you should be done with it


.. image :: https://files.inria.fr/openwind/pictures/reload.png
  :width: 750
  :align: center


Bonus : change Spyder theme
***************************

 In the parameters, go to to "Appearance (in the left pane), then change the "Interface theme" to "Light" (or "Dark"). Then change the "Syntax highlighting theme" to "Spyder" (or "Spyder-dark"), and click on "Apply". You will be asked to retart Spyder.


 .. image :: https://files.inria.fr/openwind/pictures/light.png
   :width: 750
   :align: center


 .. image :: https://files.inria.fr/openwind/pictures/syntax.png
   :width: 750
   :align: center
