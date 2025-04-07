.. meta::
   :description: ROCm Compute Profiler installation and deployment
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, tool, Instinct, accelerator, AMD,
              install, deploy, Grafana, client, configuration, modulefiles

**********************************************
Installing and deploying ROCm Compute Profiler
**********************************************

ROCm Compute Profiler consists of two installation components.

* :ref:`ROCm Compute Profiler core installation <core-install>` (client-side)

  * Provides the core application profiling capability.
  * Allows the collection of performance counters, filtering by hardware
    block, dispatch, kernel, and more.
  * Provides a CLI-based analysis mode.
  * Provides a standalone web interface for importing analysis metrics.

* :doc:`Grafana server for ROCm Compute Profiler <grafana-setup>` (server-side) (*optional*)

  * Hosts the MongoDB backend and Grafana instance.
  * Is packaged in a Docker container for easy setup.

Determine what you need to install based on how you would like to interact with
ROCm Compute Profiler. See the following decision tree to help determine what installation is
right for you.

.. image:: ../data/install/install-decision-tree.png
   :align: center
   :alt: Decision tree for installing and deploying ROCm Compute Profiler
   :width: 800

.. _core-install:

Core installation
=================

The core ROCm Compute Profiler application requires the following basic software
dependencies. As of ROCm 6.2, the core ROCm Compute Profiler is included with your ROCm
installation.

* Python ``>= 3.8``
* CMake ``>= 3.19``
* ROCm ``>= 5.7.1``

.. note::

   ROCm Compute Profiler will use the first version of ``python3`` found in your system's
   ``PATH``. If the default version of Python is older than 3.8, you may need to
   update your system's ``PATH`` to point to a newer version.

ROCm Compute Profiler depends on a number of Python packages documented in the top-level
``requirements.txt`` file. Install these *before* configuring ROCm Compute Profiler.

.. tip::

   If looking to build ROCm Compute Profiler as a developer, consider these additional
   requirements.

   .. list-table::

       * - ``docs/sphinx/requirements.txt``
         - Python packages required to build this documentation from source.

       * - ``requirements-test.txt``
         - Python packages required to run ROCm Compute Profiler's CI suite using PyTest.

The recommended procedure for ROCm Compute Profiler usage is to install into a shared file
system so that multiple users can access the final installation. The
following steps illustrate how to install the necessary Python dependencies
using `pip <https://packaging.python.org/en/latest/>`_ and ROCm Compute Profiler into a
shared location controlled by the ``INSTALL_DIR`` environment variable.

.. tip::

   To always run ROCm Compute Profiler with a particular version of Python, you can create a
   bash alias. For example, to run ROCm Compute Profiler with Python 3.10, you can run the
   following command:

   .. code-block:: shell

      alias rocprof-compute-mypython="/usr/bin/python3.10 /opt/rocm/bin/rocprof-compute"

.. _core-install-cmake-vars:

Configuration variables
-----------------------
The following installation example leverages several
`CMake <https://cmake.org/cmake/help/latest>`_ project variables defined as
follows.

.. list-table::
    :header-rows: 1

    * - CMake variable
      - Description

    * - ``CMAKE_INSTALL_PREFIX``
      - Controls the install path for ROCm Compute Profiler files.

    * - ``PYTHON_DEPS``
      - Specifies an optional path to resolve Python package dependencies.

    * - ``MOD_INSTALL_PATH``
      - Specifies an optional path for separate ROCm Compute Profiler modulefile installation.

.. _core-install-steps:

Install from source
-------------------

#. A typical install begins by downloading the latest release tarball available
   from `<https://github.com/ROCm/rocprofiler-compute/releases>`__. From there, untar and
   navigate into the top-level directory.

   ..
      {{ config.version }} substitutes the ROCm Compute Profiler version in ../conf.py

   .. datatemplate:nodata::

      .. code-block:: shell

         tar xfz rocprofiler-compute-v{{ config.version }}.tar.gz
         cd rocprofiler-compute-v{{ config.version }}

#. Next, install Python dependencies and complete the ROCm Compute Profiler configuration and
   install process.

   .. datatemplate:nodata::

      .. code-block:: shell

         # define top-level install path
         export INSTALL_DIR=<your-top-level-desired-install-path>

         # install python deps
         python3 -m pip install -t ${INSTALL_DIR}/python-libs -r requirements.txt

         # configure ROCm Compute Profiler for shared install
         mkdir build
         cd build
         cmake -DCMAKE_INSTALL_PREFIX=${INSTALL_DIR}/{{ config.version }} \
                 -DPYTHON_DEPS=${INSTALL_DIR}/python-libs \
                 -DMOD_INSTALL_PATH=${INSTALL_DIR}/modulefiles/rocprofiler-compute ..

         # install
         make install

   .. tip::

      You might need to ``sudo`` the final installation step if you don't have
      write access for the chosen installation path.

#. Upon successful installation, your top-level installation directory should
   look like this.

   .. datatemplate:nodata::

      .. code-block:: shell

         $ ls $INSTALL_DIR
         modulefiles  {{ config.version }}  python-libs

.. _core-install-modulefiles:

Execution using modulefiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^

The installation process includes the creation of an environment modulefile for
use with `Lmod <https://lmod.readthedocs.io>`_. On systems that support Lmod,
you can register the ROCm Compute Profiler modulefile directory and setup your environment
for execution of ROCm Compute Profiler as follows.

.. datatemplate:nodata::

   .. code-block:: shell

      $ module use $INSTALL_DIR/modulefiles
      $ module load rocprofiler-compute
      $ which rocprof-compute
      /opt/apps/rocprofiler-compute/{{ config.version }}/bin/rocprof-compute

      $ rocprof-compute --version
      ROC Profiler:   /opt/rocm-5.1.0/bin/rocprof

      rocprofiler-compute (v{{ config.version }})

.. tip::

   If you're relying on an Lmod Python module locally, you may wish to customize
   the resulting ROCm Compute Profiler modulefile post-installation to include extra
   module dependencies.

Execution without modulefiles
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To use ROCm Compute Profiler without the companion modulefile, update your ``PATH``
settings to enable access to the command line binary. If you installed Python
dependencies in a shared location, also update your ``PYTHONPATH``
configuration.

.. datatemplate:nodata::

   .. code-block:: shell

      export PATH=$INSTALL_DIR/{{ config.version }}/bin:$PATH
      export PYTHONPATH=$INSTALL_DIR/python-libs

.. _core-install-package:

Install via package manager
---------------------------

Once ROCm (minimum version 6.2.0) is installed, you can install ROCm Compute Profiler using
your operating system's native package manager using the following commands.
See :doc:`rocm-install-on-linux:index` for guidance on installing the ROCm
software stack.

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         $ sudo apt install rocprofiler-compute
         # Include rocprofiler-compute in your system PATH
         $ sudo update-alternatives --install /usr/bin/rocprof-compute rocprof-compute /opt/rocm/bin/rocprof-compute 0
         # Install Python dependencies
         $ python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt

   .. tab-item:: Red Hat Enterprise Linux

      .. code-block:: shell

         $ sudo dnf install rocprofiler-compute
         # Include rocprofiler-compute in your system PATH
         $ sudo update-alternatives --install /usr/bin/rocprof-compute rocprof-compute /opt/rocm/bin/rocprof-compute 0
         # Install Python dependencies
         $ python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt

   .. tab-item:: SUSE Linux Enterprise Server

      .. code-block:: shell

         $ sudo zypper install rocprofiler-compute
         # Include rocprofiler-compute in your system PATH
         $ sudo update-alternatives --install /usr/bin/rocprof-compute rocprof-compute /opt/rocm/bin/rocprof-compute 0
         # Install Python dependencies
         $ python3 -m pip install -r /opt/rocm/libexec/rocprofiler-compute/requirements.txt

.. _core-install-rocprof-var:

ROCProfiler
-----------

ROCm Compute Profiler relies on :doc:`ROCProfiler <rocprofiler:index>`'s ``rocprof`` binary
during the profiling process. Normally, the path to this binary is detected
automatically, but you can override the path by the setting the optional
``ROCPROF`` environment variable.
