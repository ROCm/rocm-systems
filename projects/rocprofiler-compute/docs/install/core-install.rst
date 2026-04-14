.. meta::
   :description: ROCm Compute Profiler installation and deployment
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, tool, Instinct, accelerator, AMD,
              install, deploy, client, configuration, modulefiles

**********************************************
Installing and deploying ROCm Compute Profiler
**********************************************

* :ref:`ROCm Compute Profiler core installation <core-install>`

  * Provides the core application profiling capability.
  * Allows the collection of performance counters, filtering by hardware
    block, dispatch, kernel, and more.
  * Provides a CLI-based analysis mode.
  * Provides a standalone web interface for importing analysis metrics.

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

    * - ``rocprofiler-sdk_DIR``
      - Specifies the path to the rocprofiler-sdk CMake package configuration directory used to build the rocprofiler-compute counter collection tool.
        This directory should contain ``rocprofiler-sdkConfig.cmake`` (for example, ``<rocprofiler-sdk-install-path>/lib/cmake/rocprofiler-sdk``).

    * - ``STANDALONEBINARY_EXTRACT_DIR``
      - Specifies an optional temporary path to be used for extraction by the ROCm Compute Profiler standalone binary.

    * - ``STANDALONEBINARY``
      - Should be ON to enable the build of a standalone binary for ROCm Compute Profiler.

    * - ``TEST_FROM_INSTALL``
      - Should be ON to enable testing from the installation location without dependency on the source directory.

.. _core-install-steps:

Install from the TheRock nightly releases
-----------------------------------------

#. For detailed instructions on installing TheRock nightly release artifacts, refer to `TheRock/Release <https://github.com/ROCm/TheRock/blob/main/RELEASES.md>`_.


.. _source-install:

Install from the source
-----------------------

#. Sparse clone the repository `<https://github.com/ROCm/rocm-systems>`_ to get the ROCm Compute Profiler source code.

   .. code-block:: shell

      git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
      cd rocm-systems
      git sparse-checkout init --cone
      git sparse-checkout set projects/rocprofiler-compute
      git checkout develop

#. Navigate to the `rocprofiler-compute` project root.

   .. datatemplate:nodata::

      .. code-block:: shell

         cd projects/rocprofiler-compute

#. Install Python dependencies in a virtual environment, complete the ROCm Compute Profiler configuration and install process.

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
         make -j$(nproc) install

   .. tip::

      You might need to ``sudo`` the final installation step if you don't have
      write access for the chosen installation path.

#. Upon successful installation, your top-level installation directory should
   look like this.

   .. datatemplate:nodata::

      .. code-block:: shell

         $ ls $INSTALL_DIR
         modulefiles  {{ config.version }}  python-libs

Install from the tarball
------------------------

#. Download the rocprofiler-compute specific tarball for the latest release from `<https://github.com/ROCm/rocm-systems/releases>`_.
#. Untar the downloaded tarball and navigate to the `rocprofiler-compute` directory.
#. Follow the installation steps under :ref:`source-install`.

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

.. _core-install-rocprof-var:

Configuring the environment for ROCprofiler-SDK
-----------------------------------------------

ROCm Compute Profiler profiling process relies on :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>`'s ``rocprofiler-sdk`` library.
Optionally, a ``rocprofv3`` binary can be used in substitution of rocprofiler-sdk library when ``ROCPROF`` environment variable is set to ``rocprofv3`` or to the path of ``rocprofv3`` binary.
