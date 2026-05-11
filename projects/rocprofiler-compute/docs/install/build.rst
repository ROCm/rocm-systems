.. meta::
   :description: ROCm Compute Profiler installation and deployment
   :keywords: Omniperf, ROCm Compute Profiler, ROCm, tool, Instinct, accelerator, AMD,
              install, deploy, client, configuration, modulefiles

**********************************************
Installing and deploying ROCm Compute Profiler
**********************************************

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

.. _core-install-cmake-vars:

Configuration variables
-----------------------

The following installation steps leverage several
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
      - Specifies the path to the ROCprofiler-SDK CMake package configuration directory used to build the rocprofiler-compute counter collection tool.
        This directory should contain ``rocprofiler-sdkConfig.cmake`` (for example, ``<rocprofiler-sdk-install-path>/lib/cmake/rocprofiler-sdk``).

    * - ``STANDALONEBINARY_EXTRACT_DIR``
      - Specifies an optional temporary path to be used for extraction by the ROCm Compute Profiler standalone binary.

    * - ``STANDALONEBINARY``
      - Should be ON to enable the build of a standalone binary for ROCm Compute Profiler.

    * - ``TEST_FROM_INSTALL``
      - Should be ON to enable testing from the installation location without dependency on the source directory.

    * - ``SKIP_NATIVE_TOOL_BUILD``
      - Should be ON to skip building the native profiling tool. When enabled, the native tool will be compiled at runtime instead of build time. This is useful when ROCprofiler-SDK is not available during build time.

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

.. _tarball-install:

Install from the tarball
------------------------

#. Download the rocprofiler-compute specific tarball for the latest release from `<https://github.com/ROCm/rocm-systems/releases>`_.
#. Untar the downloaded tarball and navigate to the `rocprofiler-compute` directory.
#. Follow the installation steps under :ref:`source-install`.

Execution using modulefiles
---------------------------

.. _core-install-modulefiles:

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
      ----------------------------------------
      rocprofiler-compute version: {{ config.version }} (release)
      Git revision:     abc1234
      ----------------------------------------

.. tip::

   If you're relying on an Lmod Python module locally, you may wish to customize
   the resulting ROCm Compute Profiler modulefile post-installation to include extra
   module dependencies.

Execution without modulefiles
------------------------------

To use ROCm Compute Profiler without the companion modulefile, update your ``PATH``
settings to enable access to the command line binary. If you installed Python
dependencies in a shared location, also update your ``PYTHONPATH``
configuration.

.. datatemplate:nodata::

   .. code-block:: shell

      export PATH=$INSTALL_DIR/{{ config.version }}/bin:$PATH
      export PYTHONPATH=$INSTALL_DIR/python-libs

.. tip::

   To always run ROCm Compute Profiler with a particular version of Python, you can create a
   bash alias. For example, to run ROCm Compute Profiler with Python 3.8, you can run the
   following command:

   .. code-block:: shell

      alias rocprof-compute-mypython="/usr/bin/python3.8 /opt/rocm/bin/rocprof-compute"

.. _core-install-rocprof-var:

Configuring the environment for profiling
-----------------------------------------

ROCm Compute Profiler supports two profiling backends, selectable via the ``ROCPROF`` environment variable.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Backend
     - How it is selected
     - How it works
   * - ``rocprofiler-sdk`` (**default**)
     - ``ROCPROF`` unset, or ``ROCPROF=rocprofiler-sdk``
     - Injects ``librocprofiler-sdk-tool.so`` into the target application process via ``LD_PRELOAD``. The application runs directly; profiling is configured through environment variables.
   * - ``rocprofv3``
     - ``ROCPROF=rocprofv3`` or ``ROCPROF=<path-to-rocprofv3>``
     - Launches the ``rocprofv3`` binary as a wrapper process around the target application. Profiling is configured via ``rocprofv3`` command-line arguments.

Both backends build on the same underlying ROCprofiler-SDK infrastructure. The ``rocprofiler-sdk`` backend is recommended because it supports the full feature set, including :ref:`iteration multiplexing <iteration-multiplexing>`.

.. _core-install-native-tool:

Native counter collection tool
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When using the ``rocprofiler-sdk`` backend on ROCm 7.0 or later, ROCm Compute Profiler also injects a native counter collection tool (``librocprofiler-compute-tool.so``) alongside the SDK tool via ``LD_PRELOAD``. This tool is a shared library built as part of ROCm Compute Profiler that directly uses the ROCprofiler-SDK public C API to collect hardware performance counter data per kernel dispatch.

The division of responsibility between the two injected libraries is:

* **Native tool** (``librocprofiler-compute-tool.so``): collects hardware performance counters per dispatch.
* **SDK tool** (``librocprofiler-sdk-tool.so``): handles kernel tracing and output database generation.

The native tool is required for :ref:`iteration multiplexing <iteration-multiplexing>`. Use ``--no-native-tool`` to disable it, but note that doing so also disables iteration multiplexing. The native tool is not used in :doc:`dynamic process attachment mode <../how-to/live_attach_detach>` or with the ``rocprofv3`` backend.
