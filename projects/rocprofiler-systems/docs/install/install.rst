.. meta::
   :description: ROCm Systems Profiler installation documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, installation, installer, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
ROCm Systems Profiler installation
*************************************

If you have problems using ROCm Systems Profiler after installation,
consult the :ref:`post-installation-troubleshooting` section.


Operating system support
========================================

ROCm Systems Profiler is only supported on Linux. See
`Supported operating systems <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems>`_
for ROCm supported operating systems.

Identifying the operating system
-----------------------------------

If you are unsure of the Linux distribution and version, the ``/etc/os-release`` and
``/usr/lib/os-release`` files contain this information.

.. code-block:: shell

   $ cat /etc/os-release
   NAME="Ubuntu"
   VERSION_ID="24.04"
   VERSION="24.04.3 LTS (Noble Numbat)"
   VERSION_CODENAME=noble
   ID=ubuntu

The relevant fields are ``ID`` and the ``VERSION_ID``.

Install via package manager
============================

   If you have ROCm version 6.3 or higher installed, you can use the
   package manager to install a pre-built copy of ROCm Systems Profiler.

.. tab-set::

   .. tab-item:: Ubuntu

      .. code-block:: shell

         $ sudo apt install rocprofiler-systems

   .. tab-item:: Red Hat Enterprise Linux

      .. code-block:: shell

         $ sudo dnf install rocprofiler-systems

   .. tab-item:: SUSE Linux Enterprise Server

      .. code-block:: shell

         $ sudo zypper install rocprofiler-systems

Building ROCm Systems Profiler from source
==========================================

ROCm Systems Profiler needs a GCC compiler with full support for C++17 and CMake v3.21 or higher.
The Clang compiler may be used instead of the GCC compiler if `Dyninst <https://github.com/dyninst/dyninst>`_
is already installed.

Build requirements
-----------------------------------

* GCC compiler v7+

  * Older GCC compilers may be supported but are not tested
  * Clang compilers are generally supported for ROCm Systems Profiler but not Dyninst

* `CMake <https://cmake.org/>`_ v3.21 or later

  .. note::
     If the ``CMake`` installed on the system is too old, you can install a new
     version using various methods. One of the easiest options is to use PyPi (Python's pip).

     .. code-block:: shell

        pip install --user 'cmake==3.21.0'
        export PATH=${HOME}/.local/bin:${PATH}

Required third-party packages
-----------------------------------

* `Dyninst <https://github.com/dyninst/dyninst>`_ for dynamic or static instrumentation.
  Dyninst uses the following required and optional components.

  * `TBB <https://github.com/oneapi-src/oneTBB>`_ (required)
  * `Elfutils <https://sourceware.org/elfutils/>`_ (required)
  * `Libiberty <https://github.com/gcc-mirror/gcc/tree/master/libiberty>`_ (required)
  * `Boost <https://www.boost.org/>`_ (required)
  * `OpenMP <https://www.openmp.org/>`_ (optional)

* `libunwind <https://www.nongnu.org/libunwind/>`_ for call-stack sampling
* `SQLite <https://github.com/sqlite/sqlite>`_ for database output

Any of the third-party packages required by Dyninst, along with Dyninst itself, can be built and installed
during the ROCm Systems Profiler build. The following list indicates the package, the version,
the application that requires the package (for example, ROCm Systems Profiler requires Dyninst
while Dyninst requires TBB), and the CMake option to build the package alongside ROCm Systems Profiler:

.. csv-table::
   :header: "Third-Party Library", "Minimum Version", "Required By", "CMake Option"

   "Dyninst", "13.0", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_DYNINST`` (default: OFF)"
   "Libunwind", "", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_LIBUNWIND`` (default: ON)"
   "SQLite", "", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_SQLITE`` (default: OFF)"
   "TBB", "2018.6", "Dyninst", "``ROCPROFSYS_BUILD_TBB`` (default: OFF)"
   "ElfUtils", "0.178", "Dyninst", "``ROCPROFSYS_BUILD_ELFUTILS`` (default: OFF)"
   "LibIberty",  "", "Dyninst", "``ROCPROFSYS_BUILD_LIBIBERTY`` (default: OFF)"
   "Boost",  "1.67.0", "Dyninst", "``ROCPROFSYS_BUILD_BOOST`` (default: OFF)"
   "OpenMP", "4.x", "Dyninst", ""

Optional third-party packages
-----------------------------------

* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest>`_

  * AMD SMI Lib for GPU monitoring
  * ROCprofiler SDK for GPU hardware counters and ROCm tracing

* `PAPI <https://icl.utk.edu/papi/>`_
* MPI

  * ``ROCPROFSYS_USE_MPI`` enables full MPI support
  * ``ROCPROFSYS_USE_MPI_HEADERS`` enables wrapping of the dynamically-linked MPI C function calls.
    (By default, if ROCm Systems Profiler cannot find an OpenMPI MPI distribution, it uses a local copy
    of the OpenMPI ``mpi.h``.)

.. csv-table::
   :header: "Third-Party Library", "CMake Enable Option"
   :widths: 15, 45

   "ROCm", "``ROCPROFSYS_USE_ROCM`` (default: ON)"
   "PAPI", "``ROCPROFSYS_USE_PAPI`` (default: ON)"
   "MPI", "``ROCPROFSYS_USE_MPI`` (default: OFF)"
   "MPI (header-only)", "``ROCPROFSYS_USE_MPI_HEADERS`` (default: ON)"

Installing Dyninst
-----------------------------------

The easiest way to install Dyninst is alongside ROCm Systems Profiler, but it can also be installed using Spack.

Building Dyninst alongside ROCm Systems Profiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install Dyninst alongside ROCm Systems Profiler, configure ROCm Systems Profiler with ``ROCPROFSYS_BUILD_DYNINST=ON``.
Depending on the version of Ubuntu, the ``apt`` package manager might have current enough
versions of the Dyninst Boost, TBB, and LibIberty dependencies
(use ``apt-get install libtbb-dev libiberty-dev libboost-dev``).
However, it is possible to also build and install the Dyninst dependencies
via ``ROCPROFSYS_BUILD_<DEP>=ON``, as follows:

.. code-block:: shell

   git clone https://github.com/ROCm/rocm-systems.git
   cmake -B rocprof-sys-build -DROCPROFSYS_BUILD_DYNINST=ON \
         -DROCPROFSYS_BUILD_{TBB,ELFUTILS,BOOST,LIBIBERTY}=ON \
         -S rocm-systems/projects/rocprofiler-systems

where ``-DROCPROFSYS_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON`` is expanded by
the shell to ``-DROCPROFSYS_BUILD_TBB=ON -DROCPROFSYS_BUILD_BOOST=ON ...``

Installing Dyninst via Spack
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

`Spack <https://github.com/spack/spack>`_ is another option to install Dyninst and its dependencies:

.. code-block:: shell

   git clone https://github.com/spack/spack.git
   source ./spack/share/spack/setup-env.sh
   spack compiler find
   spack external find --all --not-buildable
   spack spec -I --reuse dyninst
   spack install --reuse dyninst
   spack load -r dyninst

.. _cmake-options:

Building and installing ROCm Systems Profiler
---------------------------------------------

ROCm Systems Profiler has CMake configuration options for MPI support (``ROCPROFSYS_USE_MPI`` or
``ROCPROFSYS_USE_MPI_HEADERS``), ROCm tracing and sampling (``ROCPROFSYS_USE_ROCM``),
OpenMP-Tools (``ROCPROFSYS_USE_OMPT``), hardware counters via PAPI (``ROCPROFSYS_USE_PAPI``),
among other features.
Various additional features can be enabled via the
``TIMEMORY_USE_*`` `CMake options <https://timemory.readthedocs.io/en/develop/installation.html#cmake-options>`_.
Any ``ROCPROFSYS_USE_<VAL>`` option which has a corresponding ``TIMEMORY_USE_<VAL>``
option means that the Timemory support for this feature has been integrated
into Perfetto support for ROCm Systems Profiler, for example, ``ROCPROFSYS_USE_PAPI=<VAL>`` also configures
``TIMEMORY_USE_PAPI=<VAL>``. This means the data that Timemory is able to collect via this package
is passed along to Perfetto and is displayed when the ``.proto`` file is visualized
in `the Perfetto UI <https://ui.perfetto.dev>`_.

.. code-block:: shell

   git clone https://github.com/ROCm/rocm-systems.git
   cmake                                                 \
       -B rocprof-sys-build                              \
       -D CMAKE_INSTALL_PREFIX=/opt/rocprofiler-systems  \
       -D ROCPROFSYS_USE_ROCM=ON                         \
       -D ROCPROFSYS_USE_PYTHON=ON                       \
       -D ROCPROFSYS_USE_OMPT=ON                         \
       -D ROCPROFSYS_USE_MPI_HEADERS=ON                  \
       -D ROCPROFSYS_BUILD_PAPI=ON                       \
       -D ROCPROFSYS_BUILD_LIBUNWIND=ON                  \
       -D ROCPROFSYS_BUILD_DYNINST=ON                    \
       -D ROCPROFSYS_BUILD_TBB=ON                        \
       -D ROCPROFSYS_BUILD_BOOST=ON                      \
       -D ROCPROFSYS_BUILD_ELFUTILS=ON                   \
       -D ROCPROFSYS_BUILD_LIBIBERTY=ON                  \
       -S rocm-systems/projects/rocprofiler-systems
   cmake --build rocprof-sys-build --target all --parallel 8
   cmake --build rocprof-sys-build --target install
   source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

.. _build-script:

Using the build script
^^^^^^^^^^^^^^^^^^^^^^

This method automates the CMake process with a script that wraps the CMake
commands and handles build logic, environment variables, and packaging. Run
``./scripts/build-release.sh`` with your desired options to generate packages.

Use ``./scripts/build-release.sh --help`` for more information.

.. code-block:: shell-session

   ./scripts/build-release.sh --help
   Options:
       --core       [+nopython] [+python]                    Core (Use '+nopython' to build w/o python, use '+python' to python build with python)
       --mpi        [+nopython] [+python]                    MPI (Use '+nopython' to build w/o python, use '+python' to python build with python)
       --rocm       [+nopython] [+python]                    ROCm (Use '+nopython' to build w/o python, use '+python' to python build with python)
       --rocm-mpi   [+nopython] [+python]                    ROCm + MPI (Use '+nopython' to build w/o python, use '+python' to python build with python)
       --mpi-impl   [openmpi|mpich]                          MPI implementation

       --lto                  [on|off]                       Enable LTO (default: off)
       --strip                [on|off]                       Strip libraries (default: off)
       --perfetto-tools       [on|off]                       Install perfetto tools (default: on)
       --static-libgcc        [on|off]                       Build with static libgcc (default: on)
       --static-libstdcxx     [on|off]                       Build with static libstdc++ (default: on)
       --hidden-visibility    [on|off]                       Build with hidden visibility (default: on)
       --max-threads          N                              Max number of threads supported (default: 2048)
       --parallel             N                              Number of parallel build jobs (default: 12)
       --generators           [STGZ][DEB][RPM][+others]      CPack generators (default: stgz deb rpm)

.. _mpi-support-rocprof-sys:

MPI support within ROCm Systems Profiler
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ROCm Systems Profiler can have full (``ROCPROFSYS_USE_MPI=ON``) or partial (``ROCPROFSYS_USE_MPI_HEADERS=ON``) MPI support.
The only difference between these two modes is whether or not the results collected
via Timemory and/or Perfetto can be aggregated into a single
output file during finalization. When full MPI support is enabled, combining the
Timemory results always occurs, whereas combining the Perfetto
results is configurable via the ``ROCPROFSYS_PERFETTO_COMBINE_TRACES`` setting.

The primary benefits of partial or full MPI support are the automatic wrapping
of MPI functions and the ability
to label output with suffixes which correspond to the ``MPI_COMM_WORLD`` rank ID
instead of having to use the system process identifier (i.e. ``PID``).
In general, it's recommended to use partial MPI support with the OpenMPI
headers as this is the most portable configuration.
If full MPI support is selected, make sure your target application is built
against the same MPI distribution as ROCm Systems Profiler.
For example, do not build ROCm Systems Profiler with MPICH and use it on a target application built against OpenMPI.
If partial support is selected, the reason the OpenMPI headers are recommended instead of the MPICH headers is
because the ``MPI_COMM_WORLD`` in OpenMPI is a pointer to ``ompi_communicator_t`` (8 bytes),
whereas ``MPI_COMM_WORLD`` in MPICH is an ``int`` (4 bytes). Building ROCm Systems Profiler with partial MPI support
and the MPICH headers and then using
ROCm Systems Profiler on an application built against OpenMPI causes a segmentation fault.
This happens because the value of the ``MPI_COMM_WORLD`` is truncated
during the function wrapping before being passed along to the underlying MPI function.

ROCm Systems Profiler without ROCm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To build ROCm Systems Profiler for use on systems without a GPU or the ROCm runtime, disable ROCm
support using the CMake configuration option ``ROCPROFSYS_USE_ROCM=OFF``. See :ref:`cmake-options`
for more information.

Alternatively, use the provided build script with the appropriate options. See :ref:`build-script`.
For example, to build without ROCm support and create a STGZ installer, use the following command:

.. code-block:: shell

   ./scripts/build-release.sh --core +python --generators STGZ

.. _post-installation-steps:

Post-installation steps
========================================

After installation, you can optionally configure the ROCm Systems Profiler environment.
You should also test the executables to confirm ROCm Systems Profiler is correctly installed.

Configure the environment
-----------------------------------

If environment modules are available and preferred, then add them using these commands,
 replacing ``1.0.0`` with the desired version number to load:

.. code-block:: shell

   module use /opt/rocprofiler-systems/share/modulefiles
   module load rocprofiler-systems/1.0.0

Alternatively, you can directly source the ``setup-env.sh`` script:

.. code-block:: shell

   source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

Test the executables
-----------------------------------

Successful execution of these commands confirms that the installation does not have any
issues locating the installed libraries:

.. code-block:: shell

   rocprof-sys-instrument --help
   rocprof-sys-avail --help

.. note::

   If ROCm support is enabled, you might have to add the path to the ROCm libraries to ``LD_LIBRARY_PATH``,
   for example, ``export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}``.

.. _post-installation-troubleshooting:

Post-installation troubleshooting
========================================

This section explains how to resolve certain issues that might happen when you first use ROCm Systems Profiler.

Issues with RHEL and SELinux
----------------------------------------------------

RHEL (Red Hat Enterprise Linux) and related distributions of Linux automatically enable a security feature
named SELinux (Security-Enhanced Linux) that prevents ROCm Systems Profiler from running.
This issue applies to any Linux distribution with SELinux installed, including RHEL,
CentOS, Fedora, and Rocky Linux. The problem can happen with any GPU, or even without a GPU.

The problem occurs after you instrument a program and try to
run ``rocprof-sys-run`` with the instrumented program.

.. code-block:: shell

   g++ hello.cpp -o hello
   rocprof-sys-instrument -M sampling -o hello.instr -- ./hello
   rocprof-sys-run -- ./hello.instr

Instead of successfully running the binary with call-stack sampling,
ROCm Systems Profiler crashes with a segmentation fault.

.. note::

   If you are physically logged in on the system (not using SSH or a remote connection),
   the operating system might display an SELinux pop-up warning in the notifications.

To workaround this problem, either disable SELinux or configure it to use a more
permissive setting.

To avoid this problem for the duration of the current session, run this command
from the shell:

.. code-block:: shell

   sudo setenforce 0

For a permanent workaround, edit the SELinux configuration file using the command
``sudo vim /etc/sysconfig/selinux`` and change the ``SELINUX`` setting to
either ``Permissive`` or ``Disabled``.

.. note::

   Permanently changing the SELinux settings can have security implications.
   Ensure you review your system security settings before making any changes.

Modifying RPATH details
----------------------------------------------------

If you're experiencing problems loading your application with an instrumented library,
then you might have to check and modify the RPATH specified in your application.
See the section on `troubleshooting RPATHs <../how-to/instrumenting-rewriting-binary-application.html#rpath-troubleshooting>`_
for further details.

Configuring PAPI to collect hardware counters
----------------------------------------------------

To use PAPI to collect the majority of hardware counters, ensure
the ``/proc/sys/kernel/perf_event_paranoid`` setting has a value less than or equal to ``2``.
For more information, see the :ref:`rocprof-sys_papi_events` section.
