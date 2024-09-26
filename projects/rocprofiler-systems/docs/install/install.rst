.. meta::
   :description: Omnitrace documentation and reference
   :keywords: Omnitrace, ROCm, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

*************************************
Omnitrace installation
*************************************

The following information builds on the guidelines in the :doc:`Quick start <./quick-start>` guide.
It covers how to install `Omnitrace <https://github.com/ROCm/omnitrace>`_ from source or a binary distribution,
as well as the :ref:`post-installation-steps`.

If you have problems using Omnitrace after installation,
consult the :ref:`post-installation-troubleshooting` section.

Release links
========================================

To review and install either the current Omnitrace release or earlier releases, use these links:

* Latest Omnitrace Release: `<https://github.com/ROCm/omnitrace/releases/latest>`_
* All Omnitrace Releases: `<https://github.com/ROCm/omnitrace/releases>`_

Operating system support
========================================

Omnitrace is only supported on Linux. The following distributions are tested in the Omnitrace GitHub workflows:

* Ubuntu 20.04
* Ubuntu 22.04
* OpenSUSE 15.3
* OpenSUSE 15.4
* Red Hat 8.7
* Red Hat 9.0
* Red Hat 9.1

Other OS distributions might function but are not supported or tested.

Identifying the operating system
-----------------------------------

If you are unsure of the operating system and version, the ``/etc/os-release`` and
``/usr/lib/os-release`` files contain operating system identification data for Linux systems.

.. code-block:: shell

   $ cat /etc/os-release

.. code-block:: shell

   NAME="Ubuntu"
   VERSION="20.04.4 LTS (Focal Fossa)"
   ID=ubuntu
   ...
   VERSION_ID="20.04"
   ...

The relevant fields are ``ID`` and the ``VERSION_ID``.

Architecture
========================================

With regards to instrumentation, at present only AMD64 (x86_64) architectures are tested. However,
Dyninst supports several more architectures and Omnitrace instrumentation may support other
CPU architectures such as aarch64 and ppc64.
Other modes of use, such as sampling and causal profiling, are not dependent on Dyninst and therefore
might be more portable.

Installing Omnitrace from binary distributions
================================================

Every Omnitrace release provides binary installer scripts of the form:

.. code-block:: shell

   omnitrace-{VERSION}-{OS_DISTRIB}-{OS_VERSION}[-ROCm-{ROCM_VERSION}[-{EXTRA}]].sh

For example,

.. code-block:: shell

   omnitrace-1.0.0-ubuntu-18.04-OMPT-PAPI-Python3.sh
   omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI-Python3.sh
   ...
   omnitrace-1.0.0-ubuntu-20.04-ROCm-50000-OMPT-PAPI-Python3.sh

Any of the ``EXTRA`` fields with a CMake build option
(for example, PAPI, as referenced in a following section) or
with no link requirements (such as OMPT) have
self-contained support for these packages.

To install Omnitrace using a binary installer script, follow these steps:

#. Download the appropriate binary distribution

   .. code-block:: shell

      wget https://github.com/ROCm/omnitrace/releases/download/v<VERSION>/<SCRIPT>

#. Create the target installation directory

   .. code-block:: shell

      mkdir /opt/omnitrace

#. Run the installer script

   .. code-block:: shell

      ./omnitrace-1.0.0-ubuntu-18.04-ROCm-405000-OMPT-PAPI.sh --prefix=/opt/omnitrace --exclude-subdir

Installing Omnitrace from source
========================================

Omnitrace needs a GCC compiler with full support for C++17 and CMake v3.16 or higher.
The Clang compiler may be used in lieu of the GCC compiler if `Dyninst <https://github.com/dyninst/dyninst>`_
is already installed.

Build requirements
-----------------------------------

* GCC compiler v7+

  * Older GCC compilers may be supported but are not tested
  * Clang compilers are generally supported for Omnitrace but not Dyninst

* `CMake <https://cmake.org/>`_ v3.16+

  .. note::

     * If the installed version of CMake is too old, installing a new version of CMake can be done through several methods
     * One of the easiest options is to use the python ``pip`` utility, as follows:

     .. code-block:: shell

        pip install --user 'cmake==3.18.4'
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

Any of the third-party packages required by Dyninst, along with Dyninst itself, can be built and installed
during the Omnitrace build. The following list indicates the package, the version,
the application that requires the package (for example, Omnitrace requires Dyninst
while Dyninst requires TBB), and the CMake option to build the package alongside Omnitrace:

.. csv-table::
   :header: "Third-Party Library", "Minimum Version", "Required By", "CMake Option"
   :widths: 15, 10, 12, 40

   "Dyninst", "12.0", "Omnitrace", "``OMNITRACE_BUILD_DYNINST`` (default: OFF)"
   "Libunwind", "", "Omnitrace", "``OMNITRACE_BUILD_LIBUNWIND`` (default: ON)"
   "TBB", "2018.6", "Dyninst", "``DYNINST_BUILD_TBB`` (default: OFF)"
   "ElfUtils", "0.178", "Dyninst", "``DYNINST_BUILD_ELFUTILS`` (default: OFF)"
   "LibIberty",  "", "Dyninst", "``DYNINST_BUILD_LIBIBERTY`` (default: OFF)"
   "Boost",  "1.67.0", "Dyninst", "``DYNINST_BUILD_BOOST`` (default: OFF)"
   "OpenMP", "4.x", "Dyninst", ""

Optional third-party packages
-----------------------------------

* `ROCm <https://rocm.docs.amd.com/projects/install-on-linux/en/latest>`_

  * HIP
  * Roctracer for HIP API and kernel tracing
  * ROCM-SMI for GPU monitoring
  * Rocprofiler for GPU hardware counters

* `PAPI <https://icl.utk.edu/papi/>`_
* MPI

  * ``OMNITRACE_USE_MPI`` enables full MPI support
  * ``OMNITRACE_USE_MPI_HEADERS`` enables wrapping of the dynamically-linked MPI C function calls.
    (By default, if Omnitrace cannot find an OpenMPI MPI distribution, it uses a local copy
    of the OpenMPI ``mpi.h``.)

* Several optional third-party profiling tools supported by Timemory
  (for example, `Caliper <https://github.com/LLNL/Caliper>`_, `TAU <https://www.cs.uoregon.edu/research/tau/home.php>`_, CrayPAT, and others)

.. csv-table::
   :header: "Third-Party Library", "CMake Enable Option", "CMake Build Option"
   :widths: 15, 45, 40

   "PAPI", "``OMNITRACE_USE_PAPI`` (default: ON)", "``OMNITRACE_BUILD_PAPI`` (default: ON)"
   "MPI", "``OMNITRACE_USE_MPI`` (default: OFF)", ""
   "MPI (header-only)", "``OMNITRACE_USE_MPI_HEADERS`` (default: ON)", ""

Installing Dyninst
-----------------------------------

The easiest way to install Dyninst is alongside Omnitrace, but it can also be installed using Spack.

Building Dyninst alongside Omnitrace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To install Dyninst alongside Omnitrace, configure Omnitrace with ``OMNITRACE_BUILD_DYNINST=ON``.
Depending on the version of Ubuntu, the ``apt`` package manager might have current enough
versions of the Dyninst Boost, TBB, and LibIberty dependencies
(use ``apt-get install libtbb-dev libiberty-dev libboost-dev``).
However, it is possible to request Dyninst to install
its dependencies via ``DYNINST_BUILD_<DEP>=ON``, as follows:

.. code-block:: shell

   git clone https://github.com/ROCm/omnitrace.git omnitrace-source
   cmake -B omnitrace-build -DOMNITRACE_BUILD_DYNINST=ON -DDYNINST_BUILD_{TBB,ELFUTILS,BOOST,LIBIBERTY}=ON omnitrace-source

where ``-DDYNINST_BUILD_{TBB,BOOST,ELFUTILS,LIBIBERTY}=ON`` is expanded by
the shell to ``-DDYNINST_BUILD_TBB=ON -DDYNINST_BUILD_BOOST=ON ...``

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

Installing Omnitrace
-----------------------------------

Omnitrace has CMake configuration options for MPI support (``OMNITRACE_USE_MPI`` or
``OMNITRACE_USE_MPI_HEADERS``), HIP kernel tracing (``OMNITRACE_USE_ROCTRACER``),
ROCm device sampling (``OMNITRACE_USE_ROCM_SMI``), OpenMP-Tools (``OMNITRACE_USE_OMPT``),
hardware counters via PAPI (``OMNITRACE_USE_PAPI``), among other features.
Various additional features can be enabled via the
``TIMEMORY_USE_*`` `CMake options <https://timemory.readthedocs.io/en/develop/installation.html#cmake-options>`_.
Any ``OMNITRACE_USE_<VAL>`` option which has a corresponding ``TIMEMORY_USE_<VAL>``
option means that the Timemory support for this feature has been integrated
into Perfetto support for Omnitrace, for example, ``OMNITRACE_USE_PAPI=<VAL>`` also configures
``TIMEMORY_USE_PAPI=<VAL>``. This means the data that Timemory is able to collect via this package
is passed along to Perfetto and is displayed when the ``.proto`` file is visualized
in `the Perfetto UI <https://ui.perfetto.dev>`_.

.. important::
   Perfetto validation is done with trace_processor v46.0 as there is a known issue with v47.0.
   If you are experiencing problems viewing your trace in the latest version of `Perfetto <http://ui.perfetto.dev>`_,
   then try using `Perfetto UI v46.0 <https://ui.perfetto.dev/v46.0-35b3d9845/#!/>`_.

.. code-block:: shell

   git clone https://github.com/ROCm/omnitrace.git omnitrace-source
   cmake                                       \
       -B omnitrace-build                      \
       -D CMAKE_INSTALL_PREFIX=/opt/omnitrace  \
       -D OMNITRACE_USE_HIP=ON                 \
       -D OMNITRACE_USE_ROCM_SMI=ON            \
       -D OMNITRACE_USE_ROCTRACER=ON           \
       -D OMNITRACE_USE_PYTHON=ON              \
       -D OMNITRACE_USE_OMPT=ON                \
       -D OMNITRACE_USE_MPI_HEADERS=ON         \
       -D OMNITRACE_BUILD_PAPI=ON              \
       -D OMNITRACE_BUILD_LIBUNWIND=ON         \
       -D OMNITRACE_BUILD_DYNINST=ON           \
       -D DYNINST_BUILD_TBB=ON                 \
       -D DYNINST_BUILD_BOOST=ON               \
       -D DYNINST_BUILD_ELFUTILS=ON            \
       -D DYNINST_BUILD_LIBIBERTY=ON           \
       omnitrace-source
   cmake --build omnitrace-build --target all --parallel 8
   cmake --build omnitrace-build --target install
   source /opt/omnitrace/share/omnitrace/setup-env.sh

.. _mpi-support-omnitrace:

MPI support within Omnitrace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Omnitrace can have full (``OMNITRACE_USE_MPI=ON``) or partial (``OMNITRACE_USE_MPI_HEADERS=ON``) MPI support.
The only difference between these two modes is whether or not the results collected
via Timemory and/or Perfetto can be aggregated into a single
output file during finalization. When full MPI support is enabled, combining the
Timemory results always occurs, whereas combining the Perfetto
results is configurable via the ``OMNITRACE_PERFETTO_COMBINE_TRACES`` setting.

The primary benefits of partial or full MPI support are the automatic wrapping
of MPI functions and the ability
to label output with suffixes which correspond to the ``MPI_COMM_WORLD`` rank ID
instead of having to use the system process identifier (i.e. ``PID``).
In general, it's recommended to use partial MPI support with the OpenMPI
headers as this is the most portable configuration.
If full MPI support is selected, make sure your target application is built
against the same MPI distribution as Omnitrace.
For example, do not build Omnitrace with MPICH and use it on a target application built against OpenMPI.
If partial support is selected, the reason the OpenMPI headers are recommended instead of the MPICH headers is
because the ``MPI_COMM_WORLD`` in OpenMPI is a pointer to ``ompi_communicator_t`` (8 bytes),
whereas ``MPI_COMM_WORLD`` in MPICH is an ``int`` (4 bytes). Building Omnitrace with partial MPI support
and the MPICH headers and then using
Omnitrace on an application built against OpenMPI causes a segmentation fault.
This happens because the value of the ``MPI_COMM_WORLD`` is truncated
during the function wrapping before being passed along to the underlying MPI function.

.. _post-installation-steps:

Post-installation steps
========================================

After installation, you can optionally configure the Omnitrace environment.
You should also test the executables to confirm Omnitrace is correctly installed.

Configure the environment
-----------------------------------

If environment modules are available and preferred, add them using these commands:

.. code-block:: shell

   module use /opt/omnitrace/share/modulefiles
   module load omnitrace/1.0.0

Alternatively, you can directly source the ``setup-env.sh`` script:

.. code-block:: shell

   source /opt/omnitrace/share/omnitrace/setup-env.sh

Test the executables
-----------------------------------

Successful execution of these commands confirms that the installation does not have any
issues locating the installed libraries:

.. code-block:: shell

   omnitrace-instrument --help
   omnitrace-avail --help

.. note::

   If ROCm support is enabled, you might have to add the path to the ROCm libraries to ``LD_LIBRARY_PATH``,
   for example, ``export LD_LIBRARY_PATH=/opt/rocm/lib:${LD_LIBRARY_PATH}``.

.. _post-installation-troubleshooting:

Post-installation troubleshooting
========================================

This section explains how to resolve certain issues that might happen when you first use Omnitrace.

Issues with RHEL and SELinux
----------------------------------------------------

RHEL (Red Hat Enterprise Linux) and related distributions of Linux automatically enable a security feature
named SELinux (Security-Enhanced Linux) that prevents Omnitrace from running.
This issue applies to any Linux distribution with SELinux installed, including RHEL,
CentOS, Fedora, and Rocky Linux. The problem can happen with any GPU, or even without a GPU.

The problem occurs after you instrument a program and try to
run ``omnitrace-run`` with the instrumented program.

.. code-block:: shell

   g++ hello.cpp -o hello
   omniperf-instrument -M sampling -o hello.instr -- ./hello
   omnitrace-run -- ./hello.instr

Instead of successfully running the binary with call-stack sampling,
Omnitrace crashes with a segmentation fault.

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
For more information, see the :ref:`omnitrace_papi_events` section.