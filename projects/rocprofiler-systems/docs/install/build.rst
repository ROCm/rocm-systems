.. meta::
   :description: ROCm Systems Profiler installation documentation and reference
   :keywords: rocprof-sys, rocprofiler-systems, Omnitrace, ROCm, installation, installer, profiler, tracking, visualization, tool, Instinct, accelerator, AMD

***************************************
Build ROCm Systems Profiler from source
***************************************

To build ROCm Systems Profiler as part of the ROCm Core SDK, see `TheRock build
instructions
<https://github.com/ROCm/TheRock/blob/main/README.md#build-configuration>`__.
TheRock is the recommended way to build ROCm components from source.

To build ROCm Systems Profiler standalone, without TheRock, use the
instructions on this page.

.. seealso::

   If you encounter problems after installation, consult the
   :ref:`post-installation-troubleshooting` section.

Operating system support
========================

ROCm Systems Profiler is only supported on Linux. For more information, see
:ref:`ROCm Core SDK components <rocm:release-components>`.

Identifying the operating system
--------------------------------

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

Build ROCm Systems Profiler from source
=======================================

ROCm Systems Profiler needs a GCC compiler with full support for C++20 and CMake v3.25 or higher.
The Clang compiler may be used.

Build requirements
------------------

* GCC compiler v11+

  * GCC 11 is the first release with reasonably complete C++20 support. GCC 10 is
    missing ``using enum``, ``std::source_location``, ``std::bit_cast``, and
    ``<latch>``/``<barrier>``/``<semaphore>``, and is no longer tested.
  * On RHEL 8, the system GCC is too old; use ``gcc-toolset-11`` or later
  * Older GCC compilers may still work but are not tested and are not supported
  * Clang compilers are also supported for ROCm Systems Profiler

* `CMake <https://cmake.org/>`_ v3.25 or later

  .. note::
     If the ``CMake`` installed on the system is too old, you can install a new
     version using various methods. One of the easiest options is to use PyPi (Python's pip).

     .. code-block:: shell

        pip install --user 'cmake==3.25.0'
        export PATH=${HOME}/.local/bin:${PATH}

* `Ninja <https://ninja-build.org/>`_ (required by the CMake presets in this project,
  unless you override the generator)

Required third-party packages
-----------------------------

* `Dyninst <https://github.com/dyninst/dyninst>`_ for dynamic or static instrumentation.
  Dyninst uses the following required and optional components.

  * `TBB <https://github.com/oneapi-src/oneTBB>`_ (required)
  * `Elfutils <https://sourceware.org/elfutils/>`_ (required)
  * `Libiberty <https://github.com/gcc-mirror/gcc/tree/master/libiberty>`_ (required)
  * `OpenMP <https://www.openmp.org/>`_ (optional)

  The Dyninst sources bundled with ROCm Systems Profiler do not use Boost.
  If you build against an external, older Dyninst install instead, that layout may still require Boost development packages.

* `libunwind <https://www.nongnu.org/libunwind/>`_ for call-stack sampling
* `SQLite <https://github.com/sqlite/sqlite>`_ for database output
* `spdlog <https://github.com/gabime/spdlog>`_ for logging

Any of the third-party packages required by Dyninst, along with Dyninst itself, can be built and installed
during the ROCm Systems Profiler build. The following list indicates the package, the version,
the application that requires the package (for example, ROCm Systems Profiler requires Dyninst
while Dyninst requires TBB), and the CMake option to build the package alongside ROCm Systems Profiler:

.. csv-table::
   :header: "Third-Party Library", "Minimum Version", "Required By", "CMake Option"

   "Dyninst", "13.0", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_DYNINST`` (default: OFF)"
   "Libunwind", "", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_LIBUNWIND`` (default: ON)"
   "Nlohmann/JSON", "", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_NLOHMANN_JSON`` (default: ON)"
   "spdlog", "", "ROCm Systems Profiler", "``ROCPROFSYS_BUILD_SPDLOG`` (default: ON)"
   "TBB", "2018.6", "Dyninst", "``ROCPROFSYS_BUILD_TBB`` (default: OFF)"
   "ElfUtils", "0.178", "Dyninst", "``ROCPROFSYS_BUILD_ELFUTILS`` (default: OFF)"
   "LibIberty",  "", "Dyninst", "``ROCPROFSYS_BUILD_LIBIBERTY`` (default: OFF)"
   "OpenMP", "4.x", "Dyninst", ""

The CMake presets described later on this page turn on in-tree Dyninst, TBB, Elfutils, and
LibIberty builds even though the CMake defaults for those options are ``OFF``.

ROCm dependencies
-----------------

ROCm is required for GPU profiling features such as GPU hardware counter
collection, tracing, and GPU and AI NIC monitoring.

* :doc:`ROCm <rocm:install/rocm>`

  * :doc:`AMD SMI library <amdsmi:index>` for GPU and AI NIC monitoring

  * :doc:`ROCprofiler-SDK <rocprofiler-sdk:index>` for GPU hardware counters
    and ROCm tracing

Optional third-party packages
-----------------------------

The following packages are optional and can be enabled via the corresponding
CMake options.

* Python

  * ``ROCPROFSYS_USE_PYTHON`` enables Python support.

* `PAPI <https://icl.utk.edu/papi/>`__
* MPI

  * ``ROCPROFSYS_USE_MPI`` enables full MPI support
  * ``ROCPROFSYS_USE_MPI_HEADERS`` enables wrapping of the dynamically-linked MPI C function calls.
    (By default, if ROCm Systems Profiler cannot find an OpenMPI MPI distribution, it uses a local copy
    of the OpenMPI ``mpi.h``.)

.. csv-table::
   :header: "Third-Party Library", "CMake Enable Option"
   :widths: 15, 45

   "PAPI", "``ROCPROFSYS_USE_PAPI`` (default: ON)"
   "MPI", "``ROCPROFSYS_USE_MPI`` (default: OFF)"
   "MPI (header-only)", "``ROCPROFSYS_USE_MPI_HEADERS`` (default: ON)"

Get the source
--------------

Clone the ``rocm-systems`` repository and check out only the ROCm Systems Profiler project:

.. code-block:: shell

   git clone --filter=blob:none --sparse https://github.com/ROCm/rocm-systems.git
   git -C rocm-systems sparse-checkout set projects/rocprofiler-systems

CMake presets live in ``projects/rocprofiler-systems/CMakePresets.json``. Run the
configure and build commands from that directory.

Configure and build with CMake presets
--------------------------------------

The project ships configure presets for common developer and CI layouts. All visible
presets inherit a hidden ``default`` preset that selects the Ninja generator, the
``gcc``/``g++`` compilers, install prefix ``/opt/rocprofiler-systems``, Python support,
and in-tree builds of Dyninst, TBB, Elfutils, and LibIberty.

List the available presets, then configure, build, and install. The file defines
configure presets only, so pass the build directory (not ``--preset``) to
``cmake --build`` and ``cmake --install``.

.. code-block:: shell

   cd rocm-systems/projects/rocprofiler-systems
   cmake --list-presets
   cmake --preset release
   cmake --build build/release --parallel
   cmake --install build/release
   source /opt/rocprofiler-systems/share/rocprofiler-systems/setup-env.sh

Replace ``release`` and ``build/release`` with another preset name and its
binary directory from the table below.

.. csv-table::
   :header: "Preset", "Binary directory", "Description"
   :widths: 18, 28, 54

   "``release``", "``build/release``", "Official Release build"
   "``debug``", "``build/debug``", "Debug build with tests and examples"
   "``debug-optimized``", "``build/debug-optimized``", "RelWithDebInfo build with tests and examples"
   "``release-mpi``", "``build/release-mpi``", "``release`` plus full MPI (``ROCPROFSYS_USE_MPI=ON``)"
   "``debug-mpi``", "``build/debug-mpi``", "``debug`` plus full MPI (``ROCPROFSYS_USE_MPI=ON``)"

To override a cache variable from a preset, pass extra ``-D`` options after
``--preset``. For example, build Python support for more than one interpreter:

.. code-block:: shell

   cmake --preset release -D ROCPROFSYS_PYTHON_ROOT_DIRS="/usr/bin;/usr/bin" -D ROCPROFSYS_PYTHON_VERSIONS="3.10;3.12"

When you set ``ROCPROFSYS_PYTHON_VERSIONS`` and ``ROCPROFSYS_PYTHON_ROOT_DIRS``,
the two lists must be the same length.

.. _cmake-options:

Additional CMake options
------------------------

ROCm support is always enabled. You can still pass extra ``-D`` flags with a
preset, including OpenMP-Tools (``ROCPROFSYS_USE_OMPT``) and hardware counters via
PAPI (``ROCPROFSYS_USE_PAPI``).

Python profiling is enabled by the presets (``ROCPROFSYS_USE_PYTHON=ON``).
Use ``ROCPROFSYS_PYTHON_VERSIONS`` and, for multiple versions,
``ROCPROFSYS_PYTHON_ROOT_DIRS`` as shown in the override example above.

Post-installation
=================

See :ref:`post-installation-steps` and :ref:`post-installation-troubleshooting`
for more information.
