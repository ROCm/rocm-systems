.. meta::
   :description: Install hipFile
   :keywords: hipFile, install, ROCm, build, CMake, GPU I/O, AMD, direct storage

.. _hipfile-installation:

****************
Install hipFile
****************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For source builds, CMake options, and sparse-checkout layout from ``rocm-systems``,
see :doc:`./build-from-source`. For the Python bindings after the C library is on
the machine, see :doc:`./python-bindings`.

.. _hipfile-install-rocm:

Install the ROCm Core SDK
=========================

hipFile ships with the ROCm Core SDK on Linux. For the broadest set of
components in one step, install the ``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to match your distribution and hardware.

.. _hipfile-install-linux:

Install hipFile on Linux
========================

If you want hipFile as a smaller ``amdrocm-*`` group instead of installing the full
``amdrocm-core-sdk`` stack, install the group that carries the hipFile runtime and
headers for your ROCm release. Package names follow this pattern:

.. code-block:: shell-session

   amdrocm-<group><-dev/-devel><rocm_version><-llvm_target>

Where:

* ``<-dev/-devel>`` selects library files and headers. Omit the suffix for
  runtime-only packages.

  * ``-dev`` applies on Debian-based distributions, including Ubuntu.

  * ``-devel`` applies on RPM-based distributions, including RHEL and SLES.

* ``<rocm_version>`` pins the ROCm Core SDK version. Omit it to track the latest
  release your repository publishes.

* ``<-llvm_target>`` starting with ``gfx`` limits the install to one AMD GPU
  architecture. Omit it to pull every supported architecture at higher disk
  cost.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` so
   dependencies and GPU access permissions are in place.

2. Install the ``amdrocm-*`` group that matches your ROCm version, development
   package needs, and GPU architecture. The exact ``<group>`` string for hipFile
   can change between ROCm releases. Confirm the name in the release notes for
   your target version before you run the package manager.

3. Run the install command for your distribution. Replace ``<group>`` with the
   value from step 2.

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code:: shell

            sudo apt update
            sudo apt install amdrocm-<group>-dev

      .. tab-item:: RHEL-based distros

         .. code:: shell

            sudo dnf install amdrocm-<group>-devel

      .. tab-item:: SLES

         .. code:: shell

            sudo zypper install amdrocm-<group>-devel

.. _hipfile-install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system publishes nightly
builds for the ROCm Core SDK and its components. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for download links and
support notes.
