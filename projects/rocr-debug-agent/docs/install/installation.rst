.. meta::
   :description: Installation instructions for ROCr Debug Agent
   :keywords: rocm, rocr debug agent, install, debugger, tool

.. _installation:

************************
Install ROCr Debug Agent
************************

Before you begin, verify that your system is supported. For more information,
see :ref:`ROCm Core SDK components <rocm:release-components>`.

For advanced workflows, source builds, or custom configurations, see
`Build ROCdebug-agent library <https://github.com/ROCm/rocm-systems/blob/develop/projects/rocr-debug-agent/README.md#build-the-rocdebug-agent-library>`_.

.. _install-rocm:

Install the ROCm Core SDK
=========================

ROCr Debug Agent is included with the ROCm Core SDK on Linux. For the most
complete installation, we recommend that developers use the
``amdrocm-core-sdk`` meta package.

For instructions, see :doc:`Install AMD ROCm <rocm:install/rocm>`. Use the
selector panel on that page to view instructions appropriate for your system
environment.

.. _install-base:

Install ROCm debuggers on Linux
===============================

Alternatively, if you want to install ROCr Debug Agent as part of the ROCm
Debugger package (a subset of the ROCm Core SDK ``amdrocm-core-sdk``) without
additional ROCm libraries and tools, install the ``amdrocm-debugger`` package.
This includes the ROCm debuggers, dependencies, and base packages.

1. Complete the :doc:`ROCm installation prerequisites <rocm:install/rocm>` to
   install dependencies and configure GPU access permissions.

2. Install the ROCm Debugger package that matches your desired ROCm version.
   Package names use the following format:

   .. code-block:: shell-session

      amdrocm-debugger<rocm_version>

   Where ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this
   suffix to install the latest available version.

   For example, to install the latest ROCm Debugger package release for
   supported GPU architectures:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install amdrocm-debugger

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install amdrocm-debugger

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install amdrocm-debugger

.. _install-nightly:

Install a nightly build
=======================

The `TheRock <https://github.com/ROCm/TheRock>`__ build system also publishes
nightly builds for the ROCm Core SDK and its components, including ROCr Debug
Agent. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
