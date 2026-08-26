.. meta::
   :description: Instruction on how to install the RCCL library for collective communication primitives using prebuilt packages
   :keywords: RCCL, ROCm, library, API, install

.. _install:

************
Install RCCL
************

RCCL is distributed as a prebuilt package with ROCm. The recommended way to
install RCCL is to install ROCm using the package manager for your Linux
distribution, which provides RCCL along with the rest of the ROCm stack.

Before you begin, verify that your system is supported. For more information,
see :doc:`ROCm components <rocm:what-is-rocm>`.

For advanced workflows, source builds, or custom configurations, see
:doc:`./building-installing`.

.. note::

   ROCm is transitioning to a new packaging scheme. The package names you use
   depend on which ROCm release stream you are installing:

   * **ROCm 7.2 and earlier (production):** packages use the ``rccl`` /
     ``rccl-dev`` naming scheme and are installed from the production ROCm
     repositories. See :ref:`install-rocm`.
   * **ROCm 7.13.0 technology preview (built with TheRock):** packages use the
     new ``amdrocm-*`` naming scheme. See :ref:`install-preview`.

.. _install-rocm:

Install RCCL with ROCm (recommended)
====================================

RCCL is included with ROCm on Linux. For the most complete installation, install
the full ROCm stack, which installs RCCL along with the HIP runtime and the other
ROCm libraries and tools.

#. Complete the :doc:`ROCm installation prerequisites
   <rocm-install-on-linux:install/quick-start>` to install dependencies and
   configure GPU access permissions.

#. Install ROCm using the meta package for your distribution. This installs RCCL
   (the ``rccl`` runtime package) as part of the ROCm stack:

   .. tab-set::

      .. tab-item:: Debian-based distros

         .. code-block:: bash

            sudo apt install rocm

      .. tab-item:: RHEL-based distros

         .. code-block:: bash

            sudo dnf install rocm

      .. tab-item:: SLES

         .. code-block:: bash

            sudo zypper install rocm

For complete, distribution-specific instructions, see the
:doc:`ROCm installation guide <rocm-install-on-linux:install/quick-start>`.

Install RCCL only
-----------------

If you want to install RCCL without the rest of the ROCm stack, install the RCCL
package directly. The package depends on the core ROCm runtime, which is pulled
in automatically.

* The runtime package is named ``rccl``.
* The development package, which adds the library headers, is named ``rccl-dev``
  on all supported distributions.

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt install rccl-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf install rccl-dev

   .. tab-item:: SLES

      .. code-block:: bash

         sudo zypper install rccl-dev

.. _install-preview:

Install RCCL from the ROCm technology preview
=============================================

The ROCm 7.13.0 technology preview is built with `TheRock
<https://github.com/ROCm/TheRock>`__ and uses a new ``amdrocm-*`` package naming
scheme. These packages are published from the ROCm preview repositories and are
intended for evaluation and development.

For the most complete installation, install the ``amdrocm-core-sdk`` meta
package, which provides the full ROCm Core SDK including RCCL.

To install RCCL on its own, install the ``amdrocm-rccl`` package. RCCL package
names use the following format:

.. code-block:: shell-session

   amdrocm-rccl<-dev><rocm_version>

Where:

* ``-dev`` adds the library headers in addition to the runtime files. Omit this
  suffix to install only the runtime package.

* ``<rocm_version>`` is the ROCm Core SDK version to install. Omit this suffix to
  install the latest available version.

For example: ``amdrocm-rccl-dev7.13``.

Use the following command to install the latest RCCL development package:

.. tab-set::

   .. tab-item:: Debian-based distros

      .. code-block:: bash

         sudo apt install amdrocm-rccl-dev

   .. tab-item:: RHEL-based distros

      .. code-block:: bash

         sudo dnf install amdrocm-rccl-dev

.. _install-nightly:

Install a nightly build
=======================

`TheRock <https://github.com/ROCm/TheRock>`__ also publishes nightly builds for
the ROCm Core SDK and its components, including RCCL. See `Nightly release status
<https://github.com/ROCm/TheRock#nightly-release-status>`__ for details.
