.. meta::
   :description: How to build and install the hipFile Cython-based Python bindings, including prerequisites, virtual environment setup, and CMake override variables.
   :keywords: hipFile, Python bindings, Cython, install, ROCm, scikit-build-core, wheel, editable install

************************************
Install the hipFile Python bindings
************************************

This page covers building and installing the Cython-based Python bindings for
the hipFile C library. Before following these steps, you must have the hipFile C
library already built or installed. See :doc:`/install/build-from-source` for
instructions on building the C library.

Prerequisites
===============

The Python bindings require the following software:

- Python >= 3.10 with the ``venv`` module
- Cython (installed into your virtual environment)
- scikit-build-core (used as the build backend)
- python-build (the ``build`` front-end, or ``pip`` for editable installs)
- CMake (version 3.21 or later)
- A C compiler (for building the Cython-generated C extension)
- hipFile C library (``libhipfile.so``): built or installed before the Python bindings
- hipFile C development header (``hipfile.h``)
- HIP development headers (``hip/hip_runtime_api.h``)

.. note::

   The Python bindings currently support only the AMD platform.

Set up a virtual environment
==============================

Create and activate a Python virtual environment before installing dependencies
or building the bindings:

.. code:: shell

   python3 -m venv .venv
   source .venv/bin/activate

Install the required Python build tools inside the virtual environment:

.. code:: shell

   pip install cython scikit-build-core build

Build the hipFile C library
============================

The Python bindings link against ``libhipfile.so`` and include ``hipfile.h``. You
must build or install the C library before building the Python package.

If you are building from source, follow the steps in
:doc:`/install/build-from-source`. Note the paths to the build output directory
and the ``include/`` directory: you may need them for the override variables
described below.

Build the Python wheel
========================

Navigate to the ``python/`` directory inside the hipFile source tree and build
the wheel:

.. code:: shell

   cd projects/hipfile/python
   python -m build --wheel

The build system uses CMake (via scikit-build-core) to locate the hipFile
headers, the hipFile shared library, and the HIP runtime headers. By default, it
searches:

- The sibling ``../include`` directory (relative to the ``python/`` directory)
  is searched for ``hipfile.h``, with ``/opt/rocm/include`` as a fallback
  location
- The sibling ``../build/src/amd_detail`` directory is searched for
  ``libhipfile.so``, with ``/opt/rocm/lib`` as a fallback location

If these default paths do not match your environment, use the override variables
described in :ref:`python-cmake-overrides`.

Install an editable package
=============================

For development, you can install the Python package in editable mode. An
editable install lets you modify the Python source code and see changes
immediately without rebuilding or reinstalling the package. Changes to the
Cython source code still require a rebuild.

.. code:: shell

   cd projects/hipfile/python
   pip install -e .

.. _python-cmake-overrides:

CMake override variables
==========================

If the hipFile library, hipFile headers, or HIP headers are installed in
non-default locations, pass override variables to CMake through the build
front-end. Use the ``-Ccmake.define.<KEY>=<VALUE>`` syntax with
``python -m build``, or set them as environment-specific CMake variables with
``pip``.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Variable
     - Description
   * - ``HIPFILE_INCLUDE_DIR``
     - Path to the directory containing ``hipfile.h``. Defaults to ``../include`` (relative to ``python/``) or ``/opt/rocm/include``.
   * - ``HIPFILE_LIBRARY``
     - Path to ``libhipfile.so``. Defaults to ``../build/src/amd_detail`` or ``/opt/rocm/lib``.
   * - ``HIP_INCLUDE_DIR``
     - Path to the directory containing ``hip/hip_runtime_api.h``. Defaults to ``/opt/rocm/include``.

For example, to specify a custom hipFile install location when building the
wheel:

.. code:: shell

   python -m build --wheel \
     -Ccmake.define.HIPFILE_INCLUDE_DIR=/path/to/hipfile/include \
     -Ccmake.define.HIPFILE_LIBRARY=/path/to/lib \
     -Ccmake.define.HIP_INCLUDE_DIR=/opt/rocm/include

Or with ``pip install``:

.. code:: shell

   pip install -e . \
     -Ccmake.define.HIPFILE_INCLUDE_DIR=/path/to/hipfile/include \
     -Ccmake.define.HIPFILE_LIBRARY=/path/to/lib

.. warning::

   If CMake cannot locate ``hipfile.h``, ``libhipfile.so``, or
   ``hip/hip_runtime_api.h``, the build fails with a ``FATAL_ERROR`` message
   indicating which path to set.

Confirm the installation
=============================

After building and installing the wheel, or using an editable install, confirm
that the bindings are importable:

.. code:: shell

   python -c "import hipfile; print('hipFile Python bindings loaded successfully')"
