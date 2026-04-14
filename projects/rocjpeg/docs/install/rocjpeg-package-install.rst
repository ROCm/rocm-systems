.. meta::
  :description: Installing rocJPEG with the package installer
  :keywords: install, rocJPEG, AMD, ROCm, basic, development, package

********************************************************************
Installing rocJPEG with the package installer
********************************************************************

.. note::

  ROCm must be installed before installing rocJPEG. See `Quick start installation guide <https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/quick-start.html>`_ for detailed ROCm installation instructions. 

There are three rocJPEG packages available:

* ``rocjpeg``: The rocJPEG runtime package. This is the basic rocJPEG package. 
* ``rocjpeg-dev``: The rocJPEG development package. This package installs a full suite of libraries, header files, and samples for contributing to the rocJPEG code base.
* ``rocjpeg-test``: A test package that provides a CTest to verify the installation.

Developers who want to contribute to the rocJPEG code base must install both ``rocjpeg-dev`` and ``rocjpeg-test`` in addition to ``rocjpeg``.

All the required prerequisites are installed when the package installation method is used.

Basic installation
========================================

Use the following commands to install only the rocJPEG runtime package:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocjpeg

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocjpeg

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocjpeg


Developer installation
========================================

All three rocJPEG packages, ``rocjpeg``, ``rocjpeg-dev``, and ``rocjpeg-test`` must be installed to develop for rocJPEG. 

Use the following commands to install ``rocjpeg``, ``rocjpeg-dev``, and ``rocjpeg-test``:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocjpeg rocjpeg-dev rocjpeg-test

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocjpeg rocjpeg-devel rocjpeg-test

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocjpeg rocjpeg-devel rocjpeg-test