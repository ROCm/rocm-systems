.. meta::
  :description: Installing rocDecode with the package installer
  :keywords: install, rocDecode, AMD, ROCm, basic, development, package

********************************************************************
Installing rocDecode with the package installer
********************************************************************

Three rocDecode packages are available:

* ``rocdecode``: The rocDecode runtime package. This is the basic rocDecode package. It must always be installed.
* ``rocdecode-host``: The rocDecode software decoding package. This package must be installed to use the FFmpeg software decoder.
* ``rocdecode-dev``: The rocDecode development package. This package installs a full suite of libraries, header files, and samples. This package must be installed to use the rocDecode samples.
* ``rocdecode-test``: A test package that provides a CTest to verify the installation. This package must be installed to use the rocDecode samples.

All the required prerequisites are installed when the package installation method is used.


Basic installation without software decoding
==============================================

Use the following commands to install only the rocDecode runtime package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocdecode

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocdecode

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocdecode


Basic installation with software decoding
==============================================

Use the following commands to install the rocDecode runtime package and the host decoding package:

.. tab-set::
 
  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocdecode rocdecode-host

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocdecode rocdecode-host

  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocdecode rocdecode-host


Complete installation
========================================

Use the following commands to install ``rocdecode``, ``rocdecode-host``, ``rocdecode-dev``, and ``rocdecode-test``:

.. tab-set::

  .. tab-item:: Ubuntu

    .. code:: shell

      sudo apt install rocdecode rocdecode-host rocdecode-dev rocdecode-test 

  .. tab-item:: RHEL

    .. code:: shell

      sudo yum install rocdecode rocdecode-host rocdecode-devel rocdecode-test
  .. tab-item:: SLES

    .. code:: shell

      sudo zypper install rocdecode rocdecode-host rocdecode-devel rocdecode-test
