.. meta::
  :description: AQLProfile installation process
  :keywords: AQLProfile, ROCm, install

******************
Install AQLProfile
******************

Learn how to build AQLProfile with a script or with CMake, then install the library with a command.

Prerequisites
=============

Before you begin, ensure these tools and dependencies are installed:

* ROCm stack
* ``rocm-llvm-dev`` (required to build tests)


Build AQLProfile
================

You can build AQLProfile using either the provided build script (recommended for most users) or by manually invoking CMake for custom builds.


Option 1: Use the build script (Recommended)
--------------------------------------------

This configures and builds the project with the default settings: 

.. code:: bash 
  
  ./build.sh


Option 2: Use CMake for custom builds
-------------------------------------

For more control over the build process, you can set the CMake options manually:

.. code:: bash

      # Set the CMAKE_PREFIX_PATH to point to hsa-runtime includes path and hsa-runtime library path
      export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
      # For example, if ROCm is installed at /opt/rocm:
      # export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

      export CMAKE_BUILD_TYPE=<debug|release> # release by default

      cd /path/to/aqlprofile
      mkdir build
      cd build
      cmake ..
      make -j


(Optional) Enable debug tracing
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To enable debug tracing, set this environment variable before running CMake: 

.. code:: bash
  
  export CMAKE_DEBUG_TRACE=1

This enables verbose debug output of the command packets while this library executes.


Install the AQLProfile libraries
================================

Once your build is successful, install the AQLProfile libraries with:

.. code:: bash

    cd build
    sudo make install
