.. meta::
   :description: Tutorial showing how to query the hipFile library version at compile time and runtime using version macros and the hipFileGetVersion() function.
   :keywords: hipFile, version, HIPFILE_VERSION_MAJOR, HIPFILE_VERSION_MINOR, HIPFILE_VERSION_PATCH, hipFileGetVersion, ROCm, tutorial

**************************
Query the hipFile version
**************************

This tutorial demonstrates how to retrieve the hipFile library version using compile-time macros and the runtime ``hipFileGetVersion()`` function. Knowing the library version is useful when you need to conditionally compile code against different hipFile releases. It also lets you confirm that the installed runtime library matches expectations.

Prerequisites
================

Before working through this example, verify you have:

- hipFile installed (see :doc:`/install/install`)
- A C++ compiler and CMake
- The HIP runtime available on your system

When to use each approach
===========================

hipFile provides two ways to query its version:

Compile-time macros
   ``HIPFILE_VERSION_MAJOR``, ``HIPFILE_VERSION_MINOR``, and ``HIPFILE_VERSION_PATCH`` are preprocessor defines set in the hipFile header. Use these when you need to enable or disable code paths with ``#if`` or ``#ifdef`` directives based on the header version available at build time.

Runtime function
   ``hipFileGetVersion()`` returns the version of the hipFile shared library that is loaded at runtime. Use this when you need to confirm the version of the library that is actually linked, which might differ from the header version if the library was upgraded independently.

Complete example
=================

The following program prints the hipFile version using both methods. This code is adapted from ``examples/api/get-version.cpp`` in the hipFile source tree.

.. code-block:: cpp

   #include <hipfile.h>

   #include <cstdio>
   #include <cstdlib>

   int
   main(void)
   {
       unsigned       major = 0;
       unsigned       minor = 0;
       unsigned       patch = 0;
       hipFileError_t err;

       printf("\n");

       /* Compile-time version from header macros */
       printf("Version from the header symbols (major.minor.patch): %d.%d.%d\n",
              HIPFILE_VERSION_MAJOR,
              HIPFILE_VERSION_MINOR,
              HIPFILE_VERSION_PATCH);
       printf("\n");

       /* Runtime version from the shared library */
       err = hipFileGetVersion(&major, &minor, &patch);
       if (err.err != hipFileSuccess || err.hip_drv_err != hipSuccess)
           return EXIT_FAILURE;
       printf("Version from hipFileGetVersion() (major.minor.patch): %u.%u.%u\n",
              major, minor, patch);
       printf("\n");

       return EXIT_SUCCESS;
   }

Step-by-step walkthrough
========================

Include the header
------------------

.. code-block:: cpp

   #include <hipfile.h>

The ``hipfile.h`` header declares the version macros, ``hipFileGetVersion()``, error types, and all other hipFile API functions. Including this single header gives you access to everything in the hipFile C API.

Print the compile-time version
------------------------------

.. code-block:: cpp

   printf("Version from the header symbols (major.minor.patch): %d.%d.%d\n",
          HIPFILE_VERSION_MAJOR,
          HIPFILE_VERSION_MINOR,
          HIPFILE_VERSION_PATCH);

``HIPFILE_VERSION_MAJOR``, ``HIPFILE_VERSION_MINOR``, and ``HIPFILE_VERSION_PATCH`` are integer-valued preprocessor macros. Because they are resolved at compile time, you can also use them in preprocessor conditionals:

.. code-block:: cpp

   #if HIPFILE_VERSION_MAJOR >= 1
   // Use a feature introduced in version 1.0.0
   #endif

Query the runtime version
-------------------------

.. code-block:: cpp

   hipFileError_t err;
   unsigned major = 0, minor = 0, patch = 0;

   err = hipFileGetVersion(&major, &minor, &patch);
   if (err.err != hipFileSuccess || err.hip_drv_err != hipSuccess)
       return EXIT_FAILURE;

``hipFileGetVersion()`` writes the major, minor, and patch components of the running library's version into the provided output parameters. It returns a ``hipFileError_t`` struct containing two fields:

- ``err``: a ``hipFileOpError_t`` value indicating a hipFile-specific error.
- ``hip_drv_err``: a ``hipError_t`` value indicating a GPU driver error.

Always check both fields before using the output values.

Build the example
===================

Use CMake to build the example. If ROCm or hipFile are installed in non-standard locations, pass them through ``CMAKE_PREFIX_PATH``:

.. code:: shell

   cmake -DCMAKE_PREFIX_PATH="/path/to/rocm;/path/to/hipFile" /path/to/examples/api
   cmake --build .

Expected output
================

When you run the built ``get-version`` binary, you see output similar to the following:

.. code-block:: text

   Version from the header symbols (major.minor.patch): 0.2.0

   Version from hipFileGetVersion() (major.minor.patch): 0.2.0

The two version strings are identical when the header and the shared library come from the same hipFile release. If you see different values, the installed header and shared library are from different builds.
