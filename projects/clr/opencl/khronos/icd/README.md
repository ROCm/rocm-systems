# OpenCL ICD Loader

This repo contains the source code and tests for the Khronos official OpenCL ICD Loader.

## CI Build Status

[![Windows Build Status](https://ci.appveyor.com/api/projects/status/47uhjgp5h4de2f63/branch/master?svg=true)](https://ci.appveyor.com/project/Khronoswebmaster/opencl-icd-loader/branch/master) [![Linux OSX Build Status](https://travis-ci.com/KhronosGroup/opencl-icd-loader.svg?branch=master)](https://travis-ci.com/KhronosGroup/opencl-icd-loader)

## Introduction

OpenCL defines an *Installable Client Driver* (ICD) mechanism to allow developers to build applications against an *Installable Client Driver* loader (ICD loader) rather than linking their applications against a specific OpenCL implementation.
The ICD Loader is responsible for:

* Exporting OpenCL API entry points
* Enumerating OpenCL implementations
* Forwarding OpenCL API calls to the correct implementation

This repo contains the source code and tests for the Khronos official OpenCL ICD Loader.

Note that this repo does not contain an OpenCL implementation (ICD).
You will need to obtain and install an OpenCL implementation for your OpenCL device that supports the OpenCL ICD extension `cl_khr_icd` to run an application using the OpenCL ICD Loader.

The OpenCL *Installable Client Driver* extension (`cl_khr_icd`) is described in the OpenCL extensions specification, which may be found on the [Khronos OpenCL Registry](https://www.khronos.org/registry/OpenCL/).

## Build Instructions

### Dependencies

The OpenCL ICD Loader requires OpenCL Headers.
To use system OpenCL Headers, please specify the OpenCL Header location using the CMake variable `OPENCL_ICD_LOADER_HEADERS_DIR`.
By default, the OpenCL ICD Loader will look for OpenCL Headers in the `inc` directory.

By default, the OpenCL ICD Loader on Windows requires the Windows Driver Kit (WDK).
To build OpenCL ICD Loader with WDK support -
* Install recent Windows WDK currently at https://docs.microsoft.com/en-us/windows-hardware/drivers/download-the-wdk

* Establish environment variable WDK to include directory. Ex: set WDK=C:\Program Files (x86)\Windows Kits\10\include\10.0.17763.0

An OpenCL ICD Loader may be built without the Windows Driver Kit using the CMake variable `OPENCL_ICD_LOADER_REQUIRE_WDK`, however this option should be used with caution since it may prevent the OpenCL ICD Loader from enumerating some OpenCL implementations.
This dependency may be removed in the future.

The OpenCL ICD Loader uses CMake for its build system.
If CMake is not provided by your build system or OS package manager, please consult the [CMake website](https://cmake.org).

### Build and Install Directories

A common convention is to place the `build` directory in the top directory of the repository and to place the `install` directory as a child of the `build` directory.
The remainder of these instructions follow this convention, although you may place these directories in any location.

### Example Usage

For most Windows and Linux usages, the following steps are sufficient to build the OpenCL ICD Loader:

1. Clone this repo:

        git clone https://github.com/KhronosGroup/OpenCL-ICD-Loader

1. Obtain the OpenCL Headers, if you are not planning to use system OpenCL headers.
Headers may be obtained from the [Khronos OpenCL Headers](https://github.com/KhronosGroup/OpenCL-Headers) repository.

1. Create a `build` directory:

        cd OpenCL-ICD-Loader
        mkdir build
        cd build

1. Invoke `cmake` to generate solution files, Makefiles, or files for other build systems.

        cmake ..

1. Build using the CMake-generated files.

Notes:

* For 64-bit Windows builds, you may need to specify a 64-bit generator manually, for example:

        cmake.exe -G "Visual Studio 14 2015 Win64" ..

* Some users may prefer to use a CMake GUI frontend, such as `cmake-gui` or `ccmake`, vs. the command-line CMake.

## OpenCL ICD Loader Tests

OpenCL ICD Loader Tests can be run using `ctest`, which is a companion to CMake.
The OpenCL ICD Loader Tests can also be run directly by executing icd_loader_test(.exe) executable from the bin folder.

### Test Setup

The OpenCL ICD Loader Tests use a "stub" ICD, which must be set up manually.
The OpenCL ICD Loader Tests will "fail" if the "stub" ICD is not set up correctly.
The method to install the "stub" ICD is operating system dependent.

On Linux, install the "stub" ICD by creating a file with the full path to the "stub" ICD in `/etc/OpenCL/vendors`:

    echo full/path/to/libOpenCLDriverStub.so > /etc/OpenCL/vendors/test.icd

On Windows, add the "stub" ICD by adding a `REG_DWORD` value to the registry keys:

    // For 32-bit operating systems, or 64-bit tests on a 64-bit operating system:
    HKEY_LOCAL_MACHINE\SOFTWARE\Khronos\OpenCL\Vendors

    // For 32-bit tests on a 64-bit operating system:
    HKEY_LOCAL_MACHINE\SOFTWARE\Wow6432Node\Khronos\OpenCL\Vendors

    // The name of the REG_DWORD value should be the full path to the "stub" ICD
    // OpenCLDriverStub.dll, and the data for this value should be 0.

### Running Tests

To run the tests, invoke `ctest` from the `build` directory.
The CMake-generated build files may be able to invoke the OpenCL ICD Loader tests as well.

### Test Cleanup

Manually remove the file or registry keys added during Test Setup.

## Support

Please create a GitHub issue to report an issue or ask questions.

## Contributing

Contributions to the OpenCL ICD Loader are welcomed and encouraged.
You will be prompted with a one-time "click-through" CLA dialog as part of submitting your pull request or other contribution to GitHub.

## Table of Debug Environment Variables

The following debug environment variables are available for use with the OpenCL ICD loader:

| Environment Variable              | Behavior            |  Example Format      |
|:---------------------------------:|---------------------|----------------------|
| OCL_ICD_FILENAMES                 | Specifies a list of additional ICDs to load.  The ICDs will be enumerated first, before any ICDs discovered via default mechanisms. | `export OCL_ICD_FILENAMES=libVendorA.so:libVendorB.so`<br/><br/>`set OCL_ICD_FILENAMES=vendor_a.dll;vendor_b.dll` |
| OCL_ICD_VENDORS                   | On Linux and Android, specifies a directory to scan for ICDs to enumerate in place of the default `/etc/OpenCL/vendors'. |  `export OCL_ICD_VENDORS=/my/local/icd/search/path` |
