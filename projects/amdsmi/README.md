# AMD System Management Interface (AMD SMI) library

The AMD System Management Interface (AMD SMI) library offers a unified tool for managing and monitoring GPUs,
particularly in high-performance computing environments. It provides a user-space interface that allows applications to
control GPU operations, monitor performance, and retrieve information about the system's drivers and GPUs.

For information on available features, installation steps, API reference material, and helpful tips, refer to the online
documentation at [rocm.docs.amd.com/projects/amdsmi](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)

>[!NOTE]
>This project is a successor to [rocm_smi_lib](https://github.com/ROCm/rocm_smi_lib)
>and [esmi_ib_library](https://github.com/amd/esmi_ib_library).  
>This project is applicable to Linux Baremetal and Linux VM(Guest). To use AMD SMI for Virtualization, please refer to [AMD-SMI Virtualization](https://github.com/amd/MxGPU-Virtualization/tree/mainline/smi-lib).

## Supported platforms

The AMD SMI library supports Linux bare metal and Linux virtual machine guest
for AMD GPUs and AMD EPYC™ CPUs via
[esmi_ib_library](https://github.com/amd/esmi_ib_library).

AMD SMI library can run on AMD ROCm supported platforms, refer to
[System requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
for more information.

## Installation

* [Install the AMD SMI library and CLI tool](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)

## Requirements

The following are required to install and use the AMD SMI library through its language interfaces and CLI.

* `amdgpu` driver must be loaded for [`amdsmi_init()`](./docs/how-to/amdsmi-cpp-lib#hello-amd-smi) to work.
* Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

  ```bash
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64
  ```

### Python interface and CLI tool prerequisites

* Python 3.6.8+ (64-bit)

### Note: No module named more_itertools warning on Azure Linux 3
During the driver installation process on Azure Linux 3, you might encounter the `ModuleNotFoundError: No module named 'more_itertools'` warning. This warning is a result of the reintroduction of `python3-wheel` and `python3-setuptools` dependencies in the CMake of AMD SMI, which requires `more_itertools` to build these Python libraries. This issue will be fixed in a future ROCm release. As a workaround, use the following command before installation:
```
sudo python3 -m pip install more_itertools 
```

### Go API prerequisites

* Go version 1.20 or greater

## Install amdgpu driver and AMD SMI with ROCm

1. Get the `amdgpu-install` installer following the instructions for your Linux distribution at
   [Installation via AMDGPU installer](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/amdgpu-install.html#installation).

2. Use `amdgpu-install` to install the `amdgpu` driver and ROCm packages with
   AMD SMI included.

   ``` shell
   sudo amdgpu-install --usecase=rocm
   ```

   The `amdgpu-install --usecase=rocm` option triggers both an `amdgpu` driver
   update and AMD SMI packages to be installed on your device.

3. Verify your installation.

   ```shell
   amd-smi --help
   ```

## Install AMD SMI without ROCm

The following are example steps to install the AMD SMI libraries and CLI tool on
Ubuntu 22.04.

1. Install the library.

   ```shell
   sudo apt install amd-smi-lib
   ```

2. Add the installation directory to your PATH. If installed with ROCm, ignore
   this step.

   ```shell
   export PATH="${PATH:+${PATH}:}~/opt/rocm/bin"
   ```

3. Verify your installation.

   ```shell
   amd-smi --help
   ```

## AMD SMI basic usage

### C++ library

For developers focused on performance monitoring, system diagnostics, or resource management, the AMD SMI C++ library
offers a powerful and versatile tool to unlock the full capabilities of AMD hardware.

Refer to the [user guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cpp-lib.html) and the
detailed [C++ API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-cpp-api.html) in the
ROCm documentation portal.

### Python library

The AMD SMI Python interface provides an easy-to-use
[API](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-lib.html) for interacting with AMD
hardware. It simplifies tasks like monitoring and controlling GPU operations, allowing for rapid development.

Refer to the [user guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-py-lib.html) and the
detailed [Python API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-py-api.html) in the
ROCm documentation portal.

### Go library

The AMD SMI Go interface provides a simple
[API](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-go-lib.html)
for AMD hardware management. It streamlines hardware monitoring and control
while leveraging Golang's features.

Refer to the [user guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-go-lib.html) and the
[Go API reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/reference/amdsmi-go-api.html) in the
ROCm documentation portal.

### CLI tool

A versatile command line tool for managing and monitoring AMD hardware. You can use `amd-smi` for:

- Device information: Quickly retrieve detailed information about AMD GPUs

- Performance monitoring: Real-time monitoring of GPU utilization, memory, temperature, and power consumption

- Process information: Identify which processes are using GPUs

- Configuration management: Adjust GPU settings like clock speeds and power limits

- Error reporting: Monitor and report GPU errors for proactive maintenance

Check out
[Getting to Know Your GPU: A Deep Dive into AMD SMI -- ROCm Blogs](https://rocm.blogs.amd.com/software-tools-optimization/amd-smi-overview/README.html)
for a rundown.

### Docker container configuration

To ensure proper functionality of AMD SMI within a Docker container, the
following configuration options must be included. These settings are
particularly important for managing memory partitions, as partitioning depends
on loading and unloading kernel drivers.

- `--cap-add=SYS_MODULE`

- `-v /lib/modules:/lib/modules`

See [Using AMD SMI in a Docker
container](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/setup-docker-container.html)
for more information.

## Building AMD SMI

This section describes the prerequisites and steps to build AMD SMI from source.

### Required software

To build the AMD SMI library, the following components are required. Note that the software versions specified were used
during development; earlier versions are not guaranteed to work.

* CMake (v3.20.0 or later) -- `python3 -m pip install cmake`
* g++ (v5.4.0 or later)
* libdrm-dev (for Ubuntu and Debian)
* libdrm-devel (for RPM-based distributions)

In order to build the AMD SMI Python package, the following components are required:

* Python (3.6.8 or later)
* virtualenv -- `python3 -m pip install virtualenv`

### Build steps

1. Clone the AMD SMI repository to your local Linux machine.

   ```shell
   git clone https://github.com/ROCm/amdsmi.git
   ```

2. The default installation location for the library and headers is `/opt/rocm`. Before installation, any old ROCm
   directories should be deleted:

   * `/opt/rocm`
   * `/opt/rocm-<version_number>`

3. Build the library by following the typical CMake build sequence (run as root user or use `sudo` before `make install`
   command); for instance:

   ```bash
   mkdir -p build
   cd build
   cmake ..
   make -j $(nproc)
   make install
   ```

   The built library is located in the  `build/` directory. To build the `rpm` and `deb` packages use the following
   command:

   ```bash
   make package
   ```

### Rebuild the Python wrapper

The Python wrapper for the AMD SMI library is found in the [auto-generated file](#py_lib_fs)
`py-interface/amdsmi_wrapper.py`. It is essential to regenerate this wrapper whenever there are changes to the C++ API.
It is not regenerated automatically.

To regenerate the wrapper, use the following command.

```shell
./update_wrapper.sh
```

After this command, the file in `py-interface/amdsmi_wrapper.py` will be updated
on compile.

>[!NOTE]
>You need Docker installed on your system to regenerate the Python wrapper.

### Build the tests

To verify the build and capabilities of AMD SMI on your system, as well as to see practical examples of its usage, you
can build and run the available [tests in the repository](https://github.com/ROCm/amdsmi/tree/amd-staging/tests). Follow
these steps to build the tests:

```bash
mkdir -p build
cd build
cmake -DBUILD_TESTS=ON ..
make -j $(nproc)
```

#### Run the tests

Once the tests are [built](#build-the-tests), you can run them by executing the `amdsmitst` program. The executable can
be found at `build/tests/amd_smi_test/`.

### Build the docs

To build the documentation, follow the instructions at
[Building documentation](https://rocm.docs.amd.com/en/latest/contribute/building.html).

## DISCLAIMER

The information contained herein is for informational purposes only, and is subject to change without notice. In
addition, any stated support is planned and is also subject to change. While every precaution has been taken in the
preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is
under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no
representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes
no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular
purposes, with respect to the operation or use of AMD hardware, software or other products described herein.

© 2023-2025 Advanced Micro Devices, Inc. All Rights Reserved.
