# AMD System Management Interface (AMD SMI) Library

The AMD System Management Interface Library, or AMD SMI library, is a C library for Linux that provides a user space interface for applications to monitor and control AMD devices.

For additional information refer to [ROCm Documentation](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)

Note: This project is a successor to [rocm_smi_lib](https://github.com/RadeonOpenCompute/rocm_smi_lib)

and [esmi_ib_library](https://github.com/amd/esmi_ib_library)

## Supported platforms

At initial release, the AMD SMI library will support Linux bare metal and Linux virtual machine guest for AMD GPUs. In the future release, the library will be extended to support AMD EPYC™ CPUs.

AMD SMI library can run on AMD ROCm supported platforms, refer to [System requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) for more information.

To run the AMD SMI library, the amdgpu driver and the hsmp driver needs to be installed. Optionally, the libdrm can be
installed to query firmware information and hardware IPs.

## Install CLI Tool and Libraries

**Disclaimer: CLI Tool is provided as an example code to aid the development of telemetry tools and is not guaranteed to be backwards compatible. The Python or C++ Library is recommended as a reliable data source.**  

### Requirements

* python 3.6.8+ 64-bit
  - prerequisite modules:
    - python3-wheel
    - python3-setuptools
* amdgpu driver must be loaded for amdsmi_init() to pass

### Installation

### Install amdgpu using ROCm
* Install amdgpu driver:  
See example below, your release and link may differ. The `amdgpu-install --usecase=rocm` triggers both an amdgpu driver update and AMD SMI packages to be installed on your device.
```shell
sudo apt update
wget https://repo.radeon.com/amdgpu-install/6.0.2/ubuntu/jammy/amdgpu-install_6.0.60002-1_all.deb
sudo apt install ./amdgpu-install_6.0.60002-1_all.deb
sudo amdgpu-install --usecase=rocm
```
* amd-smi --help

### Install Example for Ubuntu 22.04 (without ROCm)

``` bash
apt install amd-smi-lib
# if installed with rocm ignore the export
export PATH="${PATH:+${PATH}:}~/opt/rocm/bin"
amd-smi --help
```

### Optional autocompletion

`amd-smi` cli application supports autocompletion. The package should attempt to install it, if argcomplete is not installed you can enable it by using the following commands:

```bash
python3 -m pip install argcomplete
activate-global-python-argcomplete --user
# restart shell to enable
```

### Manual/Multiple Rocm Instance Python Library Install

In the event there are multiple rocm installations and pyenv is not being used, to use the correct amdsmi version you must uninstall previous versions of amd-smi and install the version you want directly from your rocm instance.

#### Python Library Install Example for Ubuntu 22.04

Remove previous amdsmi installation:

```bash
python3 -m pip list | grep amd
python3 -m pip uninstall amdsmi
```

Then install Python library from your target rocm instance:

``` bash
apt install amd-smi-lib
cd /opt/rocm/share/amd_smi
python3 -m pip install --upgrade pip
python3 -m pip install --user .
```

Now you have the amdsmi python library in your python path:

``` bash
~$ python3
Python 3.8.10 (default, May 26 2023, 14:05:08)
[GCC 9.4.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import amdsmi
>>>
```

### Installing the Python Prerequisite Modules

Python3-setuptools and python3-wheel can both be installed through the pip installer as shown below:

```bash
python3 -m pip install setuptools wheel
```

## Usage Basics for the C Library

### Device/Socket handles

Many of the functions in the library take a "socket handle" or "device handle". The socket is an abstraction of hardware physical socket. This will enable amd-smi to provide a better representation of the hardware to user. Although there is always one distinct GPU for a socket, the APU may have both
GPU device and CPU device on the same socket. Moreover, for MI200, it may have multiple GCDs.

To discover the sockets in the system, `amdsmi_get_socket_handles()` is called to get list of sockets
handles, which in turn can be used to query the devices in that socket using `amdsmi_get_processor_handles()`. The device handler is used to distinguish the detected devices from one another. It is important to note that a device may end up with a different device handles after restart application, so a device handle should not be relied upon to be constant over process.

The list of socket handles discovered using `amdsmi_get_socket_handles()`,can also be used to query the cpus in that socket using `amdsmi_get_processor_handles_by_type()`, which in turn can then be used to query the cores in that cpu using `amdsmi_get_processor_handles_by_type()` again.


## Hello AMD SMI

The only required AMD-SMI call for any program that wants to use AMD-SMI is the `amdsmi_init()` call. This call initializes some internal data structures that will be used by subsequent AMD-SMI calls. In the call, a flag can be passed if the application is only interested in a specific device type.

When AMD-SMI is no longer being used, `amdsmi_shut_down()` should be called. This provides a way to do any releasing of resources that AMD-SMI may have held.

1) A simple "Hello World" type program that displays the temperature of detected devices would look like this:

```c++
#include <iostream>
#include <vector>
#include "amd_smi/amdsmi.h"

int main() {
  amdsmi_status_t ret;

  // Init amdsmi for sockets and devices. Here we are only interested in AMD_GPUS.
  ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);

  // Get all sockets
  uint32_t socket_count = 0;

  // Get the socket count available in the system.
  ret = amdsmi_get_socket_handles(&socket_count, nullptr);

  // Allocate the memory for the sockets
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  // Get the socket handles in the system
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);

  std::cout << "Total Socket: " << socket_count << std::endl;

  // For each socket, get identifier and devices
  for (uint32_t i=0; i < socket_count; i++) {
    // Get Socket info
    char socket_info[128];
    ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
    std::cout << "Socket " << socket_info<< std::endl;

    // Get the device count for the socket.
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i],
              &device_count, &processor_handles[0]);

    // For each device of the socket, get name and temperature.
    for (uint32_t j=0; j < device_count; j++) {
      // Get device type. Since the amdsmi is initialized with
      // AMD_SMI_INIT_AMD_GPUS, the processor_type must be AMDSMI_PROCESSOR_TYPE_AMD_GPU.
      processor_type_t processor_type;
      ret = amdsmi_get_processor_type(processor_handles[j], &processor_type);
      if (processor_type != AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        std::cout << "Expect AMDSMI_PROCESSOR_TYPE_AMD_GPU device type!\n";
        return 1;
      }

      // Get device name
      amdsmi_board_info_t board_info;
      ret = amdsmi_get_gpu_board_info(processor_handles[j], &board_info);
      std::cout << "\tdevice "
                  << j <<"\n\t\tName:" << board_info.product_name << std::endl;

      // Get temperature
      int64_t val_i64 = 0;
      ret =  amdsmi_get_temp_metric(processor_handles[j], AMDSMI_TEMPERATURE_TYPE_EDGE,
              AMDSMI_TEMP_CURRENT, &val_i64);
      std::cout << "\t\tTemperature: " << val_i64 << "C" << std::endl;
    }
  }

  // Clean up resources allocated at amdsmi_init. It will invalidate sockets
  // and devices pointers
  ret = amdsmi_shut_down();

  return 0;
}
```

2) A sample program that displays the power of detected cpus would look like this:

```c++
#include <iostream>
#include <vector>
#include "amd_smi/amdsmi.h"

int main(int argc, char **argv) {
    amdsmi_status_t ret;
	uint32_t socket_count = 0;

    // Initialize amdsmi for AMD CPUs
    ret = amdsmi_init(AMDSMI_INIT_AMD_CPUS);

    ret = amdsmi_get_socket_handles(&socket_count, nullptr);

    // Allocate the memory for the sockets
    std::vector<amdsmi_socket_handle> sockets(socket_count);

    // Get the sockets of the system
    ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);

    std::cout << "Total Socket: " << socket_count << std::endl;

    // For each socket, get cpus
    for (uint32_t i = 0; i < socket_count; i++) {
        uint32_t cpu_count = 0;

        // Set processor type as AMDSMI_PROCESSOR_TYPE_AMD_CPU
        processor_type_t processor_type = AMDSMI_PROCESSOR_TYPE_AMD_CPU;
        ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, nullptr, &cpu_count);

        // Allocate the memory for the cpus
        std::vector<amdsmi_processor_handle> plist(cpu_count);

		 // Get the cpus for each socket
        ret = amdsmi_get_processor_handles_by_type(sockets[i], processor_type, &plist[0], &cpu_count);

        for (uint32_t index = 0; index < plist.size(); index++) {
            uint32_t socket_power;
            std::cout<<"CPU "<<index<<"\t"<< std::endl;
            std::cout<<"Power (Watts): ";

            ret = amdsmi_get_cpu_socket_power(plist[index], &socket_power);
            if(ret != AMDSMI_STATUS_SUCCESS)
                std::cout<<"Failed to get cpu socket power"<<"["<<index<<"] , Err["<<ret<<"] "<< std::endl;

            if (!ret) {
                std::cout<<static_cast<double>(socket_power)/1000<<std::endl;
            }
            std::cout<<std::endl;
        }
    }

    // Clean up resources allocated at amdsmi_init
    ret = amdsmi_shut_down();

    return 0;
}
```

### Documentation

The reference manual, `AMD_SMI_Manual.pdf` will be in the /opt/rocm/share/doc/amd_smi directory upon a successful build.

### Sphinx Documentation

To build the documentation locally, run the commands below:

``` bash
cd docs

python3 -m pip install -r sphinx/requirements.txt

python3 -m sphinx -T -E -b html -d _build/doctrees -D language=en . _build/html
```

The output will be in `docs/_build/html`.

For additional details, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## Building AMD SMI

### Rebuilding Python wrapper

The python wrapper (binding) is an auto-generated file `py-interface/amdsmi_wrapper.py`

Wrapper should be re-generated on each C++ API change, by doing:

```bash
./update_wrapper.sh
```

After this command, the file in `py-interface/amdsmi_wrapper.py` will be automatically updated on each compile.

Note: To be able to re-generate python wrapper you need **docker** installed.

Note: python_wrapper is NOT automatically re-generated. You must run `./update_wrapper.sh`.

### Additional Required software for building

In order to build the AMD SMI library, the following components are required. Note that the software versions listed are what was used in development. Earlier versions are not guaranteed to work:

* CMake (v3.14.0) - `python3 -m pip install cmake`
* g++ (5.4.0)

In order to build the AMD SMI python package, the following components are required:

* python (3.6.8 or above)
* virtualenv - `python3 -m pip install virtualenv`

In order to build the latest documentation, the following are required:

* DOxygen (1.8.11)
* latex (pdfTeX 3.14159265-2.6-1.40.16)

The source code for AMD SMI is available on Github.

After the AMD SMI library git repository has been cloned to a local Linux machine, the Default location for the library and headers is /opt/rocm. Before installation, the old rocm directories should be deleted:
/opt/rocm
/opt/rocm-{number}

Building the library is achieved by following the typical CMake build sequence (run as root user or use 'sudo' before 'make install' command), specifically:

```bash
mkdir -p build
cd build
cmake ..
make -j $(nproc)
make install
```

The built library will appear in the `build` folder.

To build the rpm and deb packages follow the above steps with:

```bash
make package
```

### Building the Tests

In order to verify the build and capability of AMD SMI on your system and to see an example of how AMD SMI can be used, you may build and run the tests that are available in the repo. To build the tests, follow these steps:

```bash
mkdir -p build
cd build
cmake -DBUILD_TESTS=ON ..
make -j $(nproc)
```

### Run the Tests

To run the test, execute the program `amdsmitst` that is built from the steps above.
Path to the program `amdsmitst`: build/tests/amd_smi_test/

## DISCLAIMER

The information contained herein is for informational purposes only, and is subject to change without notice. In addition, any stated support is planned and is also subject to change. While every precaution has been taken in the preparation of this document, it may contain technical inaccuracies, omissions and typographical errors, and AMD is under no obligation to update or otherwise correct this information. Advanced Micro Devices, Inc. makes no representations or warranties with respect to the accuracy or completeness of the contents of this document, and assumes no liability of any kind, including the implied warranties of noninfringement, merchantability or fitness for particular purposes, with respect to the operation or use of AMD hardware, software or other products described herein.

© 2023-2024 Advanced Micro Devices, Inc. All Rights Reserved.
