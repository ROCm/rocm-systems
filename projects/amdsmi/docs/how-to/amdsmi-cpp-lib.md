---
myst:
  html_meta:
    "description lang=en": "Get started with the AMD SMI C++ library. Basic usage and examples."
    "keywords": "api, smi, lib, c++, system, management, interface, ROCm"
---

# AMD SMI C++ library usage and examples

This section presents a brief overview and some basic examples on the AMD SMI
library's usage. Whether you are developing applications for performance
monitoring, system diagnostics, or resource allocation, the AMD SMI C++ library
serves as a valuable tool for leveraging the full potential of AMD hardware in
your projects.

```{note}
``hipcc`` and other compilers will not automatically link in the ``libamd_smi``
dynamic library. To compile code that uses the AMD SMI library API, ensure the
``libamd_smi.so`` can be located by setting the ``LD_LIBRARY_PATH`` environment
variable to the directory containing ``librocm_smi64.so`` (usually
``/opt/rocm/lib``) or by passing the ``-lamd_smi`` flag to the compiler.
```

```{note}
The environment variable ``AMDSMI_GPU_METRICS_CACHE_MS`` may be set to
control the internal GPU metrics cache duration (ms). 
Default 1, set to 0 to disable.
```

```{seealso}
Refer to the [C++ library API reference](../reference/amdsmi-cpp-api.md).
```

(device_socket_handle)=
## Device and socket handles

Many functions in the library take a _socket handle_ or _device handle_. A
_socket_ refers to a physical hardware socket, abstracted by the library to
represent the hardware more effectively to the user. While there is always one
unique GPU per socket, an APU may house both a GPU and CPU on the same socket.
For MI200 GPUs, multiple GCDs may reside within a single socket

To identify the sockets in a system, use the `amdsmi_get_socket_handles()`
function, which returns a list of socket handles. These handles can then be used
with `amdsmi_get_processor_handles()` to query devices within each socket. The
device handle is used to differentiate between detected devices; however, it's
important to note that a device handle may change after restarting the
application, so it should not be considered a persistent identifier across
processes.

The list of socket handles obtained from `amdsmi_get_socket_handles()` can
also be used to query the CPUs in each socket by calling
`amdsmi_get_processor_handles_by_type()`. This function can then be called again
to query the cores within each CPU.

(cpp_hello_amdsmi)=
## Hello AMD SMI

An application using AMD SMI must call `amdsmi_init()` to initialize the AMI SMI
library before all other calls. This call initializes the internal data
structures required for subsequent AMD SMI operations. In the call, a flag can
be passed to indicate if the application is interested in a specific device
type.

`amdsmi_shut_down()` must be the last call to properly close connection to
driver and make sure that any resources held by AMD SMI are released.

1. A simple "Hello World" type program that displays the temperature of detected
   devices.

   ```{note}
   Sample build example:
   $ g++ -I/opt/rocm/include <file_name>.cc -L/opt/rocm/lib -lamd_smi -o <filename>

   Users /opt/rocm-*/bin path may differ (depending on install), please locate the path of your libamd_smi.so.*.
   For example:

   $ sudo find /opt/ -iname libamd_smi.so*
   /opt/rocm-6.4.1/lib/libamd_smi.so.25.0
   /opt/rocm-6.4.1/lib/libamd_smi.so
   ```

   The code is as follows:

   ```cpp
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

2. A sample program that displays the power of detected CPUs.

   ```{note}
   Sample build example:
   $ g++ -DENABLE_ESMI -I/opt/rocm/include <file_name>.cc -L/opt/rocm/lib -lamd_smi -o <filename>

   For finding available rocm include and library path, see building example on sample program 1 above.
   ```

   The code is as follows:

   ```cpp
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
