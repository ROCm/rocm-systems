## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2023 Advanced Micro Devices, Inc. All Rights Reserved.


## ROCProfiler API Concepts
- Session
- Filter
- Buffer


## API Philosophy

The APIs provide a common interface to the users for different
features such as profiling, tracing.

In order to make use of any functionality of rocprofv2, one needs to create
a "Session" object. This session could be a profiling session/tracing/pc-sampling session etc.

In order to set user inputs, one needs to provide a "Filter" to a session object.
This filter could be for counters/traces/pc-samples etc.

Now that the input is taken care of, one also needs to provide a "Buffer" which
will store the output results generated during a session. This buffer will contain
different records corresponding to the filter type chosen. A flush function can also
be specified for the buffer, which will be used to flush the buffer records.
A filter and buffer are associated together.

Once a Session, Buffer, Filter have all been created, the session can be started.
One can control when the session can be started, stopped and destroyed.

## Descriptions of Code Samples
### kernel_profiling_sample.cpp
This code sample demonstrates how to use the APIs to collect performance counters and metrics for every kernel dispatch.

### tracer_sample.cpp
This code sample demonstrates how to use the APIs to collect different API and activity traces:
- HIP API
- HIP OPS
- HSA API
- HSA OPS
- ROCTX

### device_profiling_sample.cpp
This code sample demonstrates how to use the APIs to collect counters and metrics from the GPU via user defined sampling, instead of per-kernel dipatch measurements.


## How to compile
In order to get the samples to compile, make sure to copy rocprofiler binaries into /opt/rocm/lib
Running 'make install' inside the rocprofiler/build folder will copy the binaries to /opt/rocm/lib

Alternately, change the 'ROCPROFILER_LIBS_PATH' variable in the Makefile to point to the rocprofiler/build folder.
After modifications to Makefile are done, run:

  ```bash
  # compile all samples
  make
  ```

  ```bash
  # compile kernel_profiling_no_replay_sample.cpp
  make kernel_profiling_no_replay_sample
  ```

### How to run
Before running, ROCPROFILER_METRICS_PATH needs to be set to point to 'derived_counters.xml'
If the rocprofiler binaries are present in the rocm installation path /opt/rocm
then below command will work:
```bash
export ROCPROFILER_METRICS_PATH=/opt/rocm/libexec/rocprofiler/counters/derived_counters.xml
```

Otherwise, make it point to rocprofiler/build/libexec/rocprofiler/counters/derived_counters.xml like below:
```bash
export ROCPROFILER_METRICS_PATH=<path_to_rocprofiler>/rocprofiler/build/libexec/rocprofiler/counters/derived_counters.xml
```

Finally, run a sample:
```bash
./kernel_profiling_no_replay_sample
```

## PC-Sampler
The ROCProfiler library includes an API to enable periodic sampling of the GPU
program counter during kernel execution. An example program is included that demonstrates the PC
sampling API, with additional code to illustrate a typical non-trivial use case:
correlation of sampled PC addresses with their disassembled machine code, as
well as source code and symbolic debugging information if available.

See [PC-Sampler README](pcsampler/code_printing_sample/README.md)