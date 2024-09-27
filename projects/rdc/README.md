# ROCm<sup>TM</sup> Data Center Tool (RDC)

The ROCm™ Data Center Tool simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

- GPU telemetry
- GPU statistics for jobs
- Integration with third-party tools
- Open source

For up-to-date document and how to start using RDC from pre-built packages, please refer to the [**ROCm DataCenter Tool User Guide**](https://rocm.docs.amd.com/projects/rdc/en/latest/)

## Certificate generation

For certificate generation, please refer to
[**RDC Developer Handbook**#generate-files-for-authentication](https://rocm.docs.amd.com/projects/rdc/en/latest/install/handbook.html#generate-files-for-authentication)  
Or read the concise guide under authentication/readme.txt

## Supported platforms

RDC can run on AMD ROCm supported platforms, please refer to the [List of Supported Operating Systems](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html#supported-operating-systems)

## Important notes

### RocProfiler metrics usage

When using rocprofiler fields (800-899) you must call
`export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so.1`
before starting a compute load.

[***See: dmon-rocprofiler-fields-return-zeros***](#dmon-rocprofiler-fields-return-zeros)

## Building RDC from source

### Dependencies

    CMake 3.15                              ## 3.15 or greater is required for gRPC
    g++ (5.4.0)
    Doxygen (1.8.11)                        ## required to build the latest documentation
    Latex (pdfTeX 3.14159265-2.6-1.40.16)   ## required to build the latest documentation
    gRPC and protoc                         ## required for communication
    libcap-dev                              ## required to manage the privileges.

    AMD ROCm platform (https://github.com/ROCm/ROCm)
        * It is recommended to install the complete AMD ROCm platform.
          For installation instruction see https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html
        * At the minimum, these two components are required
            (i)  AMDSMI Library (https://github.com/ROCm/amdsmi)
            (ii) AMD ROCk Kernel driver (https://github.com/ROCm/ROCK-Kernel-Driver)

## Building gRPC and protoc

**NOTE:** gRPC and protoc compiler must be built when building RDC from source as pre-built packages are not available. When installing RDC from a package, gRPC and protoc will be installed from the package.

**IMPORTANT:** Building gRPC and protocol buffers requires CMake 3.15 or greater. With an older version build will quietly succeed with a *message*. However, all components of gRPC will not be installed and RDC will ***fail*** to run

The following tools are required for gRPC build & installation

    automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl

### Download and build gRPC

By default (without using CMAKE_INSTALL_PREFIX option), gRPC will install to `/usr/local` lib, include and bin directories.  
It is highly recommended to install gRPC into a unique directory.
Below example installs gRPC into `/opt/grpc`

    git clone -b v1.61.0 https://github.com/grpc/grpc --depth=1 --shallow-submodules --recurse-submodules
    cd grpc
    export GRPC_ROOT=/opt/grpc
    cmake -B build \
        -DgRPC_INSTALL=ON \
        -DgRPC_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=ON \
        -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT" \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DCMAKE_BUILD_TYPE=Release
    make -C build -j $(nproc)
    sudo make -C build install
    echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf

## Building RDC

Clone the RDC source code from GitHub and use CMake to build and install

    git clone https://github.com/ROCm/rdc
    cd rdc
    # default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
    cmake -B build -DGRPC_ROOT="$GRPC_ROOT"
    # enable rocprofiler (optional)
    cmake -B build -DBUILD_PROFILER=ON
    # enable RVS (optional)
    cmake -B build -DBUILD_RVS=ON
    # build and install
    make -C build -j $(nproc)
    make -C build install

## Building RDC library only without gRPC (optional)

If only the RDC libraries are needed (i.e. only "embedded mode" is required), the user can choose to not build rdci and rdcd. This will eliminate the need for gRPC and protoc. To build in this way, -DBUILD_STANDALONE=off should be passed on the the cmake command line:

    cmake -B build -DBUILD_STANDALONE=off

## Building RDC library without ROCM Run time (optional)

The user can choose to not build RDC diagnostic ROCM Run time. This will eliminate the need for ROCM Run time. To build in this way, -DBUILD_RUNTIME=off should be passed on the the cmake command line:

    cmake -B build -DBUILD_RUNTIME=off

## Update System Library Path

    RDC_LIB_DIR=/opt/rocm/lib/rdc
    GRPC_LIB_DIR="${RDC_LIB_DIR}/grpc/lib\n/opt/grpc/lib"
    echo -e "${RDC_LIB_DIR}" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
    echo -e "${GRPC_LIB_DIR}" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
    ldconfig

## Running RDC

RDC supports encrypted communications between clients and servers. The
communication can be configured to be *authenticated* or *not authenticated*. The [**user guide**](https://rocm.docs.amd.com/projects/rdc/en/latest/) has information on how to generate and install SSL keys and certificates for authentication. By default, authentication is enabled.

## Starting ROCm™ Data Center Daemon (RDCD)

For an RDC client application to monitor and/or control a remote system, the RDC server daemon, *rdcd*, must be running on the remote system. *rdcd* can be configured to run with (a) full-capabilities which includes ability to set or change GPU configuration or (b) monitor-only capabilities which limits to monitoring GPU metrics.

### Start RDCD from command-line

When *rdcd* is started from a command-line the *capabilities* are determined by privilege of the *user* starting *rdcd*

    ## NOTE: Replace /opt/rocm with specific rocm version if needed

    ## To run with authentication. Ensure SSL keys are setup properly
    /opt/rocm/bin/rdcd           ## rdcd is started with monitor-only capabilities
    sudo /opt/rocm/bin/rdcd      ## rdcd is started will full-capabilities

    ## To run without authentication. SSL key & certificates are not required.
    /opt/rocm/bin/rdcd -u        ## rdcd is started with monitor-only capabilities
    sudo /opt/rocm/bin/rdcd -u   ## rdcd is started will full-capabilities

### Start RDCD using systemd

*rdcd* can be started by using the systemctl command. You can copy `/opt/rocm/libexec/rdc/rdc.service`, which is installed with RDC, to the systemd folder. This file has 2 lines that control what *capabilities* with which *rdcd* will run. If left uncommented, rdcd will run with full-capabilities.

    ## file: /opt/rocm/libexec/rdc/rdc.service
    ## Comment the following two lines to run with monitor-only capabilities
    CapabilityBoundingSet=CAP_DAC_OVERRIDE
    AmbientCapabilities=CAP_DAC_OVERRIDE

    systemctl start rdc         ## start rdc as systemd service

Additional options can be passed to *rdcd* by modifying `/opt/rocm/etc/rdc_options`

    ## file: /opt/rocm/etc/rdc_options
    # Append 'rdc' daemon parameters here
    RDC_OPTS="-p 50051 -u -d"

Example above does the following:

- Use port 50051
- Use unauthenticated mode
- Enable debug messages
- **NOTE:** You must add `-u` flag to `rdci` calls as well

## Invoke RDC using ROCm™ Data Center Interface (RDCI)

RDCI provides command-line interface to all RDC features. This CLI can be run locally or remotely. Refer to [**user guide**](https://rocm.docs.amd.com/projects/rdc/en/latest/how-to/features.html) for the current list of features.

    ## sample rdci commands to test RDC functionality
    ## discover devices in a local or remote compute node
    ## NOTE: option -u (for unauthenticated) is required if rdcd was started in this mode
    ## Assuming that rdc is installed into /opt/rocm

    cd /opt/rocm/bin
    ./rdci discovery -l <-u>                    ## list available GPUs in localhost
    ./rdci discovery <host> -l <-u>             ## list available GPUs in host machine
    ./rdci dmon <host> <-u> -l                  ## list most GPU counters
    # assuming rdcd is running locally, using -u instead of <host>
    ./rdci dmon -u --list-all                   ## list all GPU counters
    ./rdci dmon -u -i 0 -c 1 -e 100             ## monitor field 100 on gpu 0 for count of 1
    ./rdci dmon -u -i 0 -c 1 -e 1,2             ## monitor fields 1,2 on gpu 0 for count of 1

## Known issues

### dmon fields return N/A

1. An optional library might be missing. Do you have
   `/opt/rocm/lib/rdc/librdc_*.so`? Do you have the library it's related to?
   (rocprofiler, rocruntime...)
2. The GPU you're using might not be supported. As a rule of thumb - most
   metrics should work on MI300 and up. Less metrics are supported for MI200.
   NV21 (aka RX6800) and other consumer GPUs have even less metrics.

### dmon rocprofiler fields return zeros

Due to a rocprofiler limitation - you must set `HSA_TOOLS_LIB` environmental
variable *before* running a compute job.

If `HSA_TOOLS_LIB` is not set - most rocprofiler metrics will return all zeros.

E.g. Correct output on MI300 using [gpu-burn](https://github.com/ROCm/HIP-Examples/tree/master/gpu-burn)

    # terminal 1
    rdcd -u
    # terminal 2
    export HSA_TOOLS_LIB=/opt/rocm/lib/librocprofiler64.so.1
    gpu-burn
    # terminal 3
    rdci dmon -u -e 800,801,803 -i 0 -c 1
    # output:
    # GPU   MN_OCC_PER_CU       MN_OCC_PER_ACT_CU   ACTIVE_WAVES
    # 0     1683.422            6479.242            32640.000

### `HSA_STATUS_ERROR_OUT_OF_RESOURCES`

error:

    terminate called after throwing an instance of 'std::runtime_error'
      what():  hsa error code: 4104 HSA_STATUS_ERROR_OUT_OF_RESOURCES: The runtime failed to allocate the necessary resources. This error may also occur when the core runtime library needs to spawn threads or create internal OS-specific events.
    Aborted (core dumped)

1. Missing groups. Run `groups`. You're expected to have `video` and `render`.
   This can be fixed with `sudo usermod -aG video,render $USER` followed by
   logging out and logging back in.

## Troubleshooting rdcd

- Log messages that can provide useful debug information.

If rdcd was started as a systemd service, then use journalctl to view rdcd logs

    sudo journalctl -u rdc

To run rdcd with debug log from command-line use
version will be the version number(ex:3.10.0) of ROCm where RDC was packaged with

    RDC_LOG=DEBUG /opt/rocm/bin/rdcd

RDC_LOG=DEBUG also works on rdci

ERROR, INFO, DEBUG logging levels are supported

Additional logging messages can be enabled with `RSMI_LOGGING=3`
