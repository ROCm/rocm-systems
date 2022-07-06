
# ROCm<sup>TM</sup> Data Center Tool (RDC)

The ROCm™ Data Center Tool simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

- GPU telemetry
- GPU statistics for jobs
- Integration with third-party tools
- Open source

For up-to-date document and how to start using RDC from pre-built packages, please refer to [**user guide**](https://docs.amd.com/bundle/ROCm-Data-Center-Tool-User-Guide-v5.1/page/Overview.html)

# Supported platforms
RDC can run on AMD ROCm supported platforms, please refer to [List of Supported Operating Systems](https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html)

# Building RDC from source

## Dependencies

    CMake 3.15                              ## 3.15 or greater is required for gRPC
    g++ (5.4.0)
    Doxygen (1.8.11)                        ## required to build the latest documentation
    Latex (pdfTeX 3.14159265-2.6-1.40.16)   ## required to build the latest documentation
    gRPC and protoc                         ## required for communication
    libcap-dev                              ## required to manage the privileges.

    AMD ROCm platform (https://github.com/RadeonOpenCompute/ROCm)
        * It is recommended to install the complete AMD ROCm platform.
          For installation instruction see https://rocmdocs.amd.com/en/latest/Current_Release_Notes/Current-Release-Notes.html
        * At the minimum, these two components are required
            (i)  AMD ROCm SMI Library (https://github.com/RadeonOpenCompute/rocm_smi_lib)
            (ii) AMD ROCk Kernel driver (https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver)

## Building gRPC and protoc
**NOTE:** gRPC and protoc compiler must be built when building RDC from source as pre-built packages are not available. When installing RDC from a package, gRPC and protoc will be installed from the package.

**IMPORTANT:** Building gRPC and protocol buffers requires CMake 3.15 or greater. With an older version build will quietly succeed with a *message*. However, all components of gRPC will not be installed and RDC will ***fail*** to run

The following tools are required for gRPC build & installation

    automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl

 Download and build gRPC

    $ git clone -b v1.44.0 https://github.com/grpc/grpc
    $ cd grpc
    $ git submodule update --init
    $ mkdir -p cmake/build
    $ cd cmake/build

    ## By default (without using CMAKE_INSTALL_PREFIX option), the following will install to /usr/local lib, include and bin directories

    $ cmake -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON <-DCMAKE_INSTALL_PREFIX=<install dir>> ../..
    $ make
    $ sudo make install
    $ echo "<install dir>/lib" | sudo tee /etc/ld.so.conf.d/grpc.conf

## Building RDC

Clone the RDC source code from GitHub and use CMake to build and install

    $ git clone https://github.com/RadeonOpenCompute/rdc
    $ cd rdc
    $ mkdir -p build; cd build
    $ cmake -DROCM_DIR=/opt/rocm  -DGRPC_ROOT="$GRPC_PROTOC_ROOT" <-DCMAKE_INSTALL_PREFIX=<install dir>> ..
    $ make
    $ make install ## default installation location is /opt/rocm

## Building RDC library only without gRPC (optional)

If only the RDC libraries are needed (i.e. only "embedded mode" is required), the user can choose to not build rdci and rdcd. This will eliminate the need for gRPC and protoc. To build in this way, -DBUILD_STANDALONE=off should be passed on the the cmake command line:

    $ cmake -DROCM_DIR=/opt/rocm -DBUILD_STANDALONE=off <-DCMAKE_INSTALL_PREFIX=<install dir>> ..

## Building RDC library without ROCM Run time (optional)

The user can choose to not build RDC diagnostic ROCM Run time. This will eliminate the need for ROCM Run time. To build in this way, -DBUILD_ROCRTEST=off should be passed on the the cmake command line:

    $ cmake -DROCM_DIR=/opt/rocm -DBUILD_ROCRTEST=off <-DCMAKE_INSTALL_PREFIX=<install dir>> ..

## Update System Library Path

The following commands need to be executed as root (sudo). It may be easiest to put them into a script and then run that script as root:

    $ RDC_LIB_DIR=<RDC install dir>/lib
    $ GRPC_LIB_DIR=<gRPC install dir>/lib
    $ echo "$GRPC_LIB_DIR" > /etc/ld.so.conf.d/x86_64-librdc_client.conf
    $ echo "$GRPC_LIB_DIR"64 >> /etc/ld.so.conf.d/x86_64-librdc_client.conf
    $ echo "$RDC_LIB_DIR" >> /etc/ld.so.conf.d/x86_64-librdc_client.conf
    $ echo "$RDC_LIB_DIR"64 >> /etc/ld.so.conf.d/x86_64-librdc_client.conf
    $ ldconfig

# Running RDC
RDC supports encrypted communications between clients and servers. The
communication can be configured to be *authenticated* or *not authenticated*. The [**user guide**](https://docs.amd.com/bundle/ROCm-Data-Center-Tool-User-Guide-v5.1/page/Developer_Handbook.html) has information on how to generate and install SSL keys and certificates for authentication. By default, authentication is enabled.

## Starting ROCm™ Data Center Daemon (RDCD)
For an RDC client application to monitor and/or control a remote system, the RDC server daemon, *rdcd*, must be running on the remote system. *rdcd* can be configured to run with (a) full-capabilities which includes ability to set or change GPU configuration or (b) monitor-only capabilities which limits to monitoring GPU metrics.

### Start RDCD from command-line
When *rdcd* is started from a command-line the *capabilities* are determined by privilege of the *user* starting *rdcd*

    $ cd rdc_install_prefix     ## If specified in Building RDC section

    ## To run with authentication. Ensure SSL keys are setup properly
    ## version will be the version number(ex:3.10.0) of ROCm where RDC was pacakged with
    $ /opt/rocm-<version>/rdc/bin/rdcd           ## rdcd is started with monitor-only capabilities
    $ sudo /opt/rocm-<version>/rdc/bin/rdcd      ## rdcd is started will full-capabilities

    ## To run without authentication. SSL key & certificates are not required.
    ## version will be the version number(ex:3.10.0) of ROCm where RDC was pacakged with
    $ /opt/rocm-<version>/rdc/bin/rdcd -u        ## rdcd is started with monitor-only capabilities
    $ sudo /opt/rocm-<version>/rdc/bin/rdcd -u   ## rdcd is started will full-capabilities

### Start RDCD using systemd
*rdcd* can be started by using the systemctl command. You can copy /opt/rocm-\<version\>/rdc/lib/rdc.service, which is installed with RDC, to the systemd folder. This file has 2 lines that control what *capabilities* with which *rdcd* will run. If left uncommented, rdcd will run with full-capabilities.


    ## file: /opt/rocm-<version>/rdc/lib/rdc.service
    ## Comment the following two lines to run with monitor-only capabilities
    CapabilityBoundingSet=CAP_DAC_OVERRIDE
    AmbientCapabilities=CAP_DAC_OVERRIDE

    systemctl start rdc         ## start rdc as systemd service

## Invoke RDC using ROCm™ Data Center Interface (RDCI)
RDCI provides command-line interface to all RDC features. This CLI can be run locally or remotely. Refer to [**user guide**](https://docs.amd.com/bundle/ROCm-Data-Center-Tool-User-Guide-v5.1/page/Feature_Overview.html) for the current list of features.

    ## sample rdci commands to test RDC functionality
    ## discover devices in a local or remote compute node
    ## NOTE: option -u (for unauthenticated) is required is rdcd was started in this mode

    $ cd rdc_install_prefix                           ## If specified in Building RDC section
    ./opt/rocm-<version>/rdc/bin/rdci discovery -l <-u>         ## list available GPUs in localhost
    ./opt/rocm-<version>/rdc/bin/rdci discovery <host> -l <-u>  ## list available GPUs in host machine

## Troubleshooting rdcd

Log messages that can provide useful debug information.

    ## If rdcd was started as a systemd service, then use journalctl to view rdcd logs
    journalctl -u rdc

    ## To run rdcd with debug log from command-line use
    ## version will be the version number(ex:3.10.0) of ROCm where RDC was pacakged with
    RDC_LOG=DEBUG /opt/rocm-<version>/rdc/bin/rdcd

