
# ROCm<sup>TM</sup> Data Center Tool (RDC)

The ROCm™ Data Center Tool simplifies the administration and addresses key infrastructure challenges in AMD GPUs in cluster and datacenter environments. The main features are:

- GPU telemetry
- GPU statistics for jobs
- Integration with third-party tools
- Open source

For up-to-date document and how to start using RDC from pre-built packages, please refer to the [**ROCm DataCenter Tool User Guide**](https://docs.amd.com/bundle?name_filter.field=title&name_filter.value=ROCm%20DataCenter%20Tool%20User%20Guide&rpp=10&sort.field=title&sort.value=desc)

# Supported platforms
RDC can run on AMD ROCm supported platforms, please refer to the **List of Supported Operating Systems** under [ROCm Release Notes](https://docs.amd.com/bundle?name_filter.field=title&name_filter.value=ROCm%20Release%20Notes&rpp=10&sort.field=title&sort.value=desc)

# Building RDC from source

## Dependencies

    CMake 3.15                              ## 3.15 or greater is required for gRPC
    g++ (5.4.0)
    Doxygen (1.8.11)                        ## required to build the latest documentation
    Latex (pdfTeX 3.14159265-2.6-1.40.16)   ## required to build the latest documentation
    gRPC and protoc                         ## required for communication
    libcap-dev                              ## required to manage the privileges.
    rocmtools                               ## required for profiler metrics

    AMD ROCm platform (https://github.com/RadeonOpenCompute/ROCm)
        * It is recommended to install the complete AMD ROCm platform.
          For installation instruction see https://docs.amd.com/category/Release%20Documentation
        * At the minimum, these two components are required
            (i)  AMD ROCm SMI Library (https://github.com/RadeonOpenCompute/rocm_smi_lib)
            (ii) AMD ROCk Kernel driver (https://github.com/RadeonOpenCompute/ROCK-Kernel-Driver)
        * For profiler metrics, this component is required:
            (i)  AMD ROCm Tools (https://github.com/ROCm-Developer-Tools/rocmtools)

## Building gRPC and protoc
**NOTE:** gRPC and protoc compiler must be built when building RDC from source as pre-built packages are not available. When installing RDC from a package, gRPC and protoc will be installed from the package.

**IMPORTANT:** Building gRPC and protocol buffers requires CMake 3.15 or greater. With an older version build will quietly succeed with a *message*. However, all components of gRPC will not be installed and RDC will ***fail*** to run

The following tools are required for gRPC build & installation

    automake make g++ unzip build-essential autoconf libtool pkg-config libgflags-dev libgtest-dev clang-5.0 libc++-dev curl

Download and build gRPC
```bash
git clone -b v1.44.0 https://github.com/grpc/grpc
cd grpc
git submodule update --init --recursive
mkdir -p build
```

By default (without using CMAKE_INSTALL_PREFIX option), gRPC will install to /usr/local lib, include and bin directories.  
It is highly recommended to install gRPC into a unique directory.  
Below example installs gRPC into /opt/grpc  
```bash
export GRPC_ROOT=/opt/grpc
cmake -B build -DgRPC_INSTALL=ON -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX="$GRPC_ROOT"
make -C build -j $(nproc)
sudo make -C build install
echo "$GRPC_ROOT" | sudo tee /etc/ld.so.conf.d/grpc.conf
```

## Building RDC

Clone the RDC source code from GitHub and use CMake to build and install

```bash
git clone https://github.com/RadeonOpenCompute/rdc
cd rdc
mkdir -p build
# default installation location is /opt/rocm, specify with -DROCM_DIR or -DCMAKE_INSTALL_PREFIX
cmake -B build -DGRPC_ROOT="$GRPC_ROOT"
make -C build -j $(nproc)
make -C build install
```

## Building RDC library only without gRPC (optional)

If only the RDC libraries are needed (i.e. only "embedded mode" is required), the user can choose to not build rdci and rdcd. This will eliminate the need for gRPC and protoc. To build in this way, -DBUILD_STANDALONE=off should be passed on the the cmake command line:

```bash
cmake -B build -DBUILD_STANDALONE=off
```

## Building RDC library without ROCM Run time (optional)

The user can choose to not build RDC diagnostic ROCM Run time. This will eliminate the need for ROCM Run time. To build in this way, -DBUILD_ROCRTEST=off should be passed on the the cmake command line:

```bash
cmake -B build -DBUILD_ROCRTEST=off
```

## Update System Library Path

```bash
RDC_LIB_DIR=/opt/rocm/rdc/lib
GRPC_LIB_DIR=/opt/grpc/lib
echo -e "${GRPC_LIB_DIR}\n${GRPC_LIB_DIR}64" | sudo tee /etc/ld.so.conf.d/x86_64-librdc_client.conf
echo -e "${RDC_LIB_DIR}\n${RDC_LIB_DIR}64" | sudo tee -a /etc/ld.so.conf.d/x86_64-librdc_client.conf
ldconfig
```

# Running RDC
RDC supports encrypted communications between clients and servers. The
communication can be configured to be *authenticated* or *not authenticated*. The [**user guide**](https://docs.amd.com/bundle/ROCm-Data-Center-Tool-User-Guide-v5.3/page/Developer_Handbook.html) has information on how to generate and install SSL keys and certificates for authentication. By default, authentication is enabled.

## Starting ROCm™ Data Center Daemon (RDCD)
For an RDC client application to monitor and/or control a remote system, the RDC server daemon, *rdcd*, must be running on the remote system. *rdcd* can be configured to run with (a) full-capabilities which includes ability to set or change GPU configuration or (b) monitor-only capabilities which limits to monitoring GPU metrics.

### Start RDCD from command-line
When *rdcd* is started from a command-line the *capabilities* are determined by privilege of the *user* starting *rdcd*

```bash
## NOTE: Replace /opt/rocm with specific rocm version if needed

## To run with authentication. Ensure SSL keys are setup properly
## version will be the version number(ex:3.10.0) of ROCm where RDC was packaged with
/opt/rocm/rdc/bin/rdcd           ## rdcd is started with monitor-only capabilities
sudo /opt/rocm/rdc/bin/rdcd      ## rdcd is started will full-capabilities

## To run without authentication. SSL key & certificates are not required.
## version will be the version number(ex:3.10.0) of ROCm where RDC was packaged with
/opt/rocm/rdc/bin/rdcd -u        ## rdcd is started with monitor-only capabilities
sudo /opt/rocm/rdc/bin/rdcd -u   ## rdcd is started will full-capabilities
```

### Start RDCD using systemd
*rdcd* can be started by using the systemctl command. You can copy /opt/rocm/rdc/lib/rdc.service, which is installed with RDC, to the systemd folder. This file has 2 lines that control what *capabilities* with which *rdcd* will run. If left uncommented, rdcd will run with full-capabilities.

```bash
## file: /opt/rocm/rdc/lib/rdc.service
## Comment the following two lines to run with monitor-only capabilities
CapabilityBoundingSet=CAP_DAC_OVERRIDE
AmbientCapabilities=CAP_DAC_OVERRIDE
```

```bash
systemctl start rdc         ## start rdc as systemd service
```

## Invoke RDC using ROCm™ Data Center Interface (RDCI)
RDCI provides command-line interface to all RDC features. This CLI can be run locally or remotely. Refer to [**user guide**](https://docs.amd.com/bundle/ROCm-Data-Center-Tool-User-Guide-v5.3/page/Feature_Overview.html) for the current list of features.

```bash
## sample rdci commands to test RDC functionality
## discover devices in a local or remote compute node
## NOTE: option -u (for unauthenticated) is required if rdcd was started in this mode
## Assuming that rdc is installed into /opt/rocm

cd /opt/rocm/rdc/bin
./rdci discovery -l <-u>                    ## list available GPUs in localhost
./rdci discovery <host> -l <-u>             ## list available GPUs in host machine
./rdci dmon <host> <-u> -l                  ## list most GPU counters
# assuming rdcd is running locally, using -u instead of <host>
./rdci dmon -u --list-all                   ## list all GPU counters
./rdci dmon -u -i 0 -c 1 -e 100             ## monitor field 100 on gpu 0 for count of 1
./rdci dmon -u -i 0 -c 1 -e 1,2             ## monitor fields 1,2 on gpu 0 for count of 1
# below requires rocmtools to be installed
./rdci dmon -u -i 0 -c 5 -e 700             ## monitor field 700 on gpu 0 for count of 5
# below is only likely to work on MI series GPUs
./rdci dmon -u -i 0 -c 5 -e 700,701,702,706 ## monitor fields 700,701,702,706
```

## Troubleshooting rdcd

- Log messages that can provide useful debug information.

If rdcd was started as a systemd service, then use journalctl to view rdcd logs
```bash
sudo journalctl -u rdc
```

To run rdcd with debug log from command-line use
version will be the version number(ex:3.10.0) of ROCm where RDC was packaged with
```bash
RDC_LOG=DEBUG /opt/rocm/rdc/bin/rdcd
```

RDC_LOG=DEBUG also works on rdci

ERROR, INFO, DEBUG logging levels are supported

- Reading `RDC_FI_PROF_*` crashes rdcd
- All `RDC_FI_PROF_*` metrics return N/A

    1. Does your GPU support selected fields?  
        Field 700 (`RDC_FI_PROF_ELAPSED_CYCLES`) is supposed to be accessible on most GPUs.  
        Others are mostly intended for MI series.
    2. Set `RDC_LOG=DEBUG` as stated above
    3. Is rocmtools installed? Can you find `librocmtools.so`?
