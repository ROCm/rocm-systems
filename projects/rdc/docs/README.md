
# Radeon Data Center Tools

TODO: Add general description of RDC, link to github site 

# Important note about Versioning and Backward Compatibility
RDC library is currently under development, and therefore subject to change either at the ABI or API level. The intention is to keep the API as stable as possible even while in development, but in some cases we may need to break backwards compatibility in order to ensure future stability and usability. Following [Semantic Versioning](https://semver.org/) rules, while the ROCm SMI library is in high state of change, the major version will remain 0, and backward compatibility is not ensured.

Once new development has leveled off, the major version will become greater than 0, and backward compatibility will be enforced between major versions.

# Building RDC

#### Additional Required software for building
In order to build the RDC software, the following components are required. Note that the software versions listed are what was used in development. Earlier versions are not guaranteed to work:
TODO: see if protoc and c++ plugin can be installed from packages rather than built from
source. This will be necessary for CI integration.
* CMake (v3.5.0)
* g++ (5.4.0)
* ROCm
* [ROCm SMI Library](https://github.com/RadeonOpenCompute/rocm_smi_lib)
* [gRPC and protoc](https://github.com/grpc/grpc/blob/master/src/cpp/README.md#make)

In order to build the latest documentation, the following are required:
* DOxygen (1.8.11)
* latex (pdfTeX 3.14159265-2.6-1.40.16)

The source code for RDC is available on [ADD RDC GITHUB URL HERE]().

After the RDC library git repository has been cloned to a local Linux machine, building  is achieved by following the typical CMake build sequence. Specifically,
##### ```$ mk -p build```
##### ```$ cd build```
##### ```$ cmake -DROCM_DIR=<location of ROCm root, including ROCm SMI> <location of root of ROCm SMI library CMakeLists.txt>```
##### ```$ make```
##### ```# Install library file and header; default location is /opt/rocm```
##### ```$ make install```
The built library will appear in the `build` folder.

#### Building the Documentation
The documentation PDF file can be built with the following steps (continued from the steps above):
##### ```$ make doc```
##### ```$ cd latex```
##### ```$ make```
The reference manual, `refman.pdf` will be in the `latex` directory upon a successful build.

#### Building the Tests UPDATE FOR RDC
In order to verify the build and capability of ROCm SMI on your system and to see an example of how ROCm SMI can be used, you may build and run the tests that are available in the repo. To build the tests, follow these steps:

##### ```$ cd <ROCM SMI source root>```
##### ```$ mkdir lib```
##### ```$ cd lib```
##### ```$ cmake -DROCM_DIR=<ROCM SMI source root> <ROCm SMI source root>/tests/rocm_smi_test```
##### ```$ make

TODO: THE REMAINDER NEEDS TO BE TAILORED FOR RDC

To run the test, execute the program `rsmitst` that is built from the steps above.

# Usage Basics

# Hello RDC

