Build procedure

The CMakeLists.txt at hip-tests/samples folder can be used for building and packaging samples.

CMakeLists.txt can support shared and static libs of hip-rocclr runtime.
The same steps can be followed for both.

1. To build a specific sample (e.g. 0_Intro/bit_extract) run ..

cd samples/0_Intro/bit_extract

mkdir -p build && cd build

cmake ..

make all

2. To build all samples together run ..

cd hip-tests

mkdir -p build && cd build

rm -rf * (to clear up)

cmake ../samples

make build_samples

In order to build specific samples (Intro, Utils or Cookbook) run ..

make build_intro
make build_utils
make build_cookbook

Note that if you want debug version, add "-DCMAKE_BUILD_TYPE=Debug" in cmake cmd.

3. To package samples and generate packages. From hip-tests/build

cmake ../samples

make package_samples

## Note: sample 2_Cookbook/22_cmake_hip_lang is current not included in toplevel cmake. To build this sample from toplevel cmake, uncomment Line 43 inside samples/2_Cookbook/CMakeLists.txt.
