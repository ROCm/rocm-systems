### This sample tests CXX Language support with amdclang++

ROCM_PATH is the path where ROCM is installed. default path is /opt/rocm.

I. Build

```
$ mkdir -p build; cd build
$ rm -rf *;
$ CXX=<ROCM_PATH>/llvm/bin/amdclang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make
```
To enable compiler auto detection of gpu users may need to add ADMGPU support as command line option, if test failed to run, for example,
```
$ CXX=<ROCM_PATH>/llvm/bin/amdclang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS="gfx1102" ..
```
II. Test

```
$ ./square
info: running on device AMD Radeon Graphics
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
```
