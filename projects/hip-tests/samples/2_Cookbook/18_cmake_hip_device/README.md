### This will test linking hip::device interface in cmake
I. Build

```
$ mkdir build; cd build
$ CXX="$(hipconfig -l)"/clang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make
```

Note, users may need to add ADMGPU support as command line option, if test failed to run, for example,
```
$ CXX="$(hipconfig -l)"/clang++ cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS="gfx1102" ..
```

II. Test

```
$ ../test_cpp
info: running on device AMD Radeon Graphics
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
```
