### This will test cmake lang support: CXX and Fortran
I. Prepare
1) You must install cmake version 3.18 or above to support LINK_LANGUAGE.
   Otherwise, Fortran build will fail.
   To download the latest cmake, see https://cmake.org/download/.
2) If there is no Fortran on your system, you must install it via,
   sudo apt install gfortran

II. Build
```
$ mkdir -p build; cd build
$ rm -rf *;
$ CXX="$(hipconfig -l)"/clang++ FC=$(which gfortran) cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ cmake ..
$ make
```

Note, users may need to add AMD GPU support, if test failed, for example,
```
$ CXX="$(hipconfig -l)"/clang++ FC=$(which gfortran) cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS="gfx1102" ..
```
To enable compiler auto detection of gpu users may need to add ADMGPU support as command line option,
if test failed to run, for example,
```
$ CXX="$(hipconfig -l)"/clang++ FC=$(which gfortran) cmake -DCMAKE_PREFIX_PATH=/opt/rocm -DAMDGPU_TARGETS=native ..
```
III. Test
```
$ ./test_fortran
 Succeeded testing Fortran!

$ ./test_cpp
Device name AMD Radeon Graphics
PASSED!
```
