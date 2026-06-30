To build:

$ mkdir build
$ cd build
$ cmake -DCMAKE_MODULE_PATH=<your absolute path>/aqlprofile/cmake_modules -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make

To test:

$ export LD_LIBRARY_PATH=/opt/rocm/hsa-amd-aqlprofile/lib
$ ./run.sh
