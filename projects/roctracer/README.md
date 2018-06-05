ROC Tracer library, Callback/Activity APIs

The library source tree:
 - doc  - Documentation
 - inc/roctracer.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
 - test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample

Environment example:
$ export HIP_PATH=/opt/rocm/hip
$ export HCC_HOME=/opt/rocm/hcc
$ export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

To build:
$ cd <your path>/roctracer/build
$ cmake ..
$ make

To rebuild and run test:
$ export ITERATIONS=<iterations number [100]>
$ cd <your path>/roctracer/test/MatrixTranspose
$ make
