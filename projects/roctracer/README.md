ROC Tracer library, Callback/Activity APIs

The library source tree:
 - doc  - Documentation
 - inc/roctracer.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
 - test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample

Environment:
$ export HIP_PATH=<HIP path>
$ export HCC_HOME=<HCC path>
$ export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

To build:
$ cd <your path>/roctracer/build
$ cmake ..
$ make

To rebuild and run test:
$ export ITERATIONS=<iterations number [100]>
$ cd <your path>/roctracer/test/MatrixTranspose
$ make
