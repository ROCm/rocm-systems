# ROC-tracer

ROC tracer library. Callback/Activity APIs

The library source tree:
 - doc  - Documentation
 - inc/roctracer.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
 - test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample

## Environment example
```
  export HIP_PATH=/opt/rocm/hip
  export HCC_PATH=/opt/rocm/hcc
  export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa
```

## To build
```
  cd <your path>/roctracer/build
  cmake ..
  make
```

## To rebuild and run test
```
 - Set iterations number, 100 by default
   export ITERATIONS=<iterations number>

 - Set HCC_HOME env var to use custom HCC build

  <your path>/roctracer/build$ make mytest

  or

  cd <your path>/roctracer/test/MatrixTranspose
  make
```
