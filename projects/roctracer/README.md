# ROC-tracer
```
ROC-tracer library, Runtimes Generic Callback/Activity APIs.
The goal of the implementation is to provide a generic independent from
specific runtime profiler to trace API and asyncronous activity.

The API provides functionality for registering the runtimes API callbacks and
asyncronous activity records pool support.
```

The library source tree:
 - inc/roctracer.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
 - test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample

## Documentation
```
 - API description: inc/roctracer.h
 - Code exaple: test/MatrixTranspose/MatrixTranspose.cpp
```

## To build and run test
```
  cd <your path>

 - CLone roctracer and custom branches of HIP/HCC:
  git clone -b amd-master https://github.com/ROCmSoftwarePlatform/roctracer.git
  git clone -b roctracer-hip-frontend-180826 https://github.com/eshcherb/HIP.git
  git clone --recursive -b roctracer-hip-frontend-180823 https://github.com/eshcherb/hcc.git

 - Set environment:
  export HIP_PATH=<your path>/HIP
  export HCC_HOME=<your path>/hcc/build
  export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

 - Build HCC:
  cd <your path>/hcc && mkdir build && cd build
  cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/roctracer ..
  make -j <nproc>
  
 - Build HIP:
  cd <your path>/HIP && mkdir build && cd build
  cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/roctracer ..
  make -j <nproc>
  ln -s HIP/build HIP/lib
  
 - Build ROCtracer
  cd <your path>/roctracer && mkdir build && cd build && cmake .. && make -j <nproc>

 - To run test
  make mytest
 
 or
  cd <your path>/roctracer/test/MatrixTranspose
  make

 set test iterations number, 100 by default
  export ITERATIONS=<iterations number>
```
