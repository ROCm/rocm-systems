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
 - Code example: test/MatrixTranspose/MatrixTranspose.cpp
```

## To build and run test
```
  cd <your path>

 - CLone development branches of roctracer and HIP/HCC:
  git clone -b amd-master https://github.com/ROCmSoftwarePlatform/roctracer.git
  git clone -b master https://github.com/ROCm-Developer-Tools/HIP
  git clone --recursive -b clang_tot_upgrade https://github.com/RadeonOpenCompute/hcc

 - Set environment:
  export HIP_PATH=<your path>/HIP
  export HCC_HOME=<your path>/hcc/build
  export CMAKE_PREFIX_PATH=/opt/rocm/lib

 - Build HCC:
  cd <your path>/hcc && mkdir build && cd build &&
  cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/ext .. && make -j <nproc>
  
 - Build HIP:
  cd <your path>/HIP && mkdir build && cd build &&
  cmake -DUSE_PROF_API=1 -DPROF_API_HEADER_PATH=<your path>/roctracer/inc/ext .. && make -j <nproc>
  ln -s <your path>/HIP/build <your path>/HIP/lib
  
 - Build ROCtracer
  cd <your path>/roctracer && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm .. && make -j <nproc>

 - To build and run test
  make mytest
  run.sh
  
 - To install
  make install
 or
  make package && dpkg -i *.deb
```
