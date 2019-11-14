# ROC-tracer
```
ROC-tracer library, Runtimes Generic Callback/Activity APIs.
The goal of the implementation is to provide a generic independent from
specific runtime profiler to trace API and asyncronous activity.

The API provides functionality for registering the runtimes API callbacks and
asyncronous activity records pool support.
```

## The library source tree
```
 - inc/roctracer.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
 - test - test suit
   - MatrixTranspose - test based on HIP MatrixTranspose sample
```

## Documentation
```
 - API description: inc/roctracer.h
 - Code example: test/MatrixTranspose_test/MatrixTranspose.cpp
```

## To build and run test
```
 - ROCm-2.3 or higher is required
 - Python2.7 is required.
  The required modules: CppHeaderParser, argparse.
  To install:
  sudo pip install CppHeaderParser argparse

 - CLone development branch of roctracer:
  git clone -b amd-master https://github.com/ROCm-Developer-Tools/roctracer

 - Set environment:
  export HIP_PATH=/opt/rocm/hip
  export HCC_HOME=/opt/rocm/hcc
  export CMAKE_PREFIX_PATH=/opt/rocm

 - Build ROCtracer
  export CMAKE_BUILD_TYPE=<debug|release> # release by default
  cd <your path>/roctracer && mkdir build && cd build && cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm .. && make -j <nproc>

 - To build and run test
  make mytest
  run.sh
  
 - To install
  make install
 or
  make package && dpkg -i *.deb
```
