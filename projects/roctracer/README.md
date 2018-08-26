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

## To build and run test
```
  cd <your path>

 - CLone roctracer and custom branches of HIP/HCC:
  git clone -b amd-master https://github.com/ROCmSoftwarePlatform/roctracer.git
  git clone -b roctracer-hip-frontend-180826 https://github.com/eshcherb/HIP.git
  git clone --recursive -b roctracer-hip-frontend-180823 https://github.com/eshcherb/hcc.git

 - Set environment:
  export HIP_PATH=<your path>/HIP
  export HCC_PATH=<your path>/hcc
  export HCC_HOME=<your path>/hcc/lib
  export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

 - Build HCC:
  cd <your path>/hcc
  mkdir build && cd build && cmake -DUSE_PROF_API=1 .. && make -j <nproc>
  
 - Build HIP:
  cd <your path>/HIP && mkdir build && cd build && cmake -DUSE_PROF_API=1 .. && make -j <nproc>
  ln -s HIP/build HIP/lib
  
 - Build ROCtracer
  cd <your path>/roctracer && mkdir build && cd build && cmake .. && make -j <nproc>

 - To run test
  make mytest
 
 or
  cd <your path>/roctracer/test/MatrixTranspose
  make

 Set test iterations number, 100 by default
  export ITERATIONS=<iterations number>
```
