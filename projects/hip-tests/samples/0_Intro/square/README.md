# Square.md

ROCM_PATH is the path where ROCM is installed. default path is /opt/rocm.

Simple test below is an example, shows how to use hipify-perl to port CUDA code to HIP:

- Add hip/bin path to the PATH
```
$ export PATH=$PATH:[MYHIP]/bin
```

- Define environment variable
```
$ export HIP_PATH=[MYHIP]
```

- Build executable file
```
$ cd ~/hip/samples/0_Intro/square
  mkdir -p build && cd build

  cmake ..
  make

$ Building without cmake
<ROCM_PATH>/hip/bin/hipify-perl square.cu > square.cpp
<ROCM_PATH>/llvm/bin/amdclang++ -x hip --offload-arch=native square.cpp -o square.out
```
- Execute file
```
$ ./square.out
info: running on device AMD Radeon RX 6900 XT
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
```
