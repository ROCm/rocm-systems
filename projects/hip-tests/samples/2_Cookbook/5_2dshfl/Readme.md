## Warp shfl operations  in 2D ###

This tutorial is follow-up of the previous tutorial, where we learned how to use shfl ops. In this tutorial, we'll explain how to scale similar kind of operations to multi-dimensional space by using previous tutorial source-code.

## Introduction:

Let's talk about Warp first. The kernel code is executed in groups of fixed number of threads known as Warp. For nvidia WarpSize is 32 while for AMD, 32 for Polaris architecture and 64 for rest. Threads in a warp are referred to as lanes and are numbered from 0 to warpSize -1. With the help of shfl ops, we can directly exchange values of variable between threads without using any memory ops within a warp. There are four types of shfl ops:
```
   int   __shfl      (int var,   int srcLane, int width=warpSize);
   float __shfl      (float var, int srcLane, int width=warpSize);
   int   __shfl_up   (int var,   unsigned int delta, int width=warpSize);
   float __shfl_up   (float var, unsigned int delta, int width=warpSize);
   int   __shfl_down (int var,   unsigned int delta, int width=warpSize);
   float __shfl_down (float var, unsigned int delta, int width=warpSize);
   int   __shfl_xor  (int var,   int laneMask, int width=warpSize);
   float __shfl_xor  (float var, int laneMask, int width=warpSize);
```

## Requirement:
For hardware requirement and software installation [Installation](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the previous tutorial and modify it to learn how to use shared memory.

## __shfl ops in 2D

In the same sourcecode, we used for MatrixTranspose. We'll add the following:
```
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	out[x*width + y] = __shfl(val,y*width + x);
```

With the help of this application, we can say that kernel code can be converted into  multi-dimensional threads with ease.

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./2dshfl
Device name AMD Radeon RX 6900 XT
PASSED!
```
## requirement for nvidia
please make sure you have a 3.0 or higher compute capable device in order to use warp shfl operations and add `-gencode arch=compute=30, code=sm_30` nvcc flag in the Makefile while using this application.

## More Info:

- [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/faq.html)
- [HIP Kernel Language](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP Terminology](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/terms.html) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md)
