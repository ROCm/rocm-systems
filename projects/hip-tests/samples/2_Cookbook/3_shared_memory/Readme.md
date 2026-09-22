## Using shared memory ###

Earlier we learned how to write our first hip program, in which we compute Matrix Transpose. In this tutorial, we'll explain how to use the shared memory to improve the performance.

## Introduction:

As we mentioned earlier  that Memory bottlenecks is the main problem why we are not able to get the highest performance, therefore minimizing the latency for memory access plays prominent role in application optimization. In this tutorial, we'll learn how to use static shared memory and will explain the dynamic one latter.

## Requirement:
For hardware requirement and software installation [Installation](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the previous tutorial and modify it to learn how to use shared memory.

## Shared Memory

Shared memory is way more faster than that of global and constant memory and accessible to all the threads in the block. If the size of shared memory is known at compile time, we can specify the size and will use the static shared memory. In the same sourcecode, we will use the `__shared__` variable type qualifier as follows:

`  __shared__ float sharedMem[1024*1024];`

Be careful while using shared memory, since all threads within the block can access the shared memory, we need to sync the operation of individual threads by using:

`  __syncthreads();`

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./sharedMemory
Device name AMD Radeon RX 6900 XT
PASSED!
```

## More Info:
- [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/faq.html)
- [HIP Kernel Language](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP Terminology](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/terms.html) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md)
