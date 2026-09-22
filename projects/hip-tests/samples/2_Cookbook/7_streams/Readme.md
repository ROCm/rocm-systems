## Streams ###

In all Earlier tutorial we used single stream, In this tutorial, we'll explain how to launch multiple streams.

## Introduction:

The various instances of kernel to be executed on device in exact launch order defined by Host are called streams. We can launch multiple streams on a single device. We will learn how to learn two streams which can we scaled with ease.

## Requirement:
For hardware requirement and software installation [Installation](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the previous tutorial and modify it to learn how to launch multiple streams.

## Streams

In this tutorial, we'll use both instances of shared memory (i.e., static and dynamic) as different streams. We declare stream as follows:
`  hipStream_t streams[num_streams];                                             `

and create stream using `hipStreamCreate` as follows:
```
for(int i=0;i<num_streams;i++)
    hipStreamCreate(&streams[i]);
```

and while kernel launch, we make the following changes in 5th parameter to hipLaunchKernelGGL(having 0 as the default stream value):

```
 hipLaunchKernelGGL(matrixTranspose_static_shared,
                    dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                    0, streams[0],
                    gpuTransposeMatrix[0], data[0], width);
```

```
 hipLaunchKernelGGL(matrixTranspose_dynamic_shared,
                    dim3(WIDTH/THREADS_PER_BLOCK_X, WIDTH/THREADS_PER_BLOCK_Y),
                    dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y),
                    sizeof(float)*WIDTH*WIDTH, streams[1],
                    gpuTransposeMatrix[1], data[1], width);
```

here we replaced 4th parameter with amount of additional shared memory to allocate when launching the kernel.

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./stream
stream PASSED!
```

## More Info:
- [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/faq.html)
- [HIP Kernel Language](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP Terminology](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/terms.html) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md)
