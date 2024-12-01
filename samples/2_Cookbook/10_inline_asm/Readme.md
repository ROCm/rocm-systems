## inline asm  ###

This tutorial is about how to use inline GCN asm in kernel. In this tutorial, we'll explain how to by using the simple Matrix Transpose.

## Introduction:

If you want to take advantage of the extra performance benefits of writing in assembly as well as take advantage of special GPU hardware features that were only available through assemby, then this tutorial is for you. In this tutorial we'll be explaining how to start writing inline asm in kernel.

For more insight Please read the following blogs by Ben Sander:
[The Art of AMDGCN Assembly: How to Bend the Machine to Your Will](https://gpuopen.com/learn/amdgcn-assembly/) and [AMD GCN Assembly: Cross-Lane Operations](http://gpuopen.com/amd-gcn-assembly-cross-lane-operations/).

For more information:
[AMD GCN3 ISA Architecture Manual](https://gpuopen.com/amd-isa-documentation/),
[User Guide for AMDGPU Back-end](https://llvm.org/docs/AMDGPUUsage.html).

## Requirement:
For hardware requirement and software installation [Installation](https://rocm.docs.amd.com/projects/HIP/en/latest/install/install.html)

## prerequiste knowledge:

Programmers familiar with CUDA, OpenCL will be able to quickly learn and start coding with the HIP API. In case you are not, don't worry. You choose to start with the best one. We'll be explaining everything assuming you are completely new to gpgpu programming.

## Simple Matrix Transpose

We will be using the Simple Matrix Transpose application from the our very first tutorial.

## asm() Assembler statement

In the same sourcecode, we used for MatrixTranspose. We'll add the following:

`  asm volatile ("v_mov_b32_e32 %0, %1" : "=v" (out[x*width + y]) : "v" (in[y*width + x]));                    `

GCN ISA In-line assembly, is supported. For example:

```
asm volatile ("v_mac_f32_e32 %0, %2, %3" : "=v" (out[i]) : "0"(out[i]), "v" (a), "v" (in[i]));
```

We insert the GCN isa into the kernel using `asm()` Assembler statement.
`volatile` keyword is used so that the optimizers must not change the number of volatile operations or change their order of execution relative to other volatile operations.
`v_mac_f32_e32` is the GCN instruction, for more information please refer - [AMD GCN3 ISA architecture manual](http://gpuopen.com/compute-product/amd-gcn3-isa-architecture-manual/)
Index for the respective operand in the ordered fashion is provided by `%` followed by position in the list of operands
`"v"` is the constraint code (for target-specific AMDGPU) for 32-bit VGPR register, for more info please refer - [Supported Constraint Code List for AMDGPU](https://llvm.org/docs/LangRef.html#supported-constraint-code-list)
Output Constraints are specified by an `"="` prefix as shown above ("=v"). This indicate that assemby will write to this operand, and the operand will then be made available as a return value of the asm expression. Input constraints do not have a prefix - just the constraint code. The constraint string of `"0"` says to use the assigned register for output as an input as well (it being the 0'th constraint).

Please note, there are usage limitations in ROCm compiler for inline asm support, please refer to [Inline ASM statements](https://rocm.docs.amd.com/projects/llvm-project/en/latest/reference/rocmcc.html#inline-asm-statements) for details.

## How to build and run:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```
- Execute the sample
```
$ ./inline_asm
Device name
hipMemcpyHostToDevice time taken  =  1.057ms
kernel Execution time             =  0.509ms
hipMemcpyDeviceToHost time taken  =  1.254ms
PASSED!
```

## More Info:
## More Info:
- [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/faq.html)
- [HIP Kernel Language](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/kernel_language.html)
- [HIP Runtime API (Doxygen)](https://rocm.docs.amd.com/projects/HIP/en/latest/doxygen/html/index.html)
- [HIP Porting Guide](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_porting_guide.html)
- [HIP Terminology](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/terms.html) (including comparing syntax for different compute terms across CUDA/HIP/OpenL)
- [HIPIFY](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/index.html)
- [Developer/CONTRIBUTING Info](https://github.com/ROCm/HIP/blob/develop/CONTRIBUTING.md)
