ROCM_PATH is the path where ROCM is installed. default path is /opt/rocm.
# Compile to LLVM IR and create an executable from modified IR

This sample shows how to generate the LLVM IR for a simple HIP source application, then re-compiling it and generating a valid HIP executable.

This sample uses a previous HIP application sample, please see [0_Intro/square](https://github.com/ROCm/hip-tests/tree/develop/samples/0_Intro/square).

## Compiling the HIP source into LLVM IR
Using HIP flags `-c -emit-llvm` will help generate the host x86_64 and the device LLVM bitcode when paired with `--cuda-host-only` and `--cuda-device-only` respectively. In this sample we use these commands:
```
<ROCM_PATH>/hip/bin/hipcc -c -emit-llvm --cuda-host-only -target x86_64-linux-gnu -o square_host.bc square.cpp
<ROCM_PATH>/hip/bin/hipcc -c -emit-llvm --cuda-device-only --offload-arch=gfx900 --offload-arch=gfx906  --offload-arch=gfx908 --offload-arch=gfx1010 --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1101 --offload-arch=gfx1102 --offload-arch=gfx1103 square.cpp
```
The device LLVM IR bitcode will be output into two separate files:
- square-hip-amdgcn-amd-amdhsa-gfx900.bc
- square-hip-amdgcn-amd-amdhsa-gfx906.bc
- square-hip-amdgcn-amd-amdhsa-gfx908.bc
- square-hip-amdgcn-amd-amdhsa-gfx1010.bc
- square-hip-amdgcn-amd-amdhsa-gfx1030.bc
- square-hip-amdgcn-amd-amdhsa-gfx1100.bc
- square-hip-amdgcn-amd-amdhsa-gfx1101.bc
- square-hip-amdgcn-amd-amdhsa-gfx1102.bc
- square-hip-amdgcn-amd-amdhsa-gfx1103.bc

You may modify `--offload-arch` flag to build other archs and choose to enable or disable xnack and sram-ecc.

To transform the LLVM bitcode into human readable LLVM IR, use these commands:
```
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx900.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx906.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx908.bc -o square-hip-amdgcn-amd-amdhsa-gfx908.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1010.bc -o square-hip-amdgcn-amd-amdhsa-gfx1010.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1030.bc -o square-hip-amdgcn-amd-amdhsa-gfx1030.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1100.bc -o square-hip-amdgcn-amd-amdhsa-gfx1100.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1101.bc -o square-hip-amdgcn-amd-amdhsa-gfx1101.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1102.bc -o square-hip-amdgcn-amd-amdhsa-gfx1102.ll
<ROCM_PATH>/llvm/bin/llvm-dis square-hip-amdgcn-amd-amdhsa-gfx1103.bc -o square-hip-amdgcn-amd-amdhsa-gfx1103.ll
```

**Warning:** We cannot ensure any compiler besides the ROCm hipcc and clang will be compatible with this process. Also, there is no guarantee that the starting IR produced with `-x cl` will run with HIP runtime. Experimenting with other compilers or starting IR will be the responsibility of the developer.

## Modifying the LLVM IR
***Warning: The LLVM Language Specification may change across LLVM major releases, therefore the user must make sure the modified LLVM IR conforms to the LLVM Language Specification corresponding to the used LLVM version.***

At this point, you may evaluate the LLVM IR and make modifications if you are familiar with the LLVM IR language. Since the LLVM IR can vary between compiler versions, the safest approach would be to use the same compiler to consume the IR as the compiler producing it. It is the responsibility of the developer to ensure the IR is valid when manually modifying it.

## Compiling the LLVM IR into a valid HIP executable
If valid, the modified host and device IR may be compiled into a HIP executable. First, the readable IR must be compiled back in LLVM bitcode. The host IR can be compiled into an object using this command:
```
<ROCM_PATH>/llvm/bin/llvm-as square_host.ll -o square_host.bc
<ROCM_PATH>/hip/bin/hipcc -c square_host.bc -o square_host.o
```

However, the device IR will require a few extra steps. The device bitcodes needs to be compiled into device objects, then offload-bundled into a HIP fat binary using the clang-offload-bundler, then llvm-mc embeds the binary inside of a host object using the MC directives provided in `hip_obj_gen.mcin`. The output is a host object with an embedded device object. Here are the steps for device side compilation into an object:
```
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx900.ll -o square-hip-amdgcn-amd-amdhsa-gfx900.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx906.ll -o square-hip-amdgcn-amd-amdhsa-gfx906.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx908.ll -o square-hip-amdgcn-amd-amdhsa-gfx908.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1010.ll -o square-hip-amdgcn-amd-amdhsa-gfx1010.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1030.ll -o square-hip-amdgcn-amd-amdhsa-gfx1030.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1100.ll -o square-hip-amdgcn-amd-amdhsa-gfx1100.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1101.ll -o square-hip-amdgcn-amd-amdhsa-gfx1101.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1102.ll -o square-hip-amdgcn-amd-amdhsa-gfx1102.bc
<ROCM_PATH>/hip/../llvm/bin/llvm-as square-hip-amdgcn-amd-amdhsa-gfx1103.ll -o square-hip-amdgcn-amd-amdhsa-gfx1103.bc
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx900 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx900.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx906 square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx906.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx908 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx908.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1010 square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx1010.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1030 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx1030.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1100 square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx1100.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1101 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx1101.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1102 square-hip-amdgcn-amd-amdhsa-gfx906.bc -o square-hip-amdgcn-amd-amdhsa-gfx1102.o
<ROCM_PATH>/hip/../llvm/bin/clang -target amdgcn-amd-amdhsa -mcpu=gfx1103 square-hip-amdgcn-amd-amdhsa-gfx900.bc -o square-hip-amdgcn-amd-amdhsa-gfx1103.o
<ROCM_PATH>/hip/../llvm/bin/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-amdgcn-amd-amdhsa-gfx900,hip-amdgcn-amd-amdhsa-gfx906,hip-amdgcn-amd-amdhsa-gfx908,hip-amdgcn-amd-amdhsa-gfx1010,hip-amdgcn-amd-amdhsa-gfx1030,hip-amdgcn-amd-amdhsa-gfx1100,hip-amdgcn-amd-amdhsa-gfx1101,hip-amdgcn-amd-amdhsa-gfx1102,hip-amdgcn-amd-amdhsa-gfx1103 -inputs=/dev/null,square-hip-amdgcn-amd-amdhsa-gfx900.o,square-hip-amdgcn-amd-amdhsa-gfx906.o,square-hip-amdgcn-amd-amdhsa-gfx908.o,square-hip-amdgcn-amd-amdhsa-gfx1010.o,square-hip-amdgcn-amd-amdhsa-gfx1030.o,square-hip-amdgcn-amd-amdhsa-gfx1100.o,square-hip-amdgcn-amd-amdhsa-gfx1101.o,square-hip-amdgcn-amd-amdhsa-gfx1102.o,square-hip-amdgcn-amd-amdhsa-gfx1103.o -outputs=offload_bundle.hipfb
<ROCM_PATH>/llvm/bin/llvm-mc hip_obj_gen.mcin -o square_device.o --filetype=obj
```

**Note:** Using option `-bundle-align=4096` only works on ROCm 4.0 and newer compilers. Also, the architecture must match the same arch as when compiling to LLVM IR.

Finally, using the system linker, hipcc, or clang, link the host and device objects into an executable:
```
<ROCM_PATH>/hip/bin/hipcc square_host.o square_device.o -o square_ir.out
```
If you haven't modified the GPU archs, this executable should run on the defined `gfx900`, `gfx906`, `gfx908`, `gfx1010`, `gfx1030`, `gfx1100`, `gfx1101`, `gfx1102` and `gfx1103`.

## How to build and run this sample:
- Build the sample using cmake
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute sample
```
$ ./square_ir.out
info: running on device AMD Radeon Graphics
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'vector_square' kernel
info: copy Device2Host
info: check result
PASSED!
```

**Note:** Any undefined arch can be modified with make argument `GPU_ARCHxx`.

## For More Information, please refer to [HIP FAQ](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/faq.html).
