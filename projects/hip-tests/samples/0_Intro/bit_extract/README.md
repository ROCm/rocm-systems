# bit_extract

Show an application written directly in HIP which uses platform-specific check on __HIP_PLATFORM_AMD__ to enable use of
an instruction that only exists on the AMD platform.

See related [blog](http://gpuopen.com/platform-aware-coding-inside-hip/) demonstrating platform specialization.

- Steps to build this sample:
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute File
```
$ ./bit_extract

pch size: 11743288
__hipGetPCH succeeded!
info: running on device #0
info: allocate host mem (  7.63 MB)
info: allocate device mem (  7.63 MB)
info: copy Host2Device
info: launch 'bit_extract_kernel'
info: copy Device2Host
info: check result
PASSED!
```
