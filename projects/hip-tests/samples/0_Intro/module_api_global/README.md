# module_api_global

- Steps to build this sample
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute Code
```
$ ./runKernel1.hip.out
PASSED!
Shared Size Bytes = 0
Num Regs = 3
PASSED!
```