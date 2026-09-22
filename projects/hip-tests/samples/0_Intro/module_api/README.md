# module_api

- Steps to build this sample

```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute Code
```
$ ./launchKernelHcc.hip.out
PASSED!
$ ./runKernel.hip.out
PASSED!
$ ./defaultDriver.hip.out
PASSED!
```