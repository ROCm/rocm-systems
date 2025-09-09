# hipDispatchLatency.cpp

- Steps to build this sample
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute Code
```
$ ./hipDispatchEnqueueRateMT 1 0
Thread ID : 0 , hipModuleLaunchKernel enqueue rate: 0.8 us, std: 0.1 us

$ ./hipDispatchEnqueueRateMT 1 1
Thread ID : 0 , hipLaunchKernelGGL enqueue rate: 1.0 us, std: 0.1 us

$ ./hipDispatchLatency
hipModuleLaunchKernel enqueue rate: 0.8 us, std: 0.1 us

hipLaunchKernelGGL enqueue rate: 1.0 us, std: 0.1 us

Timing around single dispatch latency: 8.1 us, std: 4.7 us

Batch dispatch latency: 1.4 us, std: 0.0 us
```