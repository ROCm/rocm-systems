# hipInfo

Simple tool that prints properties for each device (from hipGetDeviceProperties), and compiler info.
    Properties includes all of the architectural feature flags for each device.

Also demonstrates how to use platform-specific compilation path (testing `__HIP_PLATFORM_AMD__` or `__HIP_PLATFORM_NVIDIA__`)


- Steps to build this sample
```
$ mkdir build; cd build
$ cmake .. -DCMAKE_PREFIX_PATH=/opt/rocm
$ make
```

- Execute Code
```
$ ./hipInfo
--------------------------------------------------------------------------------
device#                           0
Name:
pciBusID:                         103
pciDeviceID:                      0
pciDomainID:                      0
multiProcessorCount:              64
maxThreadsPerMultiProcessor:      2560
isMultiGpuBoard:                  0
clockRate:                        1800 Mhz
memoryClockRate:                  1000 Mhz
memoryBusWidth:                   4096
totalGlobalMem:                   31.98 GB
totalConstMem:                    2147483647
sharedMemPerBlock:                64.00 KB
canMapHostMemory:                 1
regsPerBlock:                     65536
warpSize:                         64
l2CacheSize:                      8388608
computeMode:                      0
maxThreadsPerBlock:               1024
maxThreadsDim.x:                  1024
maxThreadsDim.y:                  1024
maxThreadsDim.z:                  1024
maxGridSize.x:                    2147483647
maxGridSize.y:                    65536
maxGridSize.z:                    65536
major:                            9
minor:                            0
concurrentKernels:                1
cooperativeLaunch:                1
cooperativeMultiDeviceLaunch:     1
isIntegrated:                     0
maxTexture1D:                     16384
maxTexture2D.width:               16384
maxTexture2D.height:              16384
maxTexture3D.width:               16384
maxTexture3D.height:              16384
maxTexture3D.depth:               8192
hostNativeAtomicSupported:        1
isLargeBar:                       1
asicRevision:                     1
maxSharedMemoryPerMultiProcessor: 64.00 KB
clockInstructionRate:             1000.00 Mhz
arch.hasGlobalInt32Atomics:       1
arch.hasGlobalFloatAtomicExch:    1
arch.hasSharedInt32Atomics:       1
arch.hasSharedFloatAtomicExch:    1
arch.hasFloatAtomicAdd:           1
arch.hasGlobalInt64Atomics:       1
arch.hasSharedInt64Atomics:       1
arch.hasDoubles:                  1
arch.hasWarpVote:                 1
arch.hasWarpBallot:               1
arch.hasWarpShuffle:              1
arch.hasFunnelShift:              0
arch.hasThreadFenceSystem:        1
arch.hasSyncThreadsExt:           0
arch.hasSurfaceFuncs:             0
arch.has3dGrid:                   1
arch.hasDynamicParallelism:       0
gcnArchName:                      gfx906:sramecc+:xnack-
peers:
non-peers:                        device#0
memInfo.total:                    31.98 GB
memInfo.free:                     31.96 GB (100%)
```
