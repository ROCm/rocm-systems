# Change Log for HIP

Full documentation for HIP is available at [rocm.docs.amd.com](https://rocm.docs.amd.com/projects/HIP/en/latest/index.html)

## HIP 7.2 for ROCm 7.2

### Added

* New HIP APIs
    - `hipLibraryLoadData`      creates library object from code
    - `hipLibraryLoadFromFile`  creates library object from file
    - `hipLibraryUnload`        unloads library
    - `hipLibraryGetKernel`     gets a kernel from library
    - `hipLibraryGetKernelCount` gets kernel count in library
    - `hipStreamCopyAttributes` copies attributes from source stream to destination stream

## HIP 7.1 for ROCm 7.1

### Added

* New HIP APIs
    - `hipModuleGetFunctionCount` returns the number of functions within a module
    - `hipMemsetD2D8` sets 2D memory range with specified 8-bit values
    - `hipMemsetD2D8Async` asynchronously sets 2D memory range with specified 8-bit values
    - `hipMemsetD2D16` sets 2D memory range with specified 16-bit values
    - `hipMemsetD2D16Async` asynchronously sets 2D memory range with specified 16-bit values
    - `hipMemsetD2D32` sets 2D memory range with specified 32-bit values
    - `hipMemsetD2D32Async` asynchronously sets 2D memory range with specified 32-bit values
    - `hipStreamSetAttribute` sets attributes such as synchronization policy for a given stream
    - `hipStreamGetAttribute` returns attributes such as priority for a given stream
    - `hipModuleLoadFatBinary`  loads fatbin binary to a module
    - `hipMemcpyBatchAsync` asynchronously performs a batch copy of 1D or 2D memory
    - `hipMemcpy3DBatchAsync` asynchronously performs a batch copy of 3D memory
    - `hipMemcpy3DPeer` copies memory between devices
    - `hipMemcpy3DPeerAsync` asynchronously copies memory between devices
    - `hipMemsetD2D32Async` asynchronously sets 2D memory range with specified 32-bit values
    - `hipMemPrefetchAsync_v2`  prefetches memory to the specified location
    - `hipMemAdvise_v2`         advises about the usage of a given memory range
    - `hipGetDriverEntryPoint ` gets function pointer of a HIP API.
    - `hipSetValidDevices`      sets a default list of devices that can be used by HIP
    - `hipStreamGetId`          queries the id of a stream
* Support for the flag `hipMemLocationTypeHost`, enables handling virtual memory management in host memory location, in addition to device memory.
* Support for nested tile partitioning within cooperative groups, matching NVIDIA CUDA functionality.

### Resolved issues

* A segmentation fault occurred in application when capturing the same HIP graph from multiple streams with cross-stream dependencies.  HIP runtime fixed an issue where a forked stream joined to a parent stream which was not originally created with the API `hipStreamBeginCapture`.
* Different behavior of en-queuing command on a legacy stream during stream capture on AMD ROCM platform, compared with NVIDIA CUDA. HIP runtime now returns an error in this specific situation, to behave the same as CUDA.
* Failure of memory access fault occurred in rocm-examples test suite. When Heterogeneous Memory Management (HMM) is not supported in the driver, `hipMallocManaged` will only allocate system memory in HIP runtime.

### Optimized

* Improved hip module loading latency.
* Optimized kernel metadata retrieval during module post load.
* Optimized doorbell ring in HIP runtime, advantages the following for performance improvement,
    - Makes efficient packet batching for HIP graph launch,
    - Dynamic packet copying based on defined maximum threshold or power-of-2 staggered copy pattern,
    - If timestamps are not collected for a signal for reuse, creates a new signal. This can potentially increase signal footprint if the handler doesn't run fast enough.

## HIP 7.0.2 for ROCm 7.0.2

### Added

* Support for the `hipMemAllocationTypeUncached` flag, enabling developers to allocate uncached memory. This flag is now supported in the following APIs:
    - `hipMemGetAllocationGranularity` determines the recommended allocation granularity for uncached memory.
    - `hipMemCreate` allocates memory with uncached properties.

### Resolved issues

* A compilation failure affecting applications that compile kernels using `hiprtc` with the compiler option `std=c++11`.
* A permission-related error occurred during the execution of hipLaunchHostFunc. This API is now supported and permitted to run during stream capture, aligning its behavior with CUDA.
* A numerical error during graph capture of kernels that rely on a remainder in `globalWorkSize`, in frameworks like MIOpen and PyTorch, where the grid size is not a multiple of the block size. To ensure correct replay behavior, HIP runtime now stores this remainder in `hip::GraphKernelNode` during `hipExtModuleLaunchKernel` capture, enabling accurate execution and preventing corruption.
* A page fault occurred during viewport rendering while running the file undo.blend in Blender. The issue was resolved by the HIP runtime, which reused the same context during image creation.
* Resolved a segmentation fault in `gpu_metrics`, which is used in threshold logic for command submission patches to GPU device(s) during CPU synchronization.

## HIP 7.0 for ROCm 7.0

### Added

* New HIP APIs
    - `hipLaunchKernelEx`  dispatches the provided kernel with the given launch configuration and forwards the kernel arguments.
    - `hipLaunchKernelExC`  launches a HIP kernel using a generic function pointer and the specified configuration.
    - `hipDrvLaunchKernelEx`  dispatches the device kernel represented by a HIP function object.
    - `hipMemGetHandleForAddressRange`  gets a handle for the address range requested.
    - `num_threads`  Total number of threads in the group. The legacy API size is alias.
* New support for Open Compute Project (OCP) floating-point `FP4`/`FP6`/`FP8` as the following. For details, see [Low precision floating point document](https://rocm.docs.amd.com/projects/HIP/en/latest/reference/low_fp_types.html).
    - Data types for `FP4`/`FP6`/`FP8`.
    - HIP APIs for `FP4`/`FP6`/`FP8`, which are compatible with corresponding CUDA APIs.
    - HIP Extensions APIs for microscaling formats, which are supported on AMD GPUs.
* New `wptr` and `rptr` values in `ClPrint`, for better logging in dispatch barrier methods.
* The `_sync()` version of crosslane builtins such as `shfl_sync()` are enabled by default. These can be disabled by setting the preprocessor macro `HIP_DISABLE_WARP_SYNC_BUILTINS`.
* Added `constexpr` operators for `fp16`/`bf16`.
* Added warp level primitives: `__syncwarp` and reduce intrinsics (e.g. `__reduce_add_sync()`)
* Support for the flags in APIs as following, now allows uncached memory allocation.
    - `hipExtHostRegisterUncached`, used in `hipHostRegister`.
    - `hipHostMallocUncached` and `hipHostAllocUncached`, used in `hipHostMalloc` and `hipHostAlloc`.
* `num_threads`  total number of threads in the group. The legacy API size is alias.
* Added PCI CHIP ID information as the device attribute.
* Added new tests applications for OCP data types `FP4`/`FP6`/`FP8`.
* A new attribute in HIP runtime was implemented which exposes a new device capability of how many compute dies (chiplets, xcc) are available on a given GPU. Developers can get this attribute via the API `hipDeviceGetAttribute`, to make use of the best cache locality in a kernel, and optimize the Kernel launch grid layout, for performance improvement.

### Changed
* Deprecated GPUs.
Some unsupported GPUs such as gfx9, gfx8 and gfx7 are deprecated on Microsoft Windows.
* Removal of Beta warnings in HIP Graph APIs
All Beta warnings in usage of HIP Graph APIs are removed, they are now officially and fully supported.
* Behavior changes
    - `hipGetLastError`  now returns the error code which is the last actual error caught in the current thread during the application execution.
    - Cooperative groups  in `hipLaunchCooperativeKernelMultiDevice` and `hipLaunchCooperativeKernel` functions, additional input parameter validation checks are added.
    - `hipPointerGetAttributes` returns `hipSuccess` instead of an error with invalid value `hipErrorInvalidValue`, in case `NULL` host or attribute pointer is passed as input parameter. It now matches the functionality of `cudaPointerGetAttributes` which changed with CUDA 11 and above releases.
    - `hipFree` previously there was an implicit wait which was applicable for all memory allocations, for synchronization purpose. This wait is now disabled for allocations made with `hipMallocAsync` and `hipMallocFromPoolAsync`, to match the behavior of CUDA API `cudaFree`
    - `hipFreeAsync` now returns `hipSuccess` when the input pointer is NULL, instead of ` hipErrorInvalidValue` , to be consistent with `hipFree`.
    - Exceptions occurring during a kernel execution will not abort the process anymore but will return an error unless core dump is enabled.
* Changes in hipRTC.
    - Removal of `hipRTC` symbols from HIP Runtime Library.
    Any application using `hipRTC` APIs should link explicitly with the `hipRTC` library. This makes the usage of `hipRTC` library on Linux the same as on Windows and matches the behavior of CUDA `nvRTC`.
    - `hipRTC` compilation
    The device code compilation now uses namespace `__hip_internal`, instead of the standard headers `std`, to avoid namespace collision.
    - Changes of datatypes from `hipRTC`.
    Datatype definitions such as `int64_t`, `uint64_t`, `int32_t`, and `uint32_t`, etc. are removed to avoid any potential conflicts in some applications. HIP now uses internal datatypes instead, prefixed with `__hip`, for example, `__hip_int64_t`.
* HIP header clean up
    - Usage of STD headers, HIP header files only include necessary STL headers.
    - Deprecated structure `HIP_MEMSET_NODE_PARAMS` is removed. Developers can use the definition `hipMemsetParams` instead.
* API signature/struct changes
    - API signatures are adjusted in some APIs to match corresponding CUDA APIs. Impacted APIs are as folloing:
      * `hiprtcCreateProgram`
      * `hiprtcCompileProgram`
      * `hipMemcpyHtoD`
      * `hipCtxGetApiVersion`
    - HIP struct change in `hipMemsetParams`, it is updated and compatible with CUDA.
    - HIP vector constructor change in `hipComplex` initialization now generates correct values. The affected constructors will be small vector types such as `float2`, `int4`, etc.
* Stream Capture updates
    - Restricted stream capture mode, it is made in HIP APIs via adding the macro `CHECK_STREAM_CAPTURE_SUPPORTED ()`.
In the previous HIP enumeration `hipStreamCaptureMode`, three capture modes were defined. With checking in the macro, the only supported stream capture mode is now `hipStreamCaptureModeRelaxed`. The rest are not supported, and the macro will return `hipErrorStreamCaptureUnsupported`. This update involves the following APIs, which is allowed only in relaxed stream capture mode,
      * `hipMallocManaged`
      * `hipMemAdvise`
    - Checks stream capture mode, the following APIs check the stream capture mode and return error codes to match the behavior of CUDA.
      * `hipLaunchCooperativeKernelMultiDevice`
      * `hipEventQuery`
      * `hipStreamAddCallback`
    - Returns error during stream capture. The following HIP APIs now returns specific error `hipErrorStreamCaptureUnsupported` on the AMD platform, but not always `hipSuccess`, to match behavior with CUDA.
      * `hipDeviceSetMemPool`
      * `hipMemPoolCreate`
      * `hipMemPoolDestroy`
      * `hipDeviceSetSharedMemConfig`
      * `hipDeviceSetCacheConfig`
      * `hipMemcpyWithStream`
* Error code update
Returned error/value codes are updated in the following HIP APIs to match the corresponding CUDA APIs.
    - Module Management Related APIs
      * `hipModuleLaunchKernel`
      * `hipExtModuleLaunchKernel`
      * `hipExtLaunchKernel`
      * `hipDrvLaunchKernelEx`
      * `hipLaunchKernel`
      * `hipLaunchKernelExC`
      * `hipModuleLaunchCooperativeKernel`
      * `hipModuleLoad`
    - Texture Management Related APIs
The following APIs update the return codes to match the behavior with CUDA:
      * `hipTexObjectCreate`, supports zero width and height for 2D image. If either is zero, will not return `false`.
      * `hipBindTexture2D`, adds extra check, if pointer for texture reference or device is NULL, returns `hipErrorNotFound`.
      * `hipBindTextureToArray`, if any NULL pointer is input for texture object, resource descriptor, or texture descriptor, returns error `hipErrorInvalidChannelDescriptor`, instead of `hipErrorInvalidValue`.
      * `hipGetTextureAlignmentOffset`, adds a return code `hipErrorInvalidTexture` when the texture reference pointer is NULL.
    - Cooperative Group Related APIs, more calidations are added in the following API implementation,
      * `hipLaunchCooperativeKernelMultiDevice`
      * `hipLaunchCooperativeKernel`
* Invalid stream input parameter handling
In order to match the CUDA runtime behavior more closely, HIP APIs with streams passed as input parameters no longer check the stream validity. Previously, the HIP runtime returned an error code `hipErrorContextIsDestroyed` if the stream was invalid. In CUDA version 12 and later, the equivalent behavior is to raise a segmentation fault. HIP runtime now matches the CUDA by causing a segmentation fault. The list of APIs impacted by this change are as follows:
    - Stream Management Related APIs
      * `hipStreamGetCaptureInfo`
      * `hipStreamGetPriority`
      * `hipStreamGetFlags`
      * `hipStreamDestroy`
      * `hipStreamAddCallback`
      * `hipStreamQuery`
      * `hipLaunchHostFunc`
    - Graph Management Related APIs
      * `hipGraphUpload`
      * `hipGraphLaunch`
      * `hipStreamBeginCaptureToGraph`
      * `hipStreamBeginCapture`
      * `hipStreamIsCapturing`
      * `hipStreamGetCaptureInfo`
      * `hipGraphInstantiateWithParams`
    - Memory Management Related APIs
      * `hipMemcpyPeerAsync`
      * `hipMemcpy2DValidateParams`
      * `hipMallocFromPoolAsync`
      * `hipFreeAsync`
      * `hipMallocAsync`
      * `hipMemcpyAsync`
      * `hipMemcpyToSymbolAsync`
      * `hipStreamAttachMemAsync`
      * `hipMemPrefetchAsync`
      * `hipDrvMemcpy3D`
      * `hipDrvMemcpy3DAsync`
      * `hipDrvMemcpy2DUnaligned`
      * `hipMemcpyParam2D`
      * `hipMemcpyParam2DAsync`
      * `hipMemcpy2DArrayToArray`
      * `hipMemcpy2D`
      * `hipMemcpy2DAsync`
      * `hipDrvMemcpy2DUnaligned`
      * `hipMemcpy3D`
    - Event Management Related APIs
      * `hipEventRecord`
      * `hipEventRecordWithFlags`
* `warpSize` Change
In order to match the CUDA specification, the `warpSize` variable is no longer `constexpr`. In general, this should be a transparent change; however, if an application was using `warpSize` as a compile-time constant, it will have to be updated to handle the new definition. For more information, see either the discussion of `warpSize` within the [HIP C++ language extensions](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warpsize).

### Optimized

HIP runtime has the following functional improvements which greatly improve runtime performance and user experience.

* Reduced usage of the lock scope in events and kernel handling.
    - Switches to `shared_mutex` for event validation, uses `std::unique_lock` in HIP runtime to create/destroy event, instead of `scopedLock`.
    - Reduces the `scopedLock` in handling of kernel execution. HIP runtime now calls `scopedLock` during kernel binary creation/initialization, doesn't call it again during kernel vector iteration before launch.
* Implementation of unifying managed buffer and kernel argument buffer so HIP runtime doesn't need to create/load a separate kernel argument buffer.
* Refactored memory validation, creates a unique function to validate a variety of memory copy operations.
* Improved kernel logging using demangling shader names.
* Advanced support for SPIRV, now kernel compilation caching is enabled by default. This feature is controlled by the environment variable `AMD_COMGR_CACHE`, for details, see [hip_rtc document](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_rtc.html).
* Programmatic support for scratch limits on MI300 and MI350 series up GPU devices. More enumeration values were added in `hipLimit_t` as following,
   - `hipExtLimitScratchMin`, minimum allowed value in bytes for scratch limit on the device.
   - `hipExtLimitScratchMax`, maximum allowed value in bytes for scratch limit on the device.
   - `hipExtLimitScratchCurrent`, current scratch limit threshold in bytes on the device. Must be between the value `hipExtLimitScratchMin` and `hipExtLimitScratchMax`.
 Developers can now use the environment variable `HSA_SCRATCH_SINGLE_LIMIT_ASYNC` to change the default allocation size with expected scratch limit in ROCR runtime. On top of it, this value can also be overwritten programmatically in the application using the HIP API `hipDeviceSetLimit(hipExtLimitScratchCurrent, value)` to reset the scratch limit value.
* HIP runtime now enables peer-to-peer (P2P) memory copies to utilize all available SDMA engines, rather than being limited to a single engine. It also selects the best engine first to give optimal bandwidth.
* Improved launch latency for `D2D` copies and `memset` on MI300 series.
* Introduced a threshold to handle the command submission patch to the GPU device(s), considering the synchronization with CPU, for performance improvement.

### Resolved issues

* Error of "unable to find modules" in HIP clean up for code object module.
* The issue of incorrect return error `hipErrorNoDevice`, when a crash occurred on GPU device due to illegal operation or memory violation. HIP runtime now handles the failure on the GPU side properly and reports the precise error code based on the last error seen on the GPU.
* Failures in some framework test applications, HIP runtime fixed the bug in retrieving a memory object from the IPC memory handle.
* A crash in TensorFlow related application. HIP runtime now combines multiple definitions of `callbackQueue` into a single function, in case of an exception, passes its handler to the application and provides corresponding error code.
* Fixed issue of handling the kernel parameters for the graph launch.
* Failures in roc-obj tools. HIP runtime now makes `DEPRECATED` message in roc-obj tools as `STDERR`.
* Support of `hipDeviceMallocContiguous` flags in `hipExtMallocWithFlags()`. It now enables `HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG` in the memory pool allocation on GPU device.
* Compilation failure, HIP runtime refactored the vector type alignment with `__hip_vec_align_v`
* A numerical error/corruption found in Pytorch  during graph replay. HIP runtime fixed the input sizes of kernel launch dimensions in hipExtModuleLaunchKernel for the execution of hipGraph capture.
* A crash during kernel execution in a customer application. The structure of kernel arguments was updated via adding the size of kernel arguments, and HIP runtime does validation before launch kernel with the structured arguments.
* Compilation error when using bfloat16 functions. HIP runtime removed the anonymous namespace from FP16 functions to resolve this issue.

## HIP 6.4.2 for ROCm 6.4.2

### Added

* HIP API implementation for `hipEventRecordWithFlags`, records an event in the specified stream with flags.
* Support for the pointer attribute `HIP_POINTER_ATTRIBUTE_CONTEXT`.
* Support for the flags `hipEventWaitDefault` and `hipEventWaitExternal`.

### Optimized

* Improved implementation in `hipEventSynchronize`, HIP runtime now makes internal callbacks as non-blocking operations to improve performance.

### Resolved issues

* Issue of dependency on `libgcc-s1` during rocm-dev install on Debian Buster. HIP runtime removed this Debian package dependency, and uses `libgcc1` instead for this distros.
* Building issue for `COMGR` dynamic load on Fedora and other Distros. HIP runtime now doesn't link against `libamd_comgr.so`.
* Failure in the API `hipStreamDestroy`, when stream type is `hipStreamLegacy`. The API now returns error code `hipErrorInvalidResourceHandle` on this condition.
* Kernel launch errors, such as `shared object initialization failed`, `invalid device function` or `kernel execution failure`. HIP runtime now loads `COMGR` properly considering the file with its name and mapped image.
* Memory access fault in some applications. HIP runtime fixed offset accumulation in memory address.
* The memory leak in virtual memory management (VMM). HIP runtime now uses the size of handle for allocated memory range instead of actual size for physical memory, which fixed the issue of address clash with VMM.
* Large memory allocation issue. HIP runtime now checks GPU video RAM and system RAM properly and sets size limits during memory allocation either on the host or the GPU device.
* Support of `hipDeviceMallocContiguous` flags in `hipExtMallocWithFlags()`. It now enables `HSA_AMD_MEMORY_POOL_CONTIGUOUS_FLAG` in the memory pool allocation on GPU device.
* Radom memory segmentation fault in handling `GraphExec` object release and `hipDeviceSyncronization`. HIP runtime now uses internal device synchronize function in `__hipUnregisterFatBinary`.

## HIP 6.4.1 for ROCm 6.4.1

### Added

* New log mask enumeration `LOG_COMGR` enables logging precise code object information.

### Changed

* HIP runtime uses device bitcode before SPIRV.
* The implementation of preventing `hipLaunchKernel` latency degradation with number of idle streams is reverted/disabled by default.
* Stop using `__AMDGCN_WAVEFRONT_SIZE` and `warpSize` as compile-time constants. The `warpSize` variable is no longer `constexpr`, in order to match the CUDA specification. See more details of the `warpSize` change within the ROCm 6.4.1 [deprecation notice](https://rocm.docs.amd.com/en/latest/about/release-notes.html#amdgpu-wavefront-size-compiler-macro-deprecation).

### Optimized

* Improved kernel logging includes de-mangling shader names.
* Refined implementation in HIP APIs `hipEventRecords` and `hipStreamWaitEvent` for performance improvement.

### Resolved issues

* Stale state during the graph capture. The return error was fixed, HIP runtime now always uses the latest dependent nodes during `hipEventRecord` capture.
* Segmentation fault during kernel execution. HIP runtime now allows maximum stack size as per ISA on the GPU device.

## HIP 6.4 (For ROCm 6.4)

### Added

* New HIP APIs
    - `hipDeviceGetTexture1DLinearMaxWidth`  returns the maximum width of elements in a 1D linear texture, that can be allocated on the specified device.
    - `hipStreamBatchMemOp`  enqueues an array of batch memory operations in the stream, for stream synchronization.
    - `hipGraphAddBatchMemOpNode`  creates a batch memory operation node and adds it to a graph.
    - `hipGraphBatchMemOpNodeGetParams`  returns the pointer of parameters from the batch memory operation node.
    - `hipGraphBatchMemOpNodeSetParams`  sets parameters for the batch memory operation node.
    - `hipGraphExecBatchMemOpNodeSetParams`  sets the parameters for a batch memory operation node in the given executable graph.
    - `hipLinkAddData` adds SPIRV code object data to linker instance with options.
    - `hipLinkAddFile` adds SPIRV code object file to linker instance with options.
    - `hipLinkCreate`  creates linker instance at runtime with options.
    - `hipLinkComplete` completes linking of program and output linker binary to use with hipModuleLoadData.
    - `hipLinkDestroy`  deletes linker instance.

### Changed

* roc-obj* tools are being deprecated, and will be removed in an upcoming release.
    - Perl package dependencies are now RECOMMENDS or SUGGESTS.  Users will need to install these themselves.
    - Support for ROCm Object tooling has moved into llvm-objdump provided by package rocm-llvm.
* SDMA retainer logic is removed for engine selection in operation of runtime buffer copy.

### Optimized

* `hipGraphLaunch` parallelism is improved for complex data-parallel graphs.
* Round-robin queue mechanism is updated for command scheduling. For multi-streams execution, HSA queue from null stream lock is freed and won't occupy the queue ID after the kernel in the stream is finished.
* The HIP runtime doesn't free bitcode object before code generation. It adds a cache, which allows compiled code objects to be reused instead of recompiling. This improves performance on multi-GPU systems.
* Runtime uses unified copy approach
    - Unpinned `H2D`copies are no longer blocking until the size of 1MB.
    - Kernel copy path is enabled for unpinned `H2D`/`D2H` methods.
    - The default environment variable `GPU_FORCE_BLIT_COPY_SIZE` is set to `16`, which limits the kernel copy to sizes less than 16 KB, while copies about that would be handled by `SDMA` engine.
    - Blit code is refactored and ASAN instrumentation is cleaned up.
* HIP runtime uses signals without interrupts.
    - In active wait mode, uses signals without interrupts by default.
    - Only when a callback is required, switches to the interrupts.

### Resolved issues

* Out of memory error on Windows. When the user calls `hipMalloc` for device memory allocation while specifying a size larger than the available device memory, the HIP runtime fixes the error in the API implementation, allocating the available device memory plus system memory (shared virtual memory).
* Error of dependency on libgcc-s1 during rocm-dev install on Debian Buster. HIP runtime now uses libgcc1 for this distros.
* Stack corruption during kernel execution. HIP runtime now adds maximum stack size limit based on the GPU device feature.

### Upcoming changes

The following are the list of backwards incompatible changes planned for the upcoming major ROCm release.

* Signature changes in APIs to match corresponding CUDA APIs,
    - `hiprtcCreateProgram`
    - `hiprtcCompileProgram`
    - `hipCtxGetApiVersion`
* Behavior of `hipPointerGetAttributes` is changed to match corresponding CUDA API in version 11 and later releases.
* Behavior of `hipFree` is changed to match corresponding CUDA API `cudaFree`.
* HIP vector constructor changes for `hipComplex`.
* Return error/value codes update in the following hip APIs, they now match the corresponding CUDA APIs,
    - `hipModuleLaunchKernel`
    - `hipExtModuleLaunchKernel`
    - `hipModuleLaunchCooperativeKernel`
    - `hipGetTextureAlignmentOffset`
    - `hipTexObjectCreate`
    - `hipBindTexture2D`
    - `hipBindTextureToArray`
    - `hipModuleLoad`
    - `hipLaunchCooperativeKernelMultiDevice`
    - `hipExtLaunchCooperativeKernelMultiDevice`

* HIPRTC implementation, the compilation of hiprtc now uses  namespace ` __hip_internal`, instead of the standard headers `std`.
* Stream capture mode update in the following hip APIs. Stream can only be captured in relax mode, to match the behavior of the corresponding CUDA APIs,
   - `hipMallocManaged`
   - `hipMemAdvise`
   - `hipLaunchCooperativeKernelMultiDevice`
   - `hipDeviceSetCacheConfig`
   - `hipDeviceSetSharedMemConfig`
   - `hipMemPoolCreate`
   - `hipMemPoolDestory`
   - `hipDeviceSetMemPool`
   - `hipEventQuery`
* The implementation of `hipStreamAddCallback` is updated, to match the behavior of CUDA.
* Removal of hiprtc symbols from hip library.
    - hiprtc will be a independent library, all symbols supported in hip library are removed.
    - Any application using hiprtc APIs should link explicitly with hiprtc library.
    - This change makes the usage of hiprtc library on Linux the same as on Windows, and matches the behavior of CUDA nvrtc.
* Removal of deprecated struct `HIP_MEMSET_NODE_PARAMS`, developers can use definition `hipMemsetParams` instead.
* `warpSize` change. Usages of `__AMDGCN_WAVEFRONT_SIZE` and `warpSize` as compile-time constants will be removed in HIP header files. In order to match the CUDA specification, the `warpSize` variable is no longer `constexpr`. If an application was using `warpSize` as a compile-time constant, it will have to be updated to handle the new definition. For details usage of the `warpSize`, see [the best practice for warpSize handling](https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_cpp_language_extensions.html#warpsize).

## HIP 6.3.2 for ROCm 6.3.2

### Added

* Tracking of Heterogeneous System Architecture (HSA) handlers:
    - Adds an atomic counter to track the outstanding HSA handlers.
    - Waits on CPU for the callbacks if the number exceeds the defined value.
* Codes to capture Architected Queueing Language (AQL) packets for HIP graph memory copy node between host and device. HIP enqueues AQL packets during graph launch.
* Control to use system pool implementation in runtime commands handling. By default, it is disabled.
* A new path to avoid `WaitAny` calls in `AsyncEventsLoop`. The new path is selected by default.
* Runtime control on decrement counter only if event is popped. There is a new way to restore dead signals cleanup for the old path.
* A new logic in runtime to track the age of events from the kernel mode driver.

### Optimized

* HSA callback performance. The HIP runtime creates and submits commands in the queue and interacts with HSA through a callback function. HIP waits for the CPU status from HSA to optimize handling of events, profiling, commands, and HSA signals for higher performance.
* Runtime optimisation which combines all logic of `WaitAny` in a single processing loop and avoids extra memory allocations or reference counting. The runtime won't spin on the CPU if all events are busy.
* Multi-threaded dispatches for performance improvement.
* Command submissions and processing between CPU and GPU by introducing a way to limit the software batch size.
* Switch to `std::shared_mutex` in book/keep logic in streams from multiple threads simultaneously, for performance improvement in specific customer applications.
* `std::shared_mutex` is used in memory object mapping, for performance improvement.

### Resolved issues

* Race condition in multi-threaded producer/consumer scenario with `hipMallocFromPoolAsync`.
* Segmentation fault with `hipStreamLegacy` while using the API `hipStreamWaitEvent`.
* Usage of `hipStreamLegacy` in HIP event record.
* A soft hang in graph execution process from HIP user object. The fix handles the release of graph execution object properly considering synchronization on the device/stream. The user application now behaves the same with  hipUserObject  on both the AMD ROCm and NVIDIA CUDA platforms.


## HIP 6.3.1 for ROCm 6.3.1

### Added

* An activeQueues set that tracks only the queues that have a command submitted to them, which allows fast iteration in `waitActiveStreams`.

### Optimized

* Mechanism of preventing `hipLaunchKernel` latency degradation with number of idle streams is implemented for performance improvement.

## HIP 6.3 for ROCm 6.3

### Added

* New HIP APIs
    - `hipGraphExecGetFlags`  returns the flags on executable graph.
    - `hipGraphNodeSetParams`  updates parameters of a created node.
    - `hipGraphExecNodeSetParams`  updates parameters of a created node on executable graph.
    - `hipDrvGraphMemcpyNodeGetParams`  gets a memcpy node's parameters.
    - `hipDrvGraphMemcpyNodeSetParams`  sets a memcpy node's parameters.
    - `hipDrvGraphAddMemFreeNode`  creates a memory free node and adds it to a graph.
    - `hipDrvGraphExecMemcpyNodeSetParams`  sets the parameters for a memcpy node in the given graphExec.
    - `hipDrvGraphExecMemsetNodeSetParams`  sets the parameters for a memset node in the given graphExec.

### Changed

* Un-deprecated HIP APIs
    - `hipHostAlloc`
    - `hipFreeHost`

### Optimized

* Disabled CPU wait in device synchronize to avoid idle time in applications such as Hugging Face models and PyTorch.
* Optimized multi-threaded dispatches to improve performance.
* Limited the software batch size to control the number of command submissions for runtime to handle efficiently.
* Optimizes HSA callback performance when a large number of events are recorded by multiple threads and submitted to multiple GPUs.
* HIP graph execution perfomance improvement.
    - Added the optimized multistream path in graph execution. It uses a fixed number of async streams in the execution
    - Optimized the launch latency, where commands creation and execution is done at the same time
    - Optimized the scheduling to use less barriers and waiting signals if the same queue  can be detected
    - The new path is controlled by a new environment variable, with the options either to use the original path, or to force the number of asynchronous queues for execution.

### Resolved issues

* Soft hang in runtime wait event when run TensorFlow.
* Memory leak in the API `hipGraphInstantiate` when kernel is launched using `hipExtLaunchKernelGGL` with event.
* Memory leak when the API `hipGraphAddMemAllocNode` is called.
* The `_sync()` version of crosslane builtins such as `shfl_sync()`,
  `__all_sync()` and `__any_sync()`, continue to be hidden behind the
  preprocessor macro `HIP_ENABLE_WARP_SYNC_BUILTINS`, and will be enabled
  unconditionally in the next ROCm release.


## HIP 6.2.41134 for ROCm 6.2.1

### Resolved issues

* Soft hang when use AMD_SERIALIZE_KERNEL.
* Memory leak in hipIpcCloseMemHandle.


## HIP 6.2 (For ROCm 6.2)

### Added
- Introduced the `_sync()` version of crosslane builtins such as `shfl_sync()`, `__all_sync()`
  and `__any_sync()`. These take a 64-bit integer as an explicit mask argument.
  - In HIP 6.2, these are hidden behind the preprocessor macro
    `HIP_ENABLE_WARP_SYNC_BUILTINS`, and will be enabled unconditionally in HIP 6.3.
- Added new HIP APIs
    - `hipGetProcAddress` returns the pointer to driver function, corresponding to the defined driver function symbol.
    - `hipGetFuncBySymbol` returns the pointer to device entry function that matches entry function symbolPtr.
    - `hipStreamBeginCaptureToGraph` begins graph capture on a stream to an existing graph.
    - `hipGraphInstantiateWithParams`  creates an executable graph from a graph.
    - `hipMemcpyAtoA`  copies from one 1D array to another.
    - `hipMemcpyDtoA`  copies from device memory to a 1D array.
    - `hipMemcpyAtoD`  copies from one 1D array to device memory.
    - `hipMemcpyAtoHAsync`  copies from one 1D array to host memory.
    - `hipMemcpyHtoAAsync`  copies from host memory to a 1D array.
    - `hipMemcpy2DArrayToArray`  copies data between host and device.

- Added a new flag `integrated` support in device property

    The `integrated` flag is added in the struct `hipDeviceProp_t`.
    On the integrated `APU` system, the runtime driver detects and sets this flag to `1`, in which case the API `hipDeviceGetAttribute` returns enum `hipDeviceAttribute_t` for hipDeviceAttributeIntegrated as value `1`, for integrated GPU device.

    The enum value `hipDeviceAttributeIntegrated` corresponds to `cudaDevAttrIntegrated` on CUDA platform.
- Added initial support for 8-bit floating point datatype in `amd_hip_fp8.h`. These are accessible via `#include <hip/hip_fp8.h>`
- Add UUID support for environment variable `HIP_VISIBLE_DEVICES`.

### Resolved issues
- Stream capture support in HIP graph.
Prohibited and unhandled operations are fixed during stream capture in HIP runtime.
- Fix undefined symbol error for hipTexRefGetArray & hipTexRefGetBorderColor.

## HIP 6.1 (For ROCm 6.1)

### Added
- New environment variable HIP_LAUNCH_BLOCKING
It is used for serialization on kernel execution.
The default value is 0 (disable), kernel will execute normally as defined in the queue. When this environment variable is set as 1 (enable), HIP runtime will serialize kernel enqueue, behaves the same as AMD_SERIALIZE_KERNEL.
- Added HIPRTC support for hip headers driver_types, math_functions, library_types, math_functions, hip_math_constants, channel_descriptor, device_functions, hip_complex, surface_types, texture_types.

### Changed
- HIPRTC now assumes WGP mode for gfx10+. CU mode can be enabled by passing `-mcumode` to the compile options from `hiprtcCompileProgram`.

### Resolved issues
- HIP complex vector type multiplication and division operations.
On AMD platform, some duplicated complex operators are removed to avoid compilation failures.
In HIP, hipFloatComplex and hipDoubleComplex are defined as complex data types,
typedef float2 hipFloatComplex;
typedef double2 hipDoubleComplex;
Any application uses complex multiplication and division operations, need to replace '*' and '/' operators with the following,
    - hipCmulf() and hipCdivf() for hipFloatComplex
    - hipCmul() and hipCdiv() for hipDoubleComplex

    Note: These complex operations are equivalent to corresponding types/functions on NVIDIA platform.

## HIP 6.0 (For ROCm 6.0)

### Added
- Addition of hipExtGetLastError
  - AMD backend specific API, to return error code from last HIP API called from the active host thread

- New fields for external resource interoperability,
  - Structs
    - hipExternalMemoryHandleDesc_st
    - hipExternalMemoryBufferDesc_st
    - hipExternalSemaphoreHandleDesc_st
    - hipExternalSemaphoreSignalParams_st
    - hipExternalSemaphoreWaitParams_st
  - Enumerations
    - hipExternalMemoryHandleType_enum
    - hipExternalSemaphoreHandleType_enum
    - hipExternalMemoryHandleType_enum

- New members are added in HIP struct hipDeviceProp_t, for new feature capabilities including,
  - Texture
     - int maxTexture1DMipmap;
     - int maxTexture2DMipmap[2];
     - int maxTexture2DLinear[3];
     - int maxTexture2DGather[2];
     - int maxTexture3DAlt[3];
     - int maxTextureCubemap;
     - int maxTexture1DLayered[2];
     - int maxTexture2DLayered[3];
     - int maxTextureCubemapLayered[2];
  - Surface
     - int maxSurface1D;
     - int maxSurface2D[2];
     - int maxSurface3D[3];
     - int maxSurface1DLayered[2];
     - int maxSurface2DLayered[3];
     - int maxSurfaceCubemap;
     - int maxSurfaceCubemapLayered[2];
  - Device
     - hipUUID uuid;
     - char luid[8];
       -- this is 8-byte unique identifier. Only valid on windows
       -- LUID (Locally Unique Identifier) is supported for interoperability between devices.
     - unsigned int luidDeviceNodeMask; \

     Note: HIP supports LUID only on Windows OS.
- Added `amd_hip_bf16.h` which adds `bfloat16` type. These definitions are accessible via `#include <hip/hip_bf16.h>`
This header exists alongside the older bfloat16 header in`amd_hip_bfloat16.h` which is included via `hip/hip_bfloat16.h`. Users are recommended to use `<hip/hip_bf16.h>` instead of `<hip/hip_bfloat16.h>`.

### Changed
- Some OpenGL Interop HIP APIs are moved from the hip_runtime_api header to a new header file hip_gl_interop.h for the AMD platform, as following,
    - hipGLGetDevices
    - hipGraphicsGLRegisterBuffer
    - hipGraphicsGLRegisterImage
- With ROCm 6.0, the HIP version is 6.0. As the HIP runtime binary suffix is updated in every major ROCm release, in ROCm 6.0, the new filename is libamdhip64.so.6. Furthermore, in ROCm 6.0 release, the libamdhip64.so.5 binary from ROCm 5.7 is made available to maintain binary backward compatibility with ROCm 5.x.

### Changed Impacting Backward Compatibility
- Data types for members in HIP_MEMCPY3D structure are changed from "unsigned int" to "size_t".
- The value of the flag hipIpcMemLazyEnablePeerAccess is changed to “0x01”, which was previously defined as “0”.
- Some device property attributes are not currently support in HIP runtime, in order to maintain consistency, the following related enumeration names are changed in hipDeviceAttribute_t
    - hipDeviceAttributeName is changed to hipDeviceAttributeUnused1
    - hipDeviceAttributeUuid is changed to hipDeviceAttributeUnused2
    - hipDeviceAttributeArch is changed to hipDeviceAttributeUnused3
    - hipDeviceAttributeGcnArch is changed to hipDeviceAttributeUnused4
    - hipDeviceAttributeGcnArchName is changed to hipDeviceAttributeUnused5
- HIP struct hipArray is removed from driver type header to be complying with cuda
- hipArray_t replaces hipArray*, as the pointer to array.
    - This allows hipMemcpyAtoH and hipMemcpyHtoA to have the correct array type which is equivalent to coresponding CUDA driver APIs.

### Removed
- Deprecated Heterogeneous Compute (HCC) symbols and flags are removed from the HIP source code, including,
    - Build options on obsolete HCC_OPTIONS was removed from cmake.
    - Micro definitions are removed.
      HIP_INCLUDE_HIP_HCC_DETAIL_DRIVER_TYPES_H
      HIP_INCLUDE_HIP_HCC_DETAIL_HOST_DEFINES_H
    - Compilation flags for the platform definitions,
      AMD platform,
      __HIP_PLATFORM_HCC__
      __HCC__
      __HIP_ROCclr__
      NVIDIA platform,
      __HIP_PLATFORM_NVCC__
- File directories in the clr repository are removed,
  https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/hcc_detail
  https://github.com/ROCm/clr/blob/develop/hipamd/include/hip/nvcc_detail
- Deprecated gcnArch is removed from hip device struct hipDeviceProp_t.
- Deprecated "enum hipMemoryType memoryType;" is removed from HIP struct hipPointerAttribute_t union.
- Deprecated HIT based tests are removed from HIP project
- Catch tests are available [hip-tests] (https://github.com/ROCm/hip-tests) project

### Resolved issues
- Kernel launch maximum dimension validation is added specifically on gridY and gridZ in the HIP API hipModule-LaunchKernel. As a result,when hipGetDeviceAttribute is called for the value of hipDeviceAttributeMaxGrid-Dim, the behavior on the AMD platform is equivalent to NVIDIA.
- The HIP stream synchronisation behavior is changed in internal stream functions, in which a flag "wait" is added and set when the current stream is null pointer while executing stream synchronisation on other explicitly created streams. This change avoids blocking of execution on null/default stream.
The change won't affect usage of applications, and makes them behave the same on the AMD platform as NVIDIA.
- Error handling behavior on unsupported GPU is fixed, HIP runtime will log out error message, instead of creating signal abortion error which is invisible to developers but continued kernel execution process. This is for the case when developers compile any application via hipcc, setting the option --offload-arch with GPU ID which is different from the one on the system.

### Known Issues
- Dynamically loaded HIP runtime library references incorrect version of hipDeviceGetProperties and hipChooseDevice APIs

When an application dynamically loads the HIP runtime library from ROCm 6.0 and attempts to get the hipDeviceGetProperties and/or hipChooseDevice entry-points using dlsym, the application gets the older version (ROCm 5.7) of those entry-points.

As a workaround, while compiling with ROCm 6.0, use the string "hipDeviceGetPropertiesR0600", and "hipChooseDeviceR0600" respectively for hipDeviceGetProperties and hipChooseDevice APIs.

## HIP 5.7.1 (For ROCm 5.7.1)

### Resolved issues
- hipPointerGetAttributes API returns the correct HIP memory type as hipMemoryTypeManaged for managed memory.

## HIP 5.7 (For ROCm 5.7)

### Added
- Added meta_group_size/rank for getting the number of tiles and rank of a tile in the partition
- Added new APIs supporting Windows only, under development on Linux

    - hipMallocMipmappedArray for allocating a mipmapped array on the device

    - hipFreeMipmappedArray for freeing a mipmapped array on the device

    - hipGetMipmappedArrayLevel for getting a mipmap level of a HIP mipmapped array

    - hipMipmappedArrayCreate for creating a mipmapped array

    - hipMipmappedArrayDestroy for destroy a mipmapped array

    - hipMipmappedArrayGetLevel for getting a mipmapped array on a mipmapped level

### Known Issues
- HIP memory type enum values currently don't support equivalent value to cudaMemoryTypeUnregistered, due to HIP functionality backward compatibility.
- HIP API hipPointerGetAttributes could return invalid value in case the input memory pointer was not allocated through any HIP API on device or host.

### Upcoming changes
- Removal of gcnarch from hipDeviceProp_t structure
- Addition of new fields in hipDeviceProp_t structure
  - maxTexture1D
  - maxTexture2D
  - maxTexture1DLayered
  - maxTexture2DLayered
  - sharedMemPerMultiprocessor
  - deviceOverlap
  - asyncEngineCount
  - surfaceAlignment
  - unifiedAddressing
  - computePreemptionSupported
  - hostRegisterSupported
  - uuid
- Removal of deprecated code
  -hip-hcc codes from hip code tree
- Correct hipArray usage in HIP APIs such as hipMemcpyAtoH and hipMemcpyHtoA
- HIPMEMCPY_3D fields correction to avoid truncation of "size_t" to "unsigned int" inside hipMemcpy3D()
- Renaming of 'memoryType' in hipPointerAttribute_t structure to 'type'
- Correct hipGetLastError to return the last error instead of last API call's return code
- Update hipExternalSemaphoreHandleDesc to add "unsigned int reserved[16]"
- Correct handling of flag values in hipIpcOpenMemHandle for hipIpcMemLazyEnablePeerAccess
- Remove hiparray* and make it opaque with hipArray_t

## HIP 5.6.1 (For ROCm 5.6.1)

### Resolved issues
- Enabled xnack+ check in HIP catch2 tests hang while tests execution
- Memory leak when code object files are loaded/unloaded via hipModuleLoad/hipModuleUnload APIs
- Resolved an issue of crash while using hipGraphAddMemFreeNode

## HIP 5.6 (For ROCm 5.6)

### Added
- Added hipRTC support for amd_hip_fp16
- Added hipStreamGetDevice implementation to get the device assocaited with the stream
- Added HIP_AD_FORMAT_SIGNED_INT16 in hipArray formats
- hipArrayGetInfo for getting information about the specified array
- hipArrayGetDescriptor for getting 1D or 2D array descriptor
- hipArray3DGetDescriptor to get 3D array descriptor

### Changed
- hipMallocAsync to return success for zero size allocation to match hipMalloc
- Separation of hipcc perl binaries from HIP project to hipcc project. hip-devel package depends on newly added hipcc package
- Consolidation of hipamd, ROCclr, and OpenCL repositories into a single repository called clr. Instructions are updated to build HIP from sources in the HIP Installation guide
- Removed hipBusBandwidth and hipCommander samples from hip-tests

### Optimized
- Consolidation of hipamd, rocclr and OpenCL projects in clr
- Optimized lock for graph global capture mode

### Resolved issues
- Fixed regression in hipMemCpyParam3D when offset is applied

### Known Issues
- Limited testing on xnack+ configuration
  - Multiple HIP tests failures (gpuvm fault or hangs)
- hipSetDevice and hipSetDeviceFlags APIs return hipErrorInvalidDevice instead of hipErrorNoDevice, on a system without GPU
- Known memory leak when code object files are loaded/unloaded via hipModuleLoad/hipModuleUnload APIs. Issue will be fixed in future release

### Upcoming changes
- Removal of gcnarch from hipDeviceProp_t structure
- Addition of new fields in hipDeviceProp_t structure
  - maxTexture1D
  - maxTexture2D
  - maxTexture1DLayered
  - maxTexture2DLayered
  - sharedMemPerMultiprocessor
  - deviceOverlap
  - asyncEngineCount
  - surfaceAlignment
  - unifiedAddressing
  - computePreemptionSupported
  - hostRegisterSupported
  - uuid
- Removal of deprecated code
  -hip-hcc codes from HIP code tree
- Correct hipArray usage in HIP APIs such as hipMemcpyAtoH and hipMemcpyHtoA
- HIPMEMCPY_3D fields correction to avoid truncation of "size_t" to "unsigned int" inside hipMemcpy3D()
- Renaming of 'memoryType' in hipPointerAttribute_t structure to 'type'
- Correct hipGetLastError to return the last error instead of last API call's return code
- Update hipExternalSemaphoreHandleDesc to add "unsigned int reserved[16]"
- Correct handling of flag values in hipIpcOpenMemHandle for hipIpcMemLazyEnablePeerAccess
- Remove hiparray* and make it opaque with hipArray_t
