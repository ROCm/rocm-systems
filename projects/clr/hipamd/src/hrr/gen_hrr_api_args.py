#!/usr/bin/env python3
"""
gen_hrr_api_args.py — Generate hrr_api_args.h, hip_capture_generated.cpp,
                      and hip_playback_generated.cpp from hip_api_trace.hpp.

Usage:
    python gen_hrr_api_args.py [--input HIP_API_TRACE_HPP]
                               [--output-header HRR_API_ARGS_H]
                               [--output-capture HIP_CAPTURE_GENERATED_CPP]
                               [--output-playback HIP_PLAYBACK_GENERATED_CPP]

Defaults:
    input            : ../../../hipamd/include/hip/amd_detail/hip_api_trace.hpp
    output-header    : ./hrr_api_args.h
    output-capture   : ./hip_capture_generated.cpp
    output-playback  : ./playback/hip_playback_generated.cpp

hrr_api_args.h
--------------
One packed C struct per HIP API (both runtime and compiler dispatch tables).
All pointer/handle fields become uint64_t so the struct layout is identical on
all platforms and in the binary archive.

Struct field rules:
  * Every struct starts with:
        uint64_t thread_id;    // OS thread that made the call
        uint64_t sequence_id;  // monotonically increasing per capture session
  * Return value stored as int32_t ret (hipError_t is int, void returns skip this).
  * Pointer parameters  -> uint64_t  (GPU address or host pointer as integer)
  * Handle parameters   -> uint64_t  (hipStream_t, hipEvent_t, hipModule_t, ...)
  * dim3 parameters     -> three uint32_t fields  (e.g. gridDim_x/y/z)
  * uint3 parameters    -> three uint32_t fields
  * Scalar int/uint/... -> kept as-is (int32_t / uint32_t / ...)
  * size_t              -> uint64_t
  * float/double        -> kept as-is
  * Enum types          -> int32_t

Special extra fields appended to certain structs (after normal params):
  __hipRegisterFatBinary:
      uint64_t blob_hash_lo;  // FNV-1a-128 lo  (raw data* replaced by hash)
      uint64_t blob_hash_hi;
      uint64_t blob_size;
  hipModuleLoadData / hipModuleLoadDataEx:
      uint64_t co_hash_lo;    // code object hash
      uint64_t co_hash_hi;
      uint32_t module_id;     // sequential module handle ID
  hipMemcpy* (H2D-capable) — 1D sync/async variants:
      uint64_t blob_hash_lo;  // hash of host src data (0 if not H2D)
      uint64_t blob_hash_hi;

hip_capture_generated.cpp
--------------------------
Contains:
  * capture_hipFoo() shim for every API not in MANUAL_CAPTURE_APIS
  * hip_capture_build_table()  — installs all runtime shims
  * hip_capture_build_compiler_table() — installs all compiler shims
  * get_thread_id() helper (platform-specific)

MANUAL_CAPTURE_APIS are implemented by hand in hip_capture.cpp because they need
complex serialization (kernel launches, memcpy blob capture, module load).
The generated shims for those are simple pass-throughs — they do NOT call
write_event(); the hand-written shims do.

hip_playback_generated.cpp
---------------------------
Contains:
  * playback_hipFoo() for every HIP API
  * hrr_playback_dispatch[HRR_API_COUNT]  — indexed by hrr_api_id_t
"""

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple


# ---------------------------------------------------------------------------
# APIs handled by hand in hip_capture.cpp — genuinely complex serialization
# ---------------------------------------------------------------------------

# These APIs get FULL hand-written shims in hip_capture.cpp that call write_event.
# The generator produces pass-through stubs for them.
MANUAL_CAPTURE_APIS: Set[str] = {
    # Kernel launches — kernel arg introspection, variable-length payload
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    # <<<>>> launch config — must save grid/block/shared/stream into TLS for hipLaunchByPtr
    "__hipPushCallConfiguration",
    # Module load — code object snapshotting to disk
    "hipModuleLoadData",
    "hipModuleLoadDataEx",
    "hipModuleLoad",
    # Memcpy H2D variants — blob snapshotting of host src data to disk
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemcpyHtoD",
    "hipMemcpyHtoDAsync",
    "hipMemcpyWithStream",
    # Memcpy D2H variants — blob capture of host dst data after copy
    "hipMemcpyDtoH",
    "hipMemcpyDtoHAsync",
    # Fat binary registration — blob snapshotting
    "__hipRegisterFatBinary",
    # Fat binary unregistration — must record *modules before the real call
    "__hipUnregisterFatBinary",
    # Host memory registration — blob snapshotting of initial host mem contents
    "hipHostRegister",
    "hipHostUnregister",
    # hipMemPoolCreate — pool_props is a struct pointer; must copy it inline
    "hipMemPoolCreate",
    # hipMemPoolSetAttribute — value is void*; copy 8 bytes inline as value_u64
    "hipMemPoolSetAttribute",
    # hipMemcpy3D family — hipMemcpy3DParms is a struct pointer; copy inline + H2D blob
    "hipMemcpy3D",
    "hipMemcpy3DAsync",
    "hipMemcpy3D_spt",
    "hipMemcpy3DAsync_spt",
    # Array creation — need handle map (manual capture for output handle)
    "hipArrayCreate",
    "hipArray3DCreate",
    # Struct-pointer capture: value stored inline
    "hipStreamSetAttribute",
    "hipMemGetAllocationGranularity",
    "hipMemPoolSetAccess",
    "hipMemSetAccess",
}

# Alias for backward compat within the file (some helpers used MANUAL_APIS)
MANUAL_APIS = MANUAL_CAPTURE_APIS

# APIs that are pass-through even for the manual path
# (hipModuleGetFunction: function handles identified by name at launch time)
PASSTHROUGH_ONLY: Set[str] = {
    "hipModuleGetFunction",
}

# ---------------------------------------------------------------------------
# Playback manual APIs — implemented by hand in hip_playback.cpp
# ---------------------------------------------------------------------------

MANUAL_PLAYBACK_APIS: Set[str] = {
    # Kernel launches — variable-length binary payload, kernarg buffer + function lookup
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    # Memcpy H2D — must load blob from disk using hash fields appended to struct
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemcpyHtoD",
    "hipMemcpyHtoDAsync",
    "hipMemcpyWithStream",
    # Module load — must load code object from archive by hash, not raw image ptr
    "hipModuleLoadData",
    "hipModuleLoadDataEx",
    "hipModuleLoad",
    # Function lookup — resolved by name at kernel launch; no handle map needed
    "hipModuleGetFunction",
    # Alloc — need ctx.record_alloc / ctx.remove_alloc (not encodable by generator)
    "hipMalloc",
    "hipMallocAsync",
    "hipMallocFromPoolAsync",
    "hipMallocManaged",
    "hipFree",
    "hipFreeAsync",
    # Stream lifecycle — need ctx.record_stream / ctx.remove_stream
    "hipStreamCreate",
    "hipStreamCreateWithFlags",
    "hipStreamCreateWithPriority",
    "hipStreamDestroy",
    # Event lifecycle — need ctx.record_event / ctx.remove_event
    "hipEventCreate",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    # Fat binary registration — load blob as module so kernel names resolve
    "__hipRegisterFatBinary",
    # Host memory registration — need handle map + blob restore + device ptr recording
    "hipHostRegister",
    "hipHostUnregister",
    "hipHostGetDevicePointer",
    # hipMemPoolCreate — pool_props stored inline; must reconstruct & pass by pointer
    "hipMemPoolCreate",
    # hipMemPoolSetAttribute / hipMemPoolGetAttribute — value is void*; stored inline
    "hipMemPoolSetAttribute",
    "hipMemPoolGetAttribute",
    # Graph stream-capture flow — hipStreamEndCapture output handle must be recorded,
    # hipGraphInstantiate must use the recorded graph handle and record exec handle.
    # hipStreamBeginCapture also handled manually for debug / stream-not-found safety.
    # hipGraphLaunch needs rd64/rd32 offset-correct reads (reinterpret_cast is broken).
    "hipStreamBeginCapture",
    "hipStreamEndCapture",
    "hipGraphInstantiate",
    "hipGraphLaunch",
    # DtoH driver-style copies — dst is a host pointer (not in alloc_map); need temp buffer
    "hipMemcpyDtoH",
    "hipMemcpyDtoHAsync",
    # hipMemcpy3D family — parms struct stored inline (parms_bytes); must reconstruct
    "hipMemcpy3D",
    "hipMemcpy3DAsync",
    "hipMemcpy3D_spt",
    "hipMemcpy3DAsync_spt",
    # Array creation — handle must be recorded in ctx.array_map
    "hipArrayCreate",
    "hipArray3DCreate",
    # hipFreeArray — skip if handle not in array_map (e.g. created by nooped hipMallocArray)
    "hipFreeArray",
    # Struct-pointer params stored inline
    "hipStreamSetAttribute",
    "hipMemGetAllocationGranularity",
    "hipMemPoolSetAccess",
    "hipMemSetAccess",
    # VMM (Virtual Memory Management) — output handles / VAs must be tracked
    "hipMemAddressReserve",
    "hipMemAddressFree",
    "hipMemCreate",
    "hipMemRelease",
    "hipMemMap",
    "hipMemUnmap",
}

# ---------------------------------------------------------------------------
# APIs that get an inline no-op playback body (return hipSuccess; immediately)
# rather than an extern declaration.  Used for:
#   Category 1: not present in ROCm SDK 6.4 headers
#   Category 3: hipDeviceptr_t output param type mismatch
#   Category 4: hipGraphNode_t* array param type mismatch
#   Category 5: other output-handle type mismatches
#   Category 6: misc wrong-return-type or missing struct fields
# ---------------------------------------------------------------------------
NOOP_PLAYBACK_APIS: Set[str] = {
    # Category 1: APIs not present in ROCm SDK 6.4 headers (C3861/LNK2019)
    "hipExtHostAlloc",
    "hipGLGetDevices",
    "hipGraphicsGLRegisterBuffer",
    "hipGraphicsGLRegisterImage",
    "hipHccModuleLaunchKernel",
    "hipOccupancyMaxActiveClusters",
    "hipOccupancyMaxPotentialClusterSize",
    "hipPointerSetAttribute",
    "hipMemGetHandleForAddressRange",
    # Category 3: hipDeviceptr_t output params — generator emits wrong cast
    "hipMemAllocPitch",
    "hipMemGetAddressRange",
    "hipModuleGetGlobal",
    "hipTexRefGetAddress",
    "hipTexRefGetFormat",
    "hipTexRefSetFormat",
    "hipMemMapArrayAsync",
    "hipMipmappedArrayGetMemoryRequirements",
    # Category 4: hipGraphNode_t* array params — generator passes void** but API needs hipGraphNode_t*
    "hipGraphAddChildGraphNode",
    "hipGraphAddDependencies",
    "hipGraphAddEmptyNode",
    "hipGraphAddEventRecordNode",
    "hipGraphAddEventWaitNode",
    "hipGraphAddExternalSemaphoresSignalNode",
    "hipGraphAddExternalSemaphoresWaitNode",
    "hipGraphAddHostNode",
    "hipGraphAddKernelNode",
    "hipGraphAddMemAllocNode",
    "hipGraphAddMemFreeNode",
    "hipGraphAddMemcpyNode",
    "hipGraphAddMemcpyNode1D",
    "hipGraphAddMemcpyNodeFromSymbol",
    "hipGraphAddMemcpyNodeToSymbol",
    "hipGraphAddMemsetNode",
    "hipGraphAddNode",
    "hipGraphAddBatchMemOpNode",
    "hipGraphExecUpdate",
    "hipGraphGetEdges",
    "hipGraphGetNodes",
    "hipGraphGetRootNodes",
    "hipGraphNodeFindInClone",
    "hipGraphNodeGetDependencies",
    "hipGraphNodeGetDependentNodes",
    "hipGraphRemoveDependencies",
    "hipStreamBeginCaptureToGraph",
    "hipStreamGetCaptureInfo_v2",
    "hipStreamGetCaptureInfo_v2_spt",
    "hipStreamUpdateCaptureDependencies",
    "hipDrvGraphAddMemFreeNode",
    "hipDrvGraphAddMemcpyNode",
    "hipDrvGraphAddMemsetNode",
    # Category 5: Other type mismatches (output handle params stored as void** by generator)
    "hipCtxCreate",
    "hipCtxGetCurrent",
    "hipCtxGetDevice",
    "hipCtxPopCurrent",
    "hipDeviceGet",
    "hipDevicePrimaryCtxRetain",
    "hipDrvGetErrorName",
    "hipDrvGetErrorString",
    "hipGetTextureReference",
    "hipKernelGetName",
    "hipMemImportFromShareableHandle",
    "hipMemRetainAllocationHandle",
    # hipMemGetAllocationPropertiesFromHandle — handle is stale at playback; query not needed
    "hipMemGetAllocationPropertiesFromHandle",
    "hipModuleGetTexRef",
    "hipStreamGetDevice",
    "hipUserObjectCreate",
    # Category 6: Misc — struct field issues, wrong return type casts, or missing types
    # hipDeviceGetByPCIBusId takes a char* PCI string (stale capture-time pointer) — noop
    "hipDeviceGetByPCIBusId",
    # hipDeviceGetName / hipDeviceGetPCIBusId write into a caller-sized char buffer.
    # The generator emits `char _out{}` (1 byte) causing a stack-smash on Linux.
    # These are query-only calls with no effect on replay correctness — noop.
    "hipDeviceGetName",
    "hipDeviceGetPCIBusId",
    # hipChooseDevice takes a hipDeviceProp_t* (stale capture-time pointer) — noop
    "hipChooseDevice",
    # hipDeviceGetGraphMemAttribute / hipDeviceSetGraphMemAttribute — void* value ptr (stale) — noop
    "hipDeviceGetGraphMemAttribute",
    "hipDeviceSetGraphMemAttribute",
    # hipDevicePrimaryCtxGetState — output pointers (stale) — noop
    "hipDevicePrimaryCtxGetState",
    # hipGetSymbolAddress / hipGetSymbolSize — symbol name is a stale host ptr — noop
    "hipGetSymbolAddress",
    "hipGetSymbolSize",
    # hipPointerGetAttribute (singular) — void* output (stale) — noop
    "hipPointerGetAttribute",
    # hipDrvPointerGetAttributes — void** array of outputs (stale) — noop
    "hipDrvPointerGetAttributes",
    # hipMemPoolGetAccess — hipMemLocation* (stale) — noop
    "hipMemPoolGetAccess",
    # hipMemPoolExportPointer / hipMemPoolImportPointer — handle data struct (stale) — noop
    "hipMemPoolExportPointer",
    "hipMemPoolImportPointer",
    # hipMallocArray / hipMalloc3DArray — hipChannelFormatDesc* (stale); output handle not needed for D2H — noop
    "hipMallocArray",
    "hipMalloc3DArray",
    # hipMalloc3D — pitchedDevPtr output (stale), hipExtent non-castable; output not needed for D2H — noop
    "hipMalloc3D",
    # hipMemAllocHost / hipMallocHost / hipHostAlloc / hipFreeHost — host ptr alloc/free; not device allocations — noop
    "hipMemAllocHost",
    "hipMallocHost",
    "hipHostAlloc",
    "hipFreeHost",
    # hipMemAllocPitch — hipDeviceptr_t* output (type mismatch) + output not in alloc_map — noop
    "hipMemAllocPitch",
    # hipHostGetFlags — output flag ptr (stale) — noop
    "hipHostGetFlags",
    # _spt memcpy variants with host ptrs: no blob fields; noop since H2D data handled by hipMemcpy blobs
    "hipMemcpy_spt",
    "hipMemcpyAsync_spt",
    "hipMemcpy2D_spt",
    "hipMemcpy2DAsync_spt",
    "hipMemcpy2DFromArray_spt",
    "hipMemcpy2DFromArrayAsync_spt",
    "hipMemcpy2DToArray_spt",
    "hipMemcpy2DToArrayAsync_spt",
    "hipMemcpyFromSymbol_spt",
    "hipMemcpyFromSymbolAsync_spt",
    "hipMemcpyToSymbol_spt",
    "hipMemcpyToSymbolAsync_spt",
    "hipMemcpyFromArray_spt",
    # hipMemcpyToArray / hipMemcpyFromArray / hipMemcpy2DToArray / hipMemcpy2DFromArray — array handle (not in array_map for generated shim) — noop
    "hipMemcpyToArray",
    "hipMemcpyFromArray",
    "hipMemcpy2DToArray",
    "hipMemcpy2DFromArray",
    "hipMemcpy2DToArrayAsync",
    "hipMemcpy2DFromArrayAsync",
    # hipMemcpyAtoH / hipMemcpyHtoA — hipArray_t not in array_map for generated shim — noop
    "hipMemcpyAtoH",
    "hipMemcpyHtoA",
    # hipMemcpyParam2D / hipMemcpyParam2DAsync — hip_Memcpy2D struct* (stale) — noop
    "hipMemcpyParam2D",
    "hipMemcpyParam2DAsync",
    # hipMemcpyToSymbol / hipMemcpyFromSymbol / Async — symbol name is host ptr (stale) — noop
    "hipMemcpyToSymbol",
    "hipMemcpyFromSymbol",
    "hipMemcpyToSymbolAsync",
    "hipMemcpyFromSymbolAsync",
    # hipMemset3D / hipMemset3DAsync — hipPitchedPtr is non-castable (stale) — noop
    "hipMemset3D",
    "hipMemset3DAsync",
    "hipMemset3D_spt",
    "hipMemset3DAsync_spt",
    # hipMemset_spt / 2D_spt variants — host ptr for dst (stale) or pitched ptr — noop
    "hipMemset_spt",
    "hipMemset2D_spt",
    "hipMemsetAsync_spt",
    "hipMemset2DAsync_spt",
    # hipMemsetD2D8/16/32 and Async — hipDeviceptr_t output correctly typed; but not in alloc_map — noop
    "hipMemsetD2D8",
    "hipMemsetD2D8Async",
    "hipMemsetD2D16",
    "hipMemsetD2D16Async",
    "hipMemsetD2D32",
    "hipMemsetD2D32Async",
    # hipMallocPitch — already in MANUAL_PLAYBACK_APIS for the DrvMemcpy test; these are the _spt wrappers
    # hipLaunchCooperativeKernel — variable args (void**); handled via regular kernel launch fallback — noop
    "hipLaunchCooperativeKernel",
    # hipOccupancyMaxActiveBlocksPerMultiprocessor / WithFlags — kernel fn ptr (stale) — noop
    # Note: already in NOOP via func pointer category above; just confirming
    "hipExtMallocWithFlags",
    "hipChooseDeviceR0000",
    "hipGetDevicePropertiesR0000",
    "hipGetErrorName",
    "hipGetErrorString",
    "hipCreateChannelDesc",
    # APIs that return const char* (not hipError_t) — generator emits wrong cast
    "hipApiName",
    "hipKernelNameRef",
    "hipKernelNameRefByPtr",
    # Category 7: APIs that take a host function pointer — meaningless at playback
    "hipOccupancyMaxPotentialBlockSize",
    # hipOccupancyMaxPotentialBlockSizeWithFlags removed from dispatch table in ROCm 6.4
    "hipOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "hipFuncSetCacheConfig",
    "hipFuncSetSharedMemConfig",
    "hipFuncGetAttributes",
    "hipFuncSetAttribute",
    # Category 8: Device extra — output ptr stale, dangerous context ops, or struct-ptr params
    # hipDeviceGetUuid — output hipUUID* stale
    "hipDeviceGetUuid",
    # hipDeviceGetTexture1DLinearMaxWidth — hipChannelFormatDesc* param (stale struct ptr)
    "hipDeviceGetTexture1DLinearMaxWidth",
    # Primary ctx ops that would destroy the context at playback
    "hipDevicePrimaryCtxRelease",
    "hipDevicePrimaryCtxReset",
    "hipDevicePrimaryCtxSetFlags",
    # Category 9: Stream advanced — stale void* device ptrs, stale stream handles
    # hipExtStreamCreateWithCUMask — output handle not tracked in stream_map by generator
    "hipExtStreamCreateWithCUMask",
    # hipExtStreamGetCUMask — uint32_t* array output (stale)
    "hipExtStreamGetCUMask",
    # hipExtGetLinkTypeAndHopCount — output uint32_t* ptrs stale
    "hipExtGetLinkTypeAndHopCount",
    # StreamWait/Write Value — void* ptr is stale device address (no alloc_map translation in generated shim)
    "hipStreamWaitValue32",
    "hipStreamWaitValue64",
    "hipStreamWriteValue32",
    "hipStreamWriteValue64",
    # hipStreamAttachMemAsync — void* dev_ptr stale
    "hipStreamAttachMemAsync",
    # hipGetStreamDeviceId — returns int not hipError_t (wrong return type cast)
    "hipGetStreamDeviceId",
    # Category 10: Context APIs — stale hipCtx_t handles or stale output pointers
    "hipCtxGetFlags",
    "hipCtxGetCacheConfig",
    "hipCtxGetSharedMemConfig",
    "hipCtxGetApiVersion",
    "hipCtxSetCurrent",
    "hipCtxEnablePeerAccess",
    "hipCtxDisablePeerAccess",
    # Category 11: Module/Library/Kernel — stale handles or unsupported output types
    # hipModuleUnload — would destroy module needed for kernel replay
    "hipModuleUnload",
    # hipModuleGetFunctionCount — output uint* stale
    "hipModuleGetFunctionCount",
    # hipModuleLoadFatBinary — fat binary ptr stale at playback (code object already loaded via registered binary)
    "hipModuleLoadFatBinary",
    # hipModule occupancy — hipFunction_t stale handle
    "hipModuleOccupancyMaxPotentialBlockSize",
    "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    # hipFuncGetAttribute — hipFunction_t stale handle + output int* stale
    "hipFuncGetAttribute",
    # hipGetFuncBySymbol — symbolPtr stale + output hipFunction_t* stale
    "hipGetFuncBySymbol",
    # hipExtLaunchKernel — function_address void* stale + void** args array stale
    "hipExtLaunchKernel",
    # Library APIs — hipLibrary_t handles not tracked at playback
    "hipLibraryLoadData",
    "hipLibraryLoadFromFile",
    "hipLibraryUnload",
    "hipLibraryGetKernel",
    "hipLibraryGetKernelCount",
    "hipLibraryEnumerateKernels",
    "hipKernelGetLibrary",
    "hipKernelGetFunction",
    "hipKernelGetParamInfo",
    "hipKernelGetAttribute",
    "hipKernelSetAttribute",
    # Category 12: Misc — stale output pointers or wrong return type
    "hipGetProcAddress",
    "hipGetDriverEntryPoint",
    "hipGetDriverEntryPoint_spt",
    # hipOccupancyAvailableDynamicSMemPerBlock — hipFunction_t stale handle
    "hipOccupancyAvailableDynamicSMemPerBlock",
    # hipSetValidDevices — device array ptr stale at playback
    "hipSetValidDevices",
    # hipMemPtrGetInfo — void* output size ptr stale
    "hipMemPtrGetInfo",
    # hipMemGetMemPool — output hipMemPool_t* (type mismatch) + handle not tracked
    "hipMemGetMemPool",
    # hipMemGetAccess — hipMemLocation* (stale struct ptr)
    "hipMemGetAccess",
    # hipMemRangeGetAttribute — segfaults on Linux ROCm 7.13; stale dev_ptr unsafe
    "hipMemRangeGetAttribute",
    # hipMemRangeGetAttributes — attribute arrays stale
    "hipMemRangeGetAttributes",
    # hipMemDiscardBatchAsync — void** dev_ptrs is a stale device address array (no alloc_map translation in generated shim)
    "hipMemDiscardBatchAsync",
    # hipDrvMemDiscardBatchAsync — hipDeviceptr_t* dptrs output param; generator emits wrong cast
    "hipDrvMemDiscardBatchAsync",
    # hipMemDiscardAndPrefetchBatchAsync — void** dptrs stale device addresses + hipMemLocation* prefetchLocs (stale struct ptr)
    "hipMemDiscardAndPrefetchBatchAsync",
    # hipDrvMemDiscardAndPrefetchBatchAsync — hipDeviceptr_t* dptrs output param; generator emits wrong cast
    "hipDrvMemDiscardAndPrefetchBatchAsync",
    # Category 13: Driver 3D/2D memcpy — HIP_MEMCPY3D* / hipMemcpy3DPeerParms* / hip_Memcpy2D* stale struct ptrs
    "hipDrvMemcpy3D",
    "hipDrvMemcpy3DAsync",
    "hipDrvMemcpy2DUnaligned",
    "hipMemcpy3DPeer",
    "hipMemcpy3DPeerAsync",
    "hipMemcpy2DArrayToArray",
    # AtoD/DtoA/AtoA — hipArray_t handles not in array_map for generated shim
    "hipMemcpyAtoD",
    "hipMemcpyDtoA",
    "hipMemcpyAtoA",
    "hipMemcpyAtoHAsync",
    "hipMemcpyHtoAAsync",
    # Category 14: Texture / Array query APIs — stale handles + non-castable struct output params
    "hipArrayGetDescriptor",
    "hipArray3DGetDescriptor",
    "hipArrayGetInfo",
    "hipArrayDestroy",
    "hipGetChannelDesc",
    "hipCreateTextureObject",
    "hipDestroyTextureObject",
    "hipTexObjectCreate",
    "hipTexObjectDestroy",
    "hipMallocMipmappedArray",
    "hipMipmappedArrayCreate",
    "hipMipmappedArrayDestroy",
    "hipMipmappedArrayGetLevel",
    "hipGetMipmappedArrayLevel",
    "hipFreeMipmappedArray",
    # Category 15: Graph explicit APIs — stale hipGraph_t/hipGraphExec_t/hipGraphNode_t handles
    "hipGraphCreate",
    "hipGraphDestroy",
    "hipGraphClone",
    "hipGraphUpload",
    "hipGraphDebugDotPrint",
    "hipGraphNodeGetType",
    "hipGraphNodeSetEnabled",
    "hipGraphNodeGetEnabled",
    "hipGraphKernelNodeGetParams",
    "hipGraphKernelNodeSetParams",
    "hipGraphKernelNodeCopyAttributes",
    "hipGraphKernelNodeGetAttribute",
    "hipGraphKernelNodeSetAttribute",
    "hipGraphMemcpyNodeGetParams",
    "hipGraphMemcpyNodeSetParams",
    "hipGraphMemcpyNodeSetParams1D",
    "hipGraphMemsetNodeGetParams",
    "hipGraphMemsetNodeSetParams",
    "hipGraphHostNodeGetParams",
    "hipGraphHostNodeSetParams",
    "hipGraphExecDestroy",
    "hipGraphExecKernelNodeSetParams",
    "hipGraphExecMemcpyNodeSetParams",
    "hipGraphExecMemcpyNodeSetParams1D",
    "hipGraphExecMemsetNodeSetParams",
    "hipGraphExecHostNodeSetParams",
    "hipGraphExecChildGraphNodeSetParams",
    "hipGraphExecGetFlags",
    "hipGraphNodeSetParams",
    "hipGraphExecNodeSetParams",
    "hipGraphInstantiateWithParams",
    "hipGraphChildGraphNodeGetGraph",
    "hipGraphEventRecordNodeGetEvent",
    "hipGraphEventRecordNodeSetEvent",
    "hipGraphEventWaitNodeGetEvent",
    "hipGraphEventWaitNodeSetEvent",
    "hipGraphExecEventRecordNodeSetEvent",
    "hipGraphExecEventWaitNodeSetEvent",
    "hipGraphMemAllocNodeGetParams",
    "hipGraphMemFreeNodeGetParams",
    "hipUserObjectRetain",
    "hipUserObjectRelease",
    "hipGraphRetainUserObject",
    "hipGraphReleaseUserObject",
    "hipGraphBatchMemOpNodeGetParams",
    "hipGraphBatchMemOpNodeSetParams",
    "hipGraphExecBatchMemOpNodeSetParams",
    "hipDrvGraphExecMemcpyNodeSetParams",
    "hipDrvGraphExecMemsetNodeSetParams",
    "hipDrvGraphMemcpyNodeGetParams",
    "hipDrvGraphMemcpyNodeSetParams",
}

# ---------------------------------------------------------------------------
# Extra struct fields appended to certain APIs
# ---------------------------------------------------------------------------

# Maps API name -> list of (field_type, field_name, comment) to append
EXTRA_FIELDS: Dict[str, List[Tuple[str, str, str]]] = {
    "__hipRegisterFatBinary": [
        ("uint64_t", "blob_hash_lo", "FNV-1a-128 lo of fat binary"),
        ("uint64_t", "blob_hash_hi", "FNV-1a-128 hi"),
        ("uint64_t", "blob_size",    "fat binary byte count"),
    ],
    "hipModuleLoadData": [
        ("uint64_t", "co_hash_lo", "code object hash lo"),
        ("uint64_t", "co_hash_hi", "code object hash hi"),
        ("uint32_t", "module_id",  "sequential module handle ID"),
    ],
    "hipModuleLoadDataEx": [
        ("uint64_t", "co_hash_lo", "code object hash lo"),
        ("uint64_t", "co_hash_hi", "code object hash hi"),
        ("uint32_t", "module_id",  "sequential module handle ID"),
    ],
    "hipModuleLoad": [
        ("uint64_t", "co_hash_lo", "code object hash lo"),
        ("uint64_t", "co_hash_hi", "code object hash hi"),
        ("uint32_t", "module_id",  "sequential module handle ID"),
    ],
    # hipHostRegister — snapshot of host memory at registration time
    "hipHostRegister":    [("uint64_t", "blob_hash_lo", "sysmem blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "sysmem blob hash hi")],
    # Memcpy 1D variants — blob hash for H2D data
    "hipMemcpy":          [("uint64_t", "blob_hash_lo", "H2D blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "H2D blob hash hi")],
    "hipMemcpyAsync":     [("uint64_t", "blob_hash_lo", "H2D blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "H2D blob hash hi")],
    "hipMemcpyHtoD":      [("uint64_t", "blob_hash_lo", "blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "blob hash hi")],
    "hipMemcpyDtoH":      [("uint64_t", "blob_hash_lo", "D2H expected-output blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "D2H expected-output blob hash hi")],
    "hipMemcpyDtoD":      [("uint64_t", "blob_hash_lo", "zero (D2D)"),
                           ("uint64_t", "blob_hash_hi", "zero (D2D)")],
    "hipMemcpyHtoDAsync": [("uint64_t", "blob_hash_lo", "blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "blob hash hi")],
    "hipMemcpyDtoHAsync": [("uint64_t", "blob_hash_lo", "D2H expected-output blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "D2H expected-output blob hash hi")],
    "hipMemcpyDtoDAsync": [("uint64_t", "blob_hash_lo", "zero (D2D)"),
                           ("uint64_t", "blob_hash_hi", "zero (D2D)")],
    # hipMemcpyWithStream — same semantics as hipMemcpyAsync (H2D blob, D2H/D2D zero)
    "hipMemcpyWithStream": [("uint64_t", "blob_hash_lo", "H2D blob hash lo"),
                            ("uint64_t", "blob_hash_hi", "H2D blob hash hi")],
    # hipMemPoolCreate — pool_props is a struct pointer; store it inline by value.
    # sizeof(hipMemPoolProps) == 88 bytes (4+4+8+8+8+56 reserved).
    "hipMemPoolCreate": [("uint8_t", "pool_props_bytes[88]", "hipMemPoolProps inline copy")],
    # hipMemPoolSetAttribute / hipMemPoolGetAttribute — value is void* to a scalar
    # (always uint64_t or uint32_t depending on attr); store 8 bytes inline.
    "hipMemPoolSetAttribute": [("uint64_t", "value_u64", "attribute value stored inline")],
    "hipMemPoolGetAttribute": [("uint64_t", "value_u64", "attribute value stored inline (unused at capture)")],
    # hipMemcpy3D family — hipMemcpy3DParms is 160 bytes; store inline + H2D blob hash.
    "hipMemcpy3D":          [("uint8_t", "parms_bytes[160]", "hipMemcpy3DParms inline copy"),
                             ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                             ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                             ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                             ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipMemcpy3DAsync":     [("uint8_t", "parms_bytes[160]", "hipMemcpy3DParms inline copy"),
                             ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                             ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                             ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                             ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipMemcpy3D_spt":      [("uint8_t", "parms_bytes[160]", "hipMemcpy3DParms inline copy"),
                             ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                             ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                             ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                             ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipMemcpy3DAsync_spt": [("uint8_t", "parms_bytes[160]", "hipMemcpy3DParms inline copy"),
                             ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                             ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                             ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                             ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    # hipArrayCreate — HIP_ARRAY_DESCRIPTOR is 24 bytes; store inline.
    "hipArrayCreate":   [("uint8_t", "array_desc_bytes[24]", "HIP_ARRAY_DESCRIPTOR inline copy")],
    # hipArray3DCreate — HIP_ARRAY3D_DESCRIPTOR is 40 bytes; store inline.
    "hipArray3DCreate": [("uint8_t", "array3d_desc_bytes[40]", "HIP_ARRAY3D_DESCRIPTOR inline copy")],
    # hipStreamSetAttribute — hipStreamAttrValue is 64 bytes; store inline.
    "hipStreamSetAttribute": [("uint8_t", "stream_attr_bytes[64]", "hipStreamAttrValue inline copy")],
    # hipMemGetAllocationGranularity — hipMemAllocationProp is 32 bytes; store inline.
    "hipMemGetAllocationGranularity": [("uint8_t", "alloc_prop_bytes[32]", "hipMemAllocationProp inline copy")],
    # hipMemPoolSetAccess / hipMemSetAccess — hipMemAccessDesc is 12 bytes; store first entry inline.
    "hipMemPoolSetAccess": [("uint8_t", "access_desc_bytes[12]", "hipMemAccessDesc[0] inline copy")],
    "hipMemSetAccess":     [("uint8_t", "access_desc_bytes[12]", "hipMemAccessDesc[0] inline copy")],
}

# Maps the base name of a uint8_t inline-array field (e.g. "pool_props_bytes")
# to the C struct type whose sizeof must match the declared array extent.
# Used to emit static_assert(sizeof(CType) == N) immediately after each struct
# that contains such a field, catching size mismatches at compile time.
_INLINE_STRUCT_ASSERTS: Dict[str, str] = {
    "pool_props_bytes":   "hipMemPoolProps",
    "parms_bytes":        "hipMemcpy3DParms",
    "array_desc_bytes":   "HIP_ARRAY_DESCRIPTOR",
    "array3d_desc_bytes": "HIP_ARRAY3D_DESCRIPTOR",
    "stream_attr_bytes":  "hipStreamAttrValue",
    "alloc_prop_bytes":   "hipMemAllocationProp",
    "access_desc_bytes":  "hipMemAccessDesc",
}


# ---------------------------------------------------------------------------
# Type normalisation
# ---------------------------------------------------------------------------

# Handle / opaque types -> uint64_t
_HANDLE_TYPES = {
    "hipStream_t", "hipEvent_t", "hipModule_t", "hipFunction_t",
    "hipCtx_t", "hipDevice_t", "hipDeviceptr_t",
    "hipArray_t", "hipArray_const_t", "hipMipmappedArray_t",
    "hipMipmappedArray_const_t", "hipSurfaceObject_t", "hipTextureObject_t",
    "hipMemPool_t", "hipGraph_t", "hipGraphNode_t", "hipGraphExec_t",
    "hipUserObject_t", "hipMemGenericAllocationHandle_t",
    "hipExternalMemory_t", "hipExternalSemaphore_t",
    "hipKernel_t", "hipLibrary_t",
    # HIP 7.14+ green context / device resource handles
    "hipExecutionCtx_t", "hipDevResourceDesc_t",
}

# Types that cannot be cast to uint64_t (structs by value, function pointers)
_NON_CASTABLE_TYPES = {
    "hipGraphicsResource_t",  # struct*
    "hipIpcEventHandle_t",    # struct by value
    "hipIpcMemHandle_t",      # struct by value
    "hipPitchedPtr",          # struct by value
    "hipExtent",              # struct by value
    "hipPos",                 # struct by value
    "hipMemLocation",         # struct with enum+int
    "hipStreamCallback_t",    # function pointer
    "hipHostFn_t",            # function pointer
    "hipLinkState_t",         # opaque ptr (treat as non-castable)
    "hipChannelFormatDesc",   # struct by value
}

# Enum types -> int32_t
_ENUM_TYPES = {
    "hipError_t", "hipMemcpyKind", "hipFuncCache_t", "hipSharedMemConfig",
    "hipJitOption", "hipLimit_t", "hipDeviceAttribute_t", "hipComputeMode",
    "hipMemoryType", "hipMemLocationType", "hipMemAllocationType",
    "hipMemoryAdvise", "hipStreamCaptureMode", "hipGraphNodeType",
    "hipKernelNodeAttrID", "hipStreamUpdateCaptureDependenciesFlags",
    "hipAccessProperty", "hipMemOperationType", "hipArraySparseSubresourceType",
    "hipMemPoolAttr", "hipMemRangeAttribute", "hipMemRangeCoherenceMode",
    "hipFuncAttribute", "hipDeviceP2PAttr", "hipGraphDebugDotFlags",
    "hipGraphInstantiateFlags", "hipUserObjectFlags", "hipUserObjectRetainFlags",
    "hipExternalMemoryHandleType", "hipExternalSemaphoreHandleType",
    "hipTextureFilterMode", "hipTextureAddressMode", "hipTextureMipmapFilterMode",
    "hipResourcetype", "hipResourceViewFormat", "hipChannelFormatKind",
    "hipKernelAttribute",
}

# Scalar type map: normalised type string -> C field type
_SCALAR_MAP = [
    # order matters: longer / more specific first
    ("unsigned long long", "uint64_t"),
    ("long long",          "int64_t"),
    ("unsigned int",       "uint32_t"),
    ("uint64_t",           "uint64_t"),
    ("int64_t",            "int64_t"),
    ("uint32_t",           "uint32_t"),
    ("uint16_t",           "uint16_t"),
    ("uint8_t",            "uint8_t"),
    ("int32_t",            "int32_t"),
    ("int16_t",            "int16_t"),
    ("int8_t",             "int8_t"),
    ("size_t",             "uint64_t"),
    ("unsigned",           "uint32_t"),
    ("int",                "int32_t"),
    ("float",              "float"),
    ("double",             "double"),
    ("bool",               "uint8_t"),
    ("char",               "int8_t"),
]


def normalise_field_type(raw_type: str) -> str:
    """
    Convert a C parameter type string to the field type used in the struct.
    Returns a sentinel "__DIM3__" for dim3/uint3 (caller expands to x/y/z).
    """
    t = raw_type.strip()
    # strip const/volatile
    t_nc = re.sub(r'\b(const|volatile|restrict)\b', '', t).strip()
    t_nc = re.sub(r'\s+', ' ', t_nc)

    # Any pointer -> uint64_t
    if '*' in t_nc:
        return "uint64_t"

    # dim3 / uint3 -> expand inline
    base = t_nc.split()[0]
    if base in ("dim3", "uint3"):
        return "__DIM3__"

    # Opaque handle
    if base in _HANDLE_TYPES:
        return "uint64_t"

    # Enum
    if base in _ENUM_TYPES:
        return "int32_t"

    # Scalar
    for key, mapped in _SCALAR_MAP:
        if t_nc == key:
            return mapped

    # Unknown composite type (struct passed by value) -> comment, uint64_t placeholder
    return f"uint64_t /* {t} */"


@dataclass
class Param:
    raw_type: str
    name: str    # may be empty for unnamed params

    def field_lines(self) -> List[str]:
        ft = normalise_field_type(self.raw_type)
        safe = self.name or "unnamed"
        if ft == "__DIM3__":
            return [
                f"uint32_t {safe}_x;",
                f"uint32_t {safe}_y;",
                f"uint32_t {safe}_z;",
            ]
        return [f"{ft} {safe};"]


@dataclass
class ApiEntry:
    name:     str
    ret_type: str
    params:   List[Param]
    table:    str   # "runtime" | "compiler"


# ---------------------------------------------------------------------------
# Low-level text extraction
# ---------------------------------------------------------------------------

def _strip_comments(text: str) -> str:
    text = re.sub(r'/\*.*?\*/', ' ', text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', ' ', text)
    return text


def _extract_balanced_parens(text: str, start: int) -> Tuple[int, str]:
    """
    Starting at text[start] which must be '(', return (end_idx, inner_text)
    where end_idx is the index AFTER the closing ')'.
    """
    assert text[start] == '('
    depth = 0
    i = start
    while i < len(text):
        if text[i] == '(':
            depth += 1
        elif text[i] == ')':
            depth -= 1
            if depth == 0:
                return i + 1, text[start + 1:i]
        i += 1
    raise ValueError(f"Unbalanced parens starting at {start}")


def find_typedef_for(text: str, typedef_name: str) -> Optional[str]:
    """
    Find the full text of   typedef ... (*typedef_name)(...)  ;
    by locating `(*typedef_name)` then parsing balanced parens.
    Returns the full typedef string or None.
    """
    needle = f"(*{typedef_name})"
    idx = text.find(needle)
    if idx == -1:
        return None

    # Walk backward to find 'typedef'
    before = text[:idx]
    typedef_start = before.rfind('typedef')
    if typedef_start == -1:
        return None

    # Now parse: after needle comes the params in parens, then optional whitespace and ';'
    after_needle = idx + len(needle)
    # skip whitespace
    i = after_needle
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '(':
        return None
    end, _inner = _extract_balanced_parens(text, i)
    # find the semicolon
    j = end
    while j < len(text) and text[j] != ';':
        j += 1
    full = text[typedef_start:j + 1]
    return full


def _split_params(s: str) -> List[str]:
    """Split comma-separated params respecting nested parens."""
    result = []
    depth = 0
    current: List[str] = []
    for ch in s:
        if ch in '(<':
            depth += 1
        elif ch in ')>':
            depth -= 1
        if ch == ',' and depth == 0:
            result.append(''.join(current).strip())
            current = []
        else:
            current.append(ch)
    if ''.join(current).strip():
        result.append(''.join(current).strip())
    return result


def _parse_param(raw: str) -> Optional[Param]:
    """Parse one parameter like 'const void* devPtr' or 'size_t size'."""
    raw = raw.strip()
    if not raw or raw == '...':
        return None
    # Array [] -> pointer *
    raw = re.sub(r'\[.*?\]', '*', raw)

    # Collapse whitespace
    raw = re.sub(r'\s+', ' ', raw)

    # If the last token is a valid identifier and not a type keyword, it's the name.
    m = re.match(r'^(.*?)\s*(\**)([a-zA-Z_]\w*)$', raw)
    if m:
        type_prefix = m.group(1).strip()
        stars        = m.group(2)
        potential_name = m.group(3)
        # Heuristic: if the type_prefix is non-empty and not just a closing bracket,
        # treat potential_name as the variable name.
        if type_prefix:
            full_type = (type_prefix + stars).strip()
            return Param(raw_type=full_type, name=potential_name)
        else:
            # Only one token: it's both the type and has no name
            return Param(raw_type=raw, name='')

    return Param(raw_type=raw, name='')


def _parse_typedef_text(full_text: str, func_name: str) -> Optional[ApiEntry]:
    """
    Parse a complete typedef string into an ApiEntry.
    full_text looks like:
        typedef hipError_t (*t_hipMalloc)(void** ptr, size_t size);
    """
    # Extract return type: everything between 'typedef' and '(*'
    m = re.match(r'typedef\s+(.+?)\s*\(\s*\*', full_text, re.DOTALL)
    if not m:
        return None
    ret_type = re.sub(r'\s+', ' ', m.group(1)).strip()

    # Extract params: content of the second paren group
    # Find '(*tname)' then the next '('
    needle = f"(*t_{func_name})"
    idx = full_text.find(needle)
    if idx == -1:
        # compiler stubs have t___name
        needle2 = f"(*t__{func_name})"
        idx = full_text.find(needle2)
        if idx == -1:
            return None

    after = idx + len(needle if idx == full_text.find(needle) else needle2)
    i = after
    while i < len(full_text) and full_text[i].isspace():
        i += 1
    if full_text[i] != '(':
        return None
    _, params_text = _extract_balanced_parens(full_text, i)

    params_text = params_text.strip()
    params: List[Param] = []
    if params_text and params_text != 'void':
        for raw_p in _split_params(params_text):
            p = _parse_param(raw_p)
            if p:
                params.append(p)

    table = "compiler" if func_name.startswith("_hip") else "runtime"
    return ApiEntry(name=func_name, ret_type=ret_type, params=params, table=table)


# ---------------------------------------------------------------------------
# Main parse entry point
# ---------------------------------------------------------------------------

# Compiler API names in dispatch table order
_COMPILER_APIS = [
    "__hipPopCallConfiguration",
    "__hipPushCallConfiguration",
    "__hipRegisterFatBinary",
    "__hipRegisterFunction",
    "__hipRegisterManagedVar",
    "__hipRegisterSurface",
    "__hipRegisterTexture",
    "__hipRegisterVar",
    "__hipUnregisterFatBinary",
]


def parse_hip_api_trace(path: Path) -> List[ApiEntry]:
    text = path.read_text(encoding='utf-8')
    text = _strip_comments(text)

    entries: List[ApiEntry] = []

    # ---- Compiler stubs (fixed list, specific typedef name) ----
    for func_name in _COMPILER_APIS:
        typedef_name = "t_" + func_name   # e.g. t___hipRegisterFatBinary
        full = find_typedef_for(text, typedef_name)
        if not full:
            print(f"WARNING: typedef not found for {func_name}", file=sys.stderr)
            continue
        # For parsing, strip the leading underscores from func_name for the needle match
        entry = _parse_typedef_text(full, func_name)
        if not entry:
            print(f"WARNING: failed to parse typedef for {func_name}", file=sys.stderr)
            continue
        entry.table = "compiler"
        entries.append(entry)

    # ---- Runtime APIs — find all t_hipXxx typedefs ----
    # Locate every  (*t_hipXxx)  occurrence, then extract the full typedef
    runtime_name_pattern = re.compile(r'\(\s*\*\s*t_(hip\w+)\s*\)')
    seen = set()
    for m in runtime_name_pattern.finditer(text):
        func_name = m.group(1)  # e.g. "hipMalloc"
        if func_name in seen:
            continue
        seen.add(func_name)

        typedef_name = "t_" + func_name
        full = find_typedef_for(text, typedef_name)
        if not full:
            print(f"WARNING: typedef not found for {func_name}", file=sys.stderr)
            continue
        entry = _parse_typedef_text(full, func_name)
        if not entry:
            print(f"WARNING: failed to parse typedef for {func_name}", file=sys.stderr)
            continue
        entry.table = "runtime"
        entries.append(entry)

    return entries


# ---------------------------------------------------------------------------
# Header generation
# ---------------------------------------------------------------------------

_HEADER_PREAMBLE = """\
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* ============================================================================
 * hrr_api_args.h  -  AUTO-GENERATED by gen_hrr_api_args.py
 *
 * DO NOT EDIT MANUALLY.
 * Regenerate with:
 *     python gen_hrr_api_args.py
 *
 * One packed struct per HIP API covering both HipDispatchTable (runtime) and
 * HipCompilerDispatchTable (compiler stubs).
 *
 * Archive format (v3):
 *   events.bin:
 *     [0..7]   hrr_file_header  { magic, version, reserved }
 *     [8..]    hrr_event_header (32 bytes) + payload bytes, repeated per event
 *   blobs/<2hex>/   content-addressed raw buffers (FNV-1a-128 hash)
 *   code_objects/   .hsaco ELFs keyed by hash
 *
 * hrr_event_header fields (32 bytes, pack(1)):
 *   - event_type     uint16_t  hrr_api_id_t index
 *   - sequence_id    uint64_t  monotonically increasing per capture session
 *   - timestamp_ns   uint64_t  wall-clock at capture time
 *   - thread_id      uint64_t  OS thread that made the call (cached per thread)
 *   - payload_length uint16_t  total record size in bytes (incl. header)
 *   - reserved       uint8_t[4]  padding to 32 bytes
 *
 * Payload bytes (after the 32-byte header):
 *   - ret          int32_t (hipError_t); absent for void returns
 *   - pointer / handle types   uint64_t  (address stored as integer)
 *   - dim3 / uint3             expanded to three uint32_t _x/_y/_z fields
 *   - size_t                   uint64_t
 *   - scalars                  native C type (int32_t, uint32_t, float, ...)
 *
 * Extra fields on selected APIs (blob hashes, module_id) are appended AFTER
 * the normal parameters — see EXTRA_FIELDS in gen_hrr_api_args.py.
 *
 * The structs use #pragma pack(1) so layout is identical on all platforms.
 * ============================================================================
 */
#pragma once

#include <stdint.h>

/* ---- Archive format constants ---- */
#define HRR_MAGIC   ((uint32_t)0x52524845u)  /* "HRRE" */
#define HRR_VERSION ((uint16_t)3u)

/* Written once at byte 0 of events.bin. */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;    /* HRR_MAGIC                */
    uint16_t version;  /* HRR_VERSION              */
    uint16_t reserved; /* zero                     */
} hrr_file_header;
#pragma pack(pop)

#ifdef __cplusplus
static_assert(sizeof(hrr_file_header) == 8, "hrr_file_header must be 8 bytes");
#endif

#pragma pack(push, 1)

/* Per-record prefix embedded as the first member of every hrr_args_* struct.
 * write_event_raw() fills all fields before writing. 32 bytes, pack(1).
 * magic and version are in hrr_file_header at byte 0 of events.bin (once). */
typedef struct {
    uint16_t event_type;     /* hrr_api_id_t index of the captured API    */
    uint64_t sequence_id;    /* monotonically increasing counter          */
    uint64_t timestamp_ns;   /* wall-clock at capture time (MONOTONIC)    */
    uint64_t thread_id;      /* OS thread ID (cached per thread)          */
    uint16_t payload_length; /* total record size in bytes (incl. header) */
    uint8_t  reserved[4];    /* padding to 32 bytes; zero on write        */
} hrr_event_header;

#ifdef __cplusplus
static_assert(sizeof(hrr_event_header) == 32, "hrr_event_header must be 32 bytes");
#endif

"""

_FOOTER = """
#pragma pack(pop)
"""


def _is_void_return(ret: str) -> bool:
    r = ret.strip()
    if '*' in r:
        return False
    return bool(re.fullmatch(r'(const\s+)?void', r))


def generate_struct(entry: ApiEntry) -> str:
    lines: List[str] = []
    sname = f"hrr_args_{entry.name}"

    # Comment showing original signature
    param_sig = ', '.join(
        (p.raw_type + ' ' + p.name).strip()
        for p in entry.params
    )
    lines.append(f"/* {entry.ret_type} {entry.name}({param_sig}) */")
    lines.append("typedef struct {")
    lines.append("    hrr_event_header hdr;")

    # Return value
    if not _is_void_return(entry.ret_type):
        if '*' in entry.ret_type:
            lines.append("    uint64_t ret;")
        else:
            ft = normalise_field_type(entry.ret_type)
            if ft == "__DIM3__":
                ft = "uint64_t"
            lines.append(f"    {ft} ret;")

    # Parameters — special-case __hipRegisterFatBinary raw data* -> blob fields
    if entry.name == "__hipRegisterFatBinary":
        # Skip the normal params (just `void* data`); extra fields below replace them
        pass
    else:
        unnamed_count = 0
        for param in entry.params:
            pname = param.name
            if not pname:
                pname = f"p{unnamed_count}"
                unnamed_count += 1
            ft = normalise_field_type(param.raw_type)
            safe = pname
            if ft == "__DIM3__":
                for suffix in ('_x', '_y', '_z'):
                    lines.append(f"    uint32_t {safe}{suffix};")
            else:
                lines.append(f"    {ft} {safe};")

    # Extra fields
    for (ftype, fname, fcomment) in EXTRA_FIELDS.get(entry.name, []):
        lines.append(f"    {ftype} {fname};  /* {fcomment} */")

    lines.append(f"}} {sname};")

    # Emit static_assert for every inline uint8_t array whose size is hard-coded.
    # Guarded by HIP_INCLUDE_HIP_HIP_RUNTIME_H because the referenced types
    # (hipMemcpy3DParms, hipMemPoolProps, etc.) are only defined when the full
    # HIP runtime header is included.  hrr_api_args.h may be included in
    # translation units (e.g. hrr_reader.cpp) that use only minimal headers.
    asserts = []
    for (ftype, fname, _fcomment) in EXTRA_FIELDS.get(entry.name, []):
        if not ftype.startswith("uint8_t") or '[' not in fname:
            continue
        base = fname[:fname.index('[')]
        ctype = _INLINE_STRUCT_ASSERTS.get(base)
        if not ctype:
            continue
        # Extract the declared array size N from "name[N]"
        n = fname[fname.index('[')+1 : fname.index(']')]
        asserts.append(
            f"static_assert(sizeof({ctype}) <= {n},"
            f' "hrr_args_{entry.name}::{fname} too small for {ctype}");'
        )
    if asserts:
        lines.append("#ifdef HIP_INCLUDE_HIP_HIP_RUNTIME_H")
        lines.extend(asserts)
        lines.append("#endif")

    lines.append("")
    return "\n".join(lines)


def generate_header(entries: List[ApiEntry]) -> str:
    parts = [_HEADER_PREAMBLE]

    parts.append("/* ---- Compiler dispatch stubs ---- */\n")
    for e in entries:
        if e.table == "compiler":
            parts.append(generate_struct(e))

    parts.append("/* ---- Runtime dispatch APIs ---- */\n")
    for e in entries:
        if e.table == "runtime":
            parts.append(generate_struct(e))

    # Enum of API IDs for use by writer/reader
    parts.append("/* ---- API id enumeration ---- */")
    parts.append("typedef enum hrr_api_id {")
    for idx, e in enumerate(entries):
        # Enum name: strip leading underscores, uppercase
        enum_name = "HRR_API_" + e.name.lstrip('_').upper()
        parts.append(f"    {enum_name} = {idx},")
    parts.append(f"    HRR_API_COUNT = {len(entries)}")
    parts.append("} hrr_api_id_t;\n")

    # Name table declaration
    parts.append("/* Array of API names indexed by hrr_api_id_t */")
    parts.append("#ifdef HRR_API_ARGS_IMPLEMENTATION")
    parts.append("const char* const hrr_api_names[HRR_API_COUNT] = {")
    for e in entries:
        parts.append(f'    "{e.name}",')
    parts.append("};")
    parts.append("#else")
    parts.append("extern const char* const hrr_api_names[HRR_API_COUNT];")
    parts.append("#endif\n")

    parts.append(_FOOTER)
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Capture CPP generation
# ---------------------------------------------------------------------------

_CPP_PREAMBLE = """\
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* ============================================================================
 * hip_capture_generated.cpp  -  AUTO-GENERATED by gen_hrr_api_args.py
 *
 * DO NOT EDIT MANUALLY.
 * Regenerate with:
 *     python gen_hrr_api_args.py
 *
 * Contains:
 *   - capture_hipFoo() shim for every HIP API not in MANUAL_CAPTURE_APIS
 *   - hip_capture_build_table()          overrides all runtime dispatch slots
 *   - hip_capture_build_compiler_table() overrides all compiler dispatch slots
 *
 * MANUAL_CAPTURE_APIS (kernel launches, memcpy blob capture, module load,
 * fat binary registration) are implemented by hand in hip_capture.cpp;
 * the generated shims for those are simple pass-throughs.
 *
 * Every generated shim:
 *   1. Calls the real function via g_real_table / g_real_compiler_table
 *   2. If the call succeeded (hipError_t return) or always (void return):
 *      - Fills hrr_args_hipFoo ret + all args (hdr stamped by write_event_raw)
 *      - Calls writer::write_event_raw() which stamps thread_id/sequence_id
 * No hip_capture_enabled() check — shims are only installed when capture is active.
 * ============================================================================
 */

// This file is compiled as part of amdhip64; it #includes internal headers.
#include "hip_capture.h"
#include "hip_capture_writer.h"

// hrr_api_args.h lives in the same directory (hipamd/src/hrr/)
#include "hrr_api_args.h"

#include "hip/amd_detail/hip_api_trace.hpp"

#include <atomic>
#include <cstdint>
#include <cstring>

// These global tables are defined (non-static) in hip_capture.cpp
extern HipDispatchTable         g_real_table;
extern HipDispatchTable         g_cap_table;
extern std::atomic<bool>        g_installed;
extern std::atomic<bool>        g_table_built;
extern HipCompilerDispatchTable g_real_compiler_table;
extern std::atomic<bool>        g_compiler_installed;

namespace hip {
const HipDispatchTable*         GetHipDispatchTable();
const HipCompilerDispatchTable* GetHipCompilerDispatchTable();
}

"""


def _cpp_param_decl(entry: ApiEntry) -> str:
    """Generate function parameter declaration list."""
    if not entry.params:
        return "void"
    parts = []
    unnamed = 0
    for p in entry.params:
        name = p.name or f"p{unnamed}"
        if not p.name:
            unnamed += 1
        parts.append(f"{p.raw_type} {name}")
    return ", ".join(parts)


def _cpp_passthrough_args(entry: ApiEntry) -> str:
    """Generate argument list to forward to the real function."""
    parts = []
    unnamed = 0
    for p in entry.params:
        name = p.name or f"p{unnamed}"
        if not p.name:
            unnamed += 1
        parts.append(name)
    return ", ".join(parts)


def _ret_type_to_error_value(ret: str) -> Optional[str]:
    """Return a safe 'failure' sentinel for the return type, or None for void."""
    r = ret.strip()
    if _is_void_return(r):
        return None
    if 'hipError_t' in r or 'hip_error' in r.lower():
        return 'hipErrorUnknown'
    if '*' in r:
        return 'nullptr'
    return '0'


def _is_output_ptr_param(param: Param) -> bool:
    """True if this param is an output pointer (T** or handle type* or int-typedef*)."""
    t = param.raw_type.strip()
    if t.count('*') >= 2:
        return True
    # Single pointer to any handle type → output
    _OUTPUT_HANDLE_TYPES = (
        'hipStream_t', 'hipEvent_t', 'hipModule_t', 'hipFunction_t',
        'hipMemPool_t', 'hipGraph_t', 'hipGraphExec_t', 'hipGraphNode_t',
        'hipArray_t', 'hipMipmappedArray_t',
        'hipSurfaceObject_t', 'hipTextureObject_t',
        'hipCtx_t', 'hipUserObject_t',
        'hipMemGenericAllocationHandle_t',
    )
    for handle in _OUTPUT_HANDLE_TYPES:
        if handle + '*' in t or handle + ' *' in t:
            return True
    # hipDevice_t is typedef'd as int — hipDevice_t* is an output int pointer
    if 'hipDevice_t' in t and '*' in t and 'hipDeviceptr_t' not in t:
        return True
    return False


def _get_base_type(raw_type: str) -> str:
    """Extract the base type name (strip const/volatile/struct/enum/class/pointers)."""
    t = raw_type.strip()
    base = re.sub(r'\b(const|volatile|restrict|struct|enum|class|union)\b', '', t).replace('*', '').strip()
    base = re.sub(r'\s+', ' ', base)
    parts = base.split()
    return parts[0] if parts else ''


def _fill_param(lines: List[str], p: Param, name: str, ft: str) -> None:
    """Emit lines that fill a.name from the C variable name."""
    t = p.raw_type.strip()
    base = _get_base_type(t)

    if ft == "__DIM3__":
        lines.append(f"    a.{name}_x = {name}.x;")
        lines.append(f"    a.{name}_y = {name}.y;")
        lines.append(f"    a.{name}_z = {name}.z;")
    elif base in _NON_CASTABLE_TYPES:
        lines.append(f"    a.{name} = 0;  // non-castable type skipped")
    elif base == 'hipDevice_t':
        # hipDevice_t is int, not a pointer
        lines.append(f"    a.{name} = static_cast<uint64_t>(static_cast<int>({name}));")
    elif base == 'hipDeviceptr_t' or t == 'hipDeviceptr_t':
        # hipDeviceptr_t is void* — must use uintptr_t cast
        lines.append(f"    a.{name} = static_cast<uint64_t>(reinterpret_cast<uintptr_t>({name}));")
    elif ft == "uint64_t" and '*' in t:
        lines.append(f"    a.{name} = reinterpret_cast<uint64_t>({name});")
    elif ft == "uint64_t" and base in _HANDLE_TYPES:
        lines.append(f"    a.{name} = reinterpret_cast<uint64_t>({name});")
    else:
        lines.append(f"    a.{name} = static_cast<decltype(a.{name})>({name});")


def _fill_output_param_post(lines: List[str], p: Param, name: str) -> None:
    """Emit lines that fill a.name AFTER the real call for output pointer params."""
    t = p.raw_type.strip()
    # hipDevice_t is int — dereference gives int, use static_cast
    if 'hipDevice_t' in t and 'hipDeviceptr_t' not in t:
        lines.append(f"    if ({name}) a.{name} = static_cast<uint64_t>(static_cast<int>(*{name}));")
    else:
        # For output pointers, dereference to get the created handle/address.
        # Guard with null check — some output params are optional (e.g. pErrorNode in hipGraphInstantiate).
        lines.append(f"    if ({name}) a.{name} = reinterpret_cast<uint64_t>(*{name});")


def generate_shim(entry: ApiEntry) -> str:
    """Generate a single capture shim function.
    MANUAL_CAPTURE_APIS: returns empty string — hand-written in hip_capture.cpp.
    """
    is_manual   = entry.name in MANUAL_CAPTURE_APIS
    is_passonly = entry.name in PASSTHROUGH_ONLY
    is_compiler = entry.table == "compiler"
    void_ret    = _is_void_return(entry.ret_type)
    # const char* return (hipGetErrorName/String) — store pointer as uint64_t
    is_const_char_ret = entry.ret_type.strip() in ('const char*', 'const char *')
    # Non-hipError_t struct/scalar return (e.g. hipCreateChannelDesc -> hipChannelFormatDesc)
    # Cannot compare r == hipSuccess; treat as passthrough (no capture)
    ret_base = _get_base_type(entry.ret_type)
    is_non_hiperrort_ret = (not void_ret and not is_const_char_ret
                            and ret_base not in ('hipError_t',)
                            and '*' not in entry.ret_type)

    # Manual APIs are fully implemented in hip_capture.cpp (non-static).
    # The build_table function will extern-reference them directly.
    if is_manual:
        return ""

    param_decl = _cpp_param_decl(entry)
    fwd_args   = _cpp_passthrough_args(entry)
    sname      = f"hrr_args_{entry.name}"

    # Which table / fn ptr name
    if is_compiler:
        table_name = "g_real_compiler_table"
        fn_field   = f"{entry.name}_fn"
    else:
        table_name = "g_real_table"
        fn_field   = f"{entry.name}_fn"

    lines = []
    lines.append(f"// Generated shim")
    lines.append(f"static {entry.ret_type} capture_{entry.name}({param_decl}) {{")

    if is_non_hiperrort_ret:
        # Return type is a struct/scalar (not hipError_t) — can't capture, just forward
        lines.append(f"  return {table_name}.{fn_field}({fwd_args});")
        lines.append(f"}}")
        return '\n'.join(lines) + '\n'

    if void_ret:
        lines.append(f"  {table_name}.{fn_field}({fwd_args});")
        if not is_passonly:
            # No hip_capture_enabled() check — shim is only installed when capture is active
            # write_event_raw() stamps thread_id / sequence_id into hdr automatically
            lines.append(f"  {{")
            lines.append(f"    {sname} a{{}};")
            # Pre-call params (non-output)
            unnamed = 0
            output_params = []
            for p in entry.params:
                name = p.name or f"p{unnamed}"
                if not p.name: unnamed += 1
                ft = normalise_field_type(p.raw_type)
                if _is_output_ptr_param(p):
                    output_params.append((p, name, ft))
                else:
                    _fill_param(lines, p, name, ft)
            # Post-call output params
            for p, name, ft in output_params:
                _fill_output_param_post(lines, p, name)
            enum_name = "HRR_API_" + entry.name.lstrip('_').upper()
            lines.append(f"    hrr_cap::writer::write_event_raw({enum_name}, &a.hdr, sizeof(a));")
            lines.append(f"  }}")
    else:
        lines.append(f"  {entry.ret_type} r = {table_name}.{fn_field}({fwd_args});")
        if not is_passonly:
            # No hip_capture_enabled() check — shim is only installed when capture is active
            # For const char* return, no hipSuccess check (always valid)
            if is_const_char_ret:
                lines.append(f"  {{")
            else:
                lines.append(f"  if (r == hipSuccess) {{")
            lines.append(f"    {sname} a{{}};")
            if is_const_char_ret:
                lines.append(f"    a.ret = reinterpret_cast<uint64_t>(r);")
            else:
                lines.append(f"    a.ret         = static_cast<int32_t>(r);")
            # Fill params — separate pre/post-call
            unnamed = 0
            output_params = []
            for p in entry.params:
                name = p.name or f"p{unnamed}"
                if not p.name: unnamed += 1
                ft = normalise_field_type(p.raw_type)
                if _is_output_ptr_param(p):
                    output_params.append((p, name, ft))
                else:
                    _fill_param(lines, p, name, ft)
            # Post-call: fill output ptr fields from dereferenced values
            for p, name, ft in output_params:
                _fill_output_param_post(lines, p, name)
            enum_name = "HRR_API_" + entry.name.lstrip('_').upper()
            lines.append(f"    hrr_cap::writer::write_event_raw({enum_name}, &a.hdr, sizeof(a));")
            lines.append(f"  }}")
        lines.append(f"  return r;")

    lines.append(f"}}")
    lines.append("")
    return "\n".join(lines)


def generate_build_table(entries: List[ApiEntry]) -> str:
    """Generate hip_capture_build_table() and hip_capture_build_compiler_table()."""
    runtime_entries  = [e for e in entries if e.table == "runtime"]
    compiler_entries = [e for e in entries if e.table == "compiler"]

    lines = []

    # Forward-declare the hand-written shims from hip_capture.cpp (non-static there)
    manual_runtime  = [e for e in runtime_entries  if e.name in MANUAL_CAPTURE_APIS]
    manual_compiler = [e for e in compiler_entries if e.name in MANUAL_CAPTURE_APIS]
    if manual_runtime or manual_compiler:
        lines.append("// Forward declarations for hand-written shims (non-static in hip_capture.cpp)")
        for e in manual_runtime:
            lines.append(f"extern {e.ret_type} capture_{e.name}({_cpp_param_decl(e)});")
        for e in manual_compiler:
            lines.append(f"extern {e.ret_type} capture_{e.name}({_cpp_param_decl(e)});")
        lines.append("")

    lines.append("void hip_capture_build_table() {")
    lines.append("  // Guard: safe to call only once. A second call after shims are installed")
    lines.append("  // would snapshot shim ptrs into g_real_table, causing infinite recursion.")
    lines.append("  if (g_table_built.exchange(true)) return;")
    lines.append("  // Snapshot the live real table; copy all slots as pass-through base")
    lines.append("  g_real_table = *hip::GetHipDispatchTable();")
    lines.append("  g_cap_table  = g_real_table;")
    lines.append("")
    lines.append("  // Override every runtime slot with its capture shim")
    for e in runtime_entries:
        lines.append(f"  g_cap_table.{e.name}_fn = capture_{e.name};")
    lines.append("}")
    lines.append("")

    lines.append("void hip_capture_build_compiler_table() {")
    lines.append("  // Guard: a second call after shims are installed would snapshot shim ptrs")
    lines.append("  // into g_real_compiler_table — __hipRegister* calls from the compiler")
    lines.append("  // would then recurse back through themselves.")
    lines.append("  if (g_compiler_installed.exchange(true)) return;")
    lines.append("  g_real_compiler_table = *hip::GetHipCompilerDispatchTable();")
    lines.append("  HipCompilerDispatchTable cap = g_real_compiler_table;")
    for e in compiler_entries:
        lines.append(f"  cap.{e.name}_fn = capture_{e.name};")
    lines.append("  std::memcpy(const_cast<HipCompilerDispatchTable*>(hip::GetHipCompilerDispatchTable()),")
    lines.append("              &cap, sizeof(HipCompilerDispatchTable));")
    lines.append("}")
    lines.append("")

    return "\n".join(lines)


def generate_capture_cpp(entries: List[ApiEntry]) -> str:
    parts = [_CPP_PREAMBLE]

    parts.append("// ============================================================")
    parts.append("// Capture shims")
    parts.append("// ============================================================")
    parts.append("")

    for e in entries:
        parts.append(generate_shim(e))

    parts.append("// ============================================================")
    parts.append("// Table builders")
    parts.append("// ============================================================")
    parts.append("")
    parts.append(generate_build_table(entries))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Playback CPP generation
# ---------------------------------------------------------------------------

_PLAYBACK_CPP_PREAMBLE = """\
/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
/* ============================================================================
 * hip_playback_generated.cpp  -  AUTO-GENERATED by gen_hrr_api_args.py
 * DO NOT EDIT MANUALLY.
 * Regenerate with:
 *     python gen_hrr_api_args.py
 *
 * Contains:
 *   - playback_hipFoo() for every HIP API
 *     Signature: hipError_t playback_foo(PlaybackContext&, const uint8_t*)
 *   - hrr_playback_dispatch[HRR_API_COUNT]  — indexed by hrr_api_id_t
 * ============================================================================ */

#include "hip_playback.h"
#include "hrr_api_args.h"
#include <hip/hip_runtime.h>
#include <cstring>

// Manual playback implementations (extern'd below) are in hip_playback.cpp
// Compiler APIs are no-ops during playback

"""

# Handle types that need translation during playback
# Maps C handle type -> (translate method, record method, remove method)
_PLAYBACK_HANDLE_INFO: Dict[str, Tuple[str, str, str]] = {
    'hipStream_t':          ('ctx.translate_stream',    'ctx.record_stream',    'ctx.remove_stream'),
    'hipEvent_t':           ('ctx.translate_event',     'ctx.record_event',     'ctx.remove_event'),
    'hipModule_t':          ('ctx.translate_module',    'ctx.record_module',    'ctx.remove_module'),
    'hipFunction_t':        ('ctx.translate_func',      'ctx.record_func',      'ctx.remove_func'),
    'hipMemPool_t':         ('ctx.translate_mempool',   'ctx.record_mempool',   'ctx.remove_mempool'),
    'hipArray_t':           ('ctx.translate_array',     'ctx.record_array',     'ctx.remove_array'),
    'hipMipmappedArray_t':  ('ctx.translate_mipmapped', 'ctx.record_mipmapped', 'ctx.remove_mipmapped'),
    'hipGraph_t':           ('ctx.translate_graph',     'ctx.record_graph',     'ctx.remove_graph'),
    'hipGraphExec_t':       ('ctx.translate_graph_exec','ctx.record_graph_exec','ctx.remove_graph_exec'),
    'hipSurfaceObject_t':   ('ctx.translate_surface',   'ctx.record_surface',   'ctx.remove_surface'),
    'hipTextureObject_t':   ('ctx.translate_texture',   'ctx.record_texture',   'ctx.remove_texture'),
}
_PLAYBACK_HANDLE_TRANSLATE = {k: v[0] for k, v in _PLAYBACK_HANDLE_INFO.items()}

# APIs that create device allocations: API name -> (rec_ptr_param, size_param)
# rec_ptr_param is the name of the output param storing the address (in the struct)
# size_param is the name of the size param in the struct
_ALLOC_CREATE_APIS: Dict[str, Tuple[str, str]] = {
    'hipMalloc':             ('ptr', 'size'),
    'hipMallocAsync':        ('dev_ptr', 'size'),
    'hipMallocFromPoolAsync':('dev_ptr', 'size'),
    'hipMallocManaged':      ('dev_ptr', 'size'),
    'hipExtMallocWithFlags': ('ptr', 'size'),
    'hipMallocPitch':        ('ptr', 'width'),  # approximate; width used as proxy
    'hipHostMalloc':         ('ptr', 'size'),
    'hipHostAlloc':          ('ptr', 'size'),
}

# APIs that free device allocations: API name -> rec_ptr_param name in struct
_ALLOC_FREE_APIS: Dict[str, str] = {
    'hipFree':         'ptr',
    'hipFreeAsync':    'dev_ptr',
    'hipHostFree':     'ptr',
}

# APIs that destroy handles: API name -> (handle param name in struct, handle type)
# Param names MUST match the actual typedef parameter names in hip_api_trace.hpp.
_HANDLE_DESTROY_APIS: Dict[str, Tuple[str, str]] = {
    'hipStreamDestroy':        ('stream',        'hipStream_t'),
    'hipEventDestroy':         ('event',         'hipEvent_t'),
    'hipModuleUnload':         ('module',        'hipModule_t'),
    'hipMemPoolDestroy':       ('mem_pool',      'hipMemPool_t'),
    'hipArrayDestroy':         ('array',         'hipArray_t'),
    'hipFreeMipmappedArray':   ('mipmappedArray','hipMipmappedArray_t'),
    'hipGraphDestroy':         ('graph',         'hipGraph_t'),
    'hipGraphExecDestroy':     ('graphExec',     'hipGraphExec_t'),
    'hipDestroySurfaceObject': ('surfaceObject', 'hipSurfaceObject_t'),
    'hipDestroyTextureObject': ('textureObject', 'hipTextureObject_t'),
}

# APIs that create handles: API name -> list of (output param name, handle type)
# The param name MUST match the actual parameter name in the hip_api_trace.hpp typedef.
_HANDLE_CREATE_APIS: Dict[str, List[Tuple[str, str]]] = {
    # Stream / event — in MANUAL_PLAYBACK_APIS; entries here only for completeness
    'hipStreamCreate':              [('stream',      'hipStream_t')],
    'hipStreamCreateWithFlags':     [('stream',      'hipStream_t')],
    'hipStreamCreateWithPriority':  [('stream',      'hipStream_t')],
    'hipEventCreate':               [('event',       'hipEvent_t')],
    'hipEventCreateWithFlags':      [('event',       'hipEvent_t')],
    # Memory pools
    'hipMemPoolCreate':             [('mem_pool',     'hipMemPool_t')],
    'hipMemPoolImportFromShareableHandle': [('mem_pool', 'hipMemPool_t')],
    # Arrays — param names from hip_api_trace.hpp
    'hipArrayCreate':               [('pHandle',      'hipArray_t')],
    'hipArray3DCreate':             [('array',        'hipArray_t')],
    'hipMallocArray':               [('array',        'hipArray_t')],
    'hipMalloc3DArray':             [('array',        'hipArray_t')],
    'hipMallocMipmappedArray':      [('mipmappedArray', 'hipMipmappedArray_t')],
    # Graphs
    'hipGraphCreate':               [('pGraph',       'hipGraph_t')],
    'hipGraphClone':                [('pGraphClone',  'hipGraph_t')],
    'hipGraphInstantiate':          [('pGraphExec',   'hipGraphExec_t')],
    'hipGraphInstantiateWithFlags': [('pGraphExec',   'hipGraphExec_t')],
    # Texture / surface objects
    'hipCreateSurfaceObject':       [('pSurfObject',  'hipSurfaceObject_t')],
    'hipCreateTextureObject':       [('pTexObject',   'hipTextureObject_t')],
}


def _playback_arg(p: Param, name: str, pre_lines: List[str]) -> str:
    """Return the expression to pass to the real HIP API during playback."""
    t = p.raw_type.strip()
    base = _get_base_type(t)
    is_output = _is_output_ptr_param(p)

    # Output pointer — declare local, pass address of local
    if is_output:
        # Check all known handle types
        for htype in _PLAYBACK_HANDLE_INFO:
            if htype in t:
                null_val = '0' if htype in ('hipSurfaceObject_t', 'hipTextureObject_t') else 'nullptr'
                pre_lines.append(f"  {htype} _out_{name} = {null_val};")
                return f"&_out_{name}"
        if 'hipArray_t' in t:
            pre_lines.append(f"  hipArray_t _out_{name} = nullptr;")
            return f"&_out_{name}"
        if 'hipMipmappedArray_t' in t:
            pre_lines.append(f"  hipMipmappedArray_t _out_{name} = nullptr;")
            return f"&_out_{name}"
        # void** or other output ptr
        pre_lines.append(f"  void* _out_{name} = nullptr;")
        return f"(void**)&_out_{name}"

    # Handle types — translate recorded handle to live handle
    if base in _PLAYBACK_HANDLE_TRANSLATE:
        fn = _PLAYBACK_HANDLE_TRANSLATE[base]
        return f"({t}){fn}(a->{name})"
    # hipArray_const_t and hipMipmappedArray_const_t
    if 'hipArray' in base:
        return f"(hipArray_t)ctx.translate_array(a->{name})"
    if 'hipMipmappedArray' in base:
        return f"(hipMipmappedArray_t)ctx.translate_mipmapped(a->{name})"

    # Device pointer types
    if base == 'hipDeviceptr_t':
        return f"(hipDeviceptr_t)ctx.translate_ptr(a->{name})"

    # void* that looks like a device ptr (dst/src/devPtr naming)
    is_ptr = '*' in t
    if is_ptr and base == 'void':
        if any(kw in name.lower() for kw in ('dst', 'dev', 'ptr', 'buf', 'src')):
            return f"ctx.translate_ptr(a->{name})"

    # dim3 — reconstruct from x/y/z fields
    if base in ('dim3', 'uint3'):
        pre_lines.append(f"  dim3 _dim_{name}(a->{name}_x, a->{name}_y, a->{name}_z);")
        return f"_dim_{name}"

    # Non-castable struct types — pass zero-initialized local
    if base in ('hipIpcEventHandle_t', 'hipIpcMemHandle_t', 'hipPitchedPtr',
                'hipExtent', 'hipPos', 'hipMemLocation', 'hipChannelFormatDesc'):
        pre_lines.append(f"  {t} _s_{name}{{}};")
        return f"_s_{name}"

    # Function pointer types — cast from stored field (stored as 0, best effort)
    if base in ('hipStreamCallback_t', 'hipHostFn_t'):
        return f"({t})a->{name}"

    # Unhandled non-const pointer to a non-void scalar type — treat as output pointer.
    # The captured value is the original process address (invalid at replay).
    # Use a zero-initialised local so the call succeeds without crashing.
    is_ptr = '*' in t
    is_const_ptr = 'const' in t and is_ptr
    if is_ptr and not is_const_ptr and base != 'void':
        inner = t.replace('*', '', 1).strip()
        pre_lines.append(f"  {inner} _out_{name}{{}};")
        return f"&_out_{name}"

    # Scalar / enum / int-like handle — cast from stored field
    return f"({t})a->{name}"


def generate_playback_shim(entry: ApiEntry) -> str:
    """Generate playback function for one API."""
    sname = f"hrr_args_{entry.name}"
    fname = f"playback_{entry.name}"
    sig   = f"static hipError_t {fname}(PlaybackContext& ctx, const uint8_t* payload)"

    # No-op playback APIs: emit a one-time warning then return hipSuccess.
    # The static bool ensures the message fires once per process, not once per event,
    # so replays with thousands of events don't spam stderr.
    if entry.name in NOOP_PLAYBACK_APIS:
        return (f"static hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
                f"  (void)ctx; (void)payload;\n"
                f"  static bool warned = false;\n"
                f"  if (!warned) {{\n"
                f"    warned = true;\n"
                f"    fprintf(stderr, \"[HRR] NOOP playback handler called for {entry.name} — \"\n"
                f"            \"this API is not replayed; results may differ from capture.\\n\");\n"
                f"  }}\n"
                f"  return hipSuccess;\n"
                f"}}\n")

    # Manual playback APIs: emit an extern declaration only, body in hip_playback.cpp
    if entry.name in MANUAL_PLAYBACK_APIS:
        return (f"extern hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload);\n")

    lines = []
    lines.append(f"{sig} {{")

    if entry.table == "compiler":
        # Compiler APIs (hipRegisterFatBinary etc.) are replay no-ops
        lines.append(f"  (void)ctx; (void)payload;")
        lines.append(f"  return hipSuccess;")
        lines.append("}")
        return "\n".join(lines) + "\n"

    # payload points to the full hrr_args_* struct (header + fields).
    lines.append(f"  const auto* a = reinterpret_cast<const {sname}*>(payload);")

    # Check if this API creates/destroys allocs or handles
    is_alloc_create  = entry.name in _ALLOC_CREATE_APIS
    is_alloc_free    = entry.name in _ALLOC_FREE_APIS
    is_hdl_create    = entry.name in _HANDLE_CREATE_APIS
    is_hdl_destroy   = entry.name in _HANDLE_DESTROY_APIS

    # For alloc-free and handle-destroy APIs: grab the recorded key before the call
    if is_alloc_free:
        rec_param = _ALLOC_FREE_APIS[entry.name]
        lines.append(f"  uint64_t _rec_ptr = a->{rec_param};")
        lines.append(f"  void*    _live_ptr = ctx.translate_ptr(_rec_ptr);")
    if is_hdl_destroy:
        rec_param, hdl_type = _HANDLE_DESTROY_APIS[entry.name]
        translate_fn = _PLAYBACK_HANDLE_TRANSLATE.get(hdl_type, '')
        if translate_fn:
            lines.append(f"  uint64_t _rec_hdl = a->{rec_param};")

    # Build argument list for the real call
    pre_lines: List[str] = []
    call_args: List[str] = []
    unnamed = 0
    for p in entry.params:
        name = p.name or f"p{unnamed}"
        if not p.name: unnamed += 1

        # For alloc-free: replace the pointer arg with the translated live ptr
        if is_alloc_free and name == _ALLOC_FREE_APIS[entry.name]:
            call_args.append("_live_ptr")
            continue
        # For handle-destroy: replace the handle arg with the live handle
        if is_hdl_destroy and name == _HANDLE_DESTROY_APIS[entry.name][0]:
            hdl_type = _HANDLE_DESTROY_APIS[entry.name][1]
            translate_fn = _PLAYBACK_HANDLE_TRANSLATE.get(hdl_type, '')
            if translate_fn:
                call_args.append(f"({hdl_type}){translate_fn}(_rec_hdl)")
            else:
                call_args.append(f"({hdl_type})a->{name}")
            continue

        call_args.append(_playback_arg(p, name, pre_lines))

    # Emit pre-call locals
    for pl in pre_lines:
        lines.append(pl)

    # Build the call
    args_str = ", ".join(call_args)
    void_ret = _is_void_return(entry.ret_type)
    if void_ret:
        lines.append(f"  {entry.name}({args_str});")
        ret_expr = "hipSuccess"
    else:
        lines.append(f"  hipError_t _r = (hipError_t){entry.name}({args_str});")
        ret_expr = "_r"

    # Post-call: register/unregister allocs and handles
    success_cond = "true" if void_ret else "_r == hipSuccess"

    if is_alloc_create:
        rec_param, sz_param = _ALLOC_CREATE_APIS[entry.name]
        # The output ptr is stored in a local _out_{rec_param} by _playback_arg
        lines.append(f"  if ({success_cond}) {{")
        lines.append(f"    ctx.record_alloc(a->{rec_param}, _out_{rec_param},"
                     f" static_cast<size_t>(a->{sz_param}));")
        lines.append(f"  }}")
    elif is_alloc_free:
        lines.append(f"  if ({success_cond}) {{")
        lines.append(f"    ctx.remove_alloc(_rec_ptr);")
        lines.append(f"  }}")

    if is_hdl_create:
        create_pairs = _HANDLE_CREATE_APIS[entry.name]
        lines.append(f"  if ({success_cond}) {{")
        for (param_name, hdl_type) in create_pairs:
            _, record_fn, _ = _PLAYBACK_HANDLE_INFO.get(hdl_type, ('', '', ''))
            if record_fn:
                lines.append(f"    {record_fn}(a->{param_name},"
                             f" _out_{param_name});")
        lines.append(f"  }}")
    elif is_hdl_destroy:
        rec_param2, hdl_type2 = _HANDLE_DESTROY_APIS[entry.name]
        _, _, remove_fn = _PLAYBACK_HANDLE_INFO.get(hdl_type2, ('', '', ''))
        if remove_fn:
            lines.append(f"  if ({success_cond}) {{")
            lines.append(f"    {remove_fn}(_rec_hdl);")
            lines.append(f"  }}")

    lines.append(f"  return {ret_expr};")
    lines.append("}")
    return "\n".join(lines) + "\n"


def generate_dispatch_table(entries: List[ApiEntry]) -> str:
    """Generate the hrr_playback_dispatch array indexed by hrr_api_id_t."""
    lines = []
    lines.append("// ============================================================")
    lines.append("// Minimum payload size per event type — indexed by hrr_api_id_t")
    lines.append("// dispatch_event() checks raw_payload.size() against this before")
    lines.append("// calling any handler to prevent OOB casts on malformed archives.")
    lines.append("// ============================================================")
    lines.append("const uint32_t hrr_api_min_payload_size[HRR_API_COUNT] = {")
    for idx, e in enumerate(entries):
        enum_name = "HRR_API_" + e.name.lstrip('_').upper()
        lines.append(f"    static_cast<uint32_t>(sizeof(hrr_args_{e.name})),  // [{idx}] {enum_name}")
    lines.append("};")
    lines.append("")
    lines.append("// ============================================================")
    lines.append("// Playback dispatch table — indexed by hrr_api_id_t")
    lines.append("// ============================================================")
    lines.append("hrr_playback_fn_t hrr_playback_dispatch[HRR_API_COUNT] = {")
    for idx, e in enumerate(entries):
        enum_name = "HRR_API_" + e.name.lstrip('_').upper()
        lines.append(f"    playback_{e.name},  // [{idx}] {enum_name}")
    lines.append("};")
    return "\n".join(lines) + "\n"


def generate_playback_cpp(entries: List[ApiEntry]) -> str:
    """Generate hip_playback_generated.cpp."""
    parts = [_PLAYBACK_CPP_PREAMBLE]

    parts.append("// ============================================================")
    parts.append("// Playback shims")
    parts.append("// ============================================================")
    parts.append("")

    for e in entries:
        parts.append(generate_playback_shim(e))

    parts.append("")
    parts.append(generate_dispatch_table(entries))

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # Script lives at hipamd/src/hrr/ — three levels up is clr/
    hrr_dir  = Path(__file__).resolve().parent          # hipamd/src/hrr/
    clr_root = hrr_dir.parent.parent.parent             # clr/
    default_input    = clr_root / "hipamd/include/hip/amd_detail/hip_api_trace.hpp"
    default_header   = hrr_dir  / "hrr_api_args.h"
    default_capture  = hrr_dir  / "hip_capture_generated.cpp"
    default_playback = hrr_dir  / "playback/hip_playback_generated.cpp"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",           default=str(default_input),
                        help="Path to hip_api_trace.hpp")
    parser.add_argument("--output-header",   default=str(default_header),
                        help="Path to generated hrr_api_args.h")
    parser.add_argument("--output-capture",  default=str(default_capture),
                        help="Path to generated hip_capture_generated.cpp")
    parser.add_argument("--output-playback", default=str(default_playback),
                        help="Path to generated hip_playback_generated.cpp")
    args = parser.parse_args()

    in_path       = Path(args.input)
    header_path   = Path(args.output_header)
    capture_path  = Path(args.output_capture)
    playback_path = Path(args.output_playback)

    if not in_path.exists():
        sys.exit(f"ERROR: input file not found: {in_path}")

    print(f"Parsing {in_path} ...")
    entries = parse_hip_api_trace(in_path)
    n_compiler       = sum(1 for e in entries if e.table == "compiler")
    n_runtime        = sum(1 for e in entries if e.table == "runtime")
    n_manual_cap     = sum(1 for e in entries if e.name in MANUAL_CAPTURE_APIS)
    n_manual_play    = sum(1 for e in entries if e.name in MANUAL_PLAYBACK_APIS)
    n_noop_play      = sum(1 for e in entries if e.name in NOOP_PLAYBACK_APIS)
    print(f"  Found {n_compiler} compiler + {n_runtime} runtime = {len(entries)} total")

    # -------------------------------------------------------------------------
    # Cross-validate classification sets against parsed API names.
    # Entries that don't exist in hip_api_trace.hpp are silent dead weight —
    # catch them here so a typo or removed API is flagged immediately.
    # -------------------------------------------------------------------------
    parsed_names: Set[str] = {e.name for e in entries}
    unknown: Dict[str, List[str]] = {}
    for set_name, api_set in [
        ("MANUAL_CAPTURE_APIS",  MANUAL_CAPTURE_APIS),
        ("MANUAL_PLAYBACK_APIS", MANUAL_PLAYBACK_APIS),
        ("NOOP_PLAYBACK_APIS",   NOOP_PLAYBACK_APIS),
        ("EXTRA_FIELDS",         set(EXTRA_FIELDS.keys())),
    ]:
        bad = sorted(n for n in api_set if n not in parsed_names)
        if bad:
            unknown[set_name] = bad
    if unknown:
        print("\nERROR: the following entries are not present in the parsed API list:")
        for set_name, names in unknown.items():
            for n in names:
                print(f"  {set_name}: '{n}'")
        sys.exit(1)
    print(f"  Manual capture (hand-written in hip_capture.cpp):  {n_manual_cap}")
    print(f"  Manual playback (hand-written in hip_playback.cpp): {n_manual_play}")
    print(f"  No-op playback (inline hipSuccess stubs):           {n_noop_play}")
    print(f"  Generated capture shims:  {len(entries) - n_manual_cap}")
    print(f"  Generated playback shims: {len(entries) - n_manual_play - n_noop_play}")

    header_path.parent.mkdir(parents=True, exist_ok=True)
    header = generate_header(entries)
    header_path.write_text(header, encoding='utf-8')
    print(f"Written header   -> {header_path}")

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_cpp = generate_capture_cpp(entries)
    capture_path.write_text(capture_cpp, encoding='utf-8')
    print(f"Written capture  -> {capture_path}")

    playback_path.parent.mkdir(parents=True, exist_ok=True)
    playback_cpp = generate_playback_cpp(entries)
    playback_path.write_text(playback_cpp, encoding='utf-8')
    print(f"Written playback -> {playback_path}")

    # Spot-check a few important structs
    _spot_check(entries)


def _spot_check(entries: List[ApiEntry]) -> None:
    check_names = {
        "__hipPushCallConfiguration",
        "__hipRegisterFatBinary",
        "__hipRegisterFunction",
        "hipMalloc",
        "hipMemcpy",
        "hipModuleLoadData",
        "hipModuleLaunchKernel",
        "hipSetDevice",
        "hipMemPoolCreate",
        "hipDeviceSetMemPool",
    }
    by_name = {e.name: e for e in entries}
    print("\nSpot-check:")
    for name in sorted(check_names):
        e = by_name.get(name)
        if not e:
            print(f"  MISSING: {name}")
            continue
        param_str = ", ".join(
            f"{normalise_field_type(p.raw_type)} {p.name}" for p in e.params
        )
        extra = EXTRA_FIELDS.get(name, [])
        extra_str = (", [EXTRA: " + ", ".join(f"{ft} {fn}" for ft, fn, _ in extra) + "]") if extra else ""
        manual_cap  = " [MANUAL_CAP]"  if name in MANUAL_CAPTURE_APIS  else ""
        manual_play = " [MANUAL_PLAY]" if name in MANUAL_PLAYBACK_APIS else ""
        noop_play   = " [NOOP_PLAY]"   if name in NOOP_PLAYBACK_APIS   else ""
        print(f"  {name}({param_str}) -> {e.ret_type}{extra_str}{manual_cap}{manual_play}{noop_play}")


if __name__ == "__main__":
    main()
