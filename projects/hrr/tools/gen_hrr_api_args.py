#!/usr/bin/env python3
"""
gen_hrr_api_args.py — Generate hrr_api_args.h, hip_capture_generated.cpp,
                      and hip_playback_generated.cpp from hip_api_trace.hpp.

Usage:
    python gen_hrr_api_args.py [--input HIP_API_TRACE_HPP]
                               [--public-header HIP_RUNTIME_API_H]
                               [--output-header HRR_API_ARGS_H]
                               [--output-capture HIP_CAPTURE_GENERATED_CPP]
                               [--output-playback HIP_PLAYBACK_GENERATED_CPP]

Defaults (paths relative to this script at projects/hrr/tools/):
    input            : ../../clr/hipamd/include/hip/amd_detail/hip_api_trace.hpp
    public-header    : ../../hip/include/hip/hip_runtime_api.h
    output-header    : ../include/hrr/hrr_api_args.h
    output-capture   : ../../clr/hipamd/src/hrr/hip_capture_generated.cpp
    output-playback  : ../playback/hip_playback_generated.cpp

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



MANUAL_CAPTURE_APIS: Set[str] = {
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    "hipLaunchKernel_spt",
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernel_spt",
    # Launch descriptor and attribute list are pointers carried in the variable-length payload.
    "hipDrvLaunchKernelEx",
    "hipLaunchKernelExC",
    "hipModuleLaunchCooperativeKernel",
    # Saves grid, block, shared size and stream in TLS for hipLaunchByPtr.
    "__hipPushCallConfiguration",
    "hipModuleLoadData",
    "hipModuleLoadDataEx",
    "hipModuleLoad",
    # Drops the program's code-object hash so a recycled amd::Program address cannot hit a stale g_prog_hash entry.
    "hipModuleUnload",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemcpyHtoD",
    "hipMemcpyHtoDAsync",
    "hipMemcpyWithStream",
    "hipMemcpyDtoH",
    "hipMemcpyDtoHAsync",
    "__hipRegisterFatBinary",
    # Records *modules before the real call.
    "__hipUnregisterFatBinary",
    "hipHostRegister",
    "hipHostUnregister",
    "hipMemPoolCreate",
    "hipMemPoolSetAttribute",
    "hipMemcpy3D",
    "hipMemcpy3DAsync",
    "hipMemcpy3D_spt",
    "hipMemcpy3DAsync_spt",
    "hipMemcpy2D",
    "hipMemcpy2DAsync",
    "hipArrayCreate",
    "hipArray3DCreate",
    "hipStreamSetAttribute",
    "hipMemGetAllocationGranularity",
    "hipMemPoolSetAccess",
    "hipMemSetAccess",
    "hipDrvMemcpy3D",
    "hipDrvMemcpy3DAsync",
    "hipDrvMemcpy2DUnaligned",
    "hipMemcpyParam2D",
    "hipMemcpyParam2DAsync",
    # hipKernelNodeParams names a host function and points at a host argument array.
    "hipGraphAddKernelNode",
    "hipGraphKernelNodeSetParams",
    "hipGraphExecKernelNodeSetParams",
    "hipGraphAddBatchMemOpNode",
    "hipGraphBatchMemOpNodeSetParams",
    "hipGraphExecBatchMemOpNodeSetParams",
    # The host buffer behind the symbol is a blob, present only for some copy kinds.
    "hipGraphAddMemcpyNodeToSymbol",
    "hipGraphAddMemcpyNodeFromSymbol",
    "hipGraphMemcpyNodeSetParamsToSymbol",
    "hipGraphExecMemcpyNodeSetParamsToSymbol",
    "hipLinkAddData",
}

MANUAL_APIS = MANUAL_CAPTURE_APIS

# Recorded as variable-length binary by serialize_kernel_launch, not as sizeof(args struct).
VARIABLE_LENGTH_KERNEL_LAUNCH_APIS: Set[str] = {
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    "hipLaunchKernel_spt",
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernel_spt",
    "hipDrvLaunchKernelEx",
    "hipLaunchKernelExC",
    "hipModuleLaunchCooperativeKernel",
}

# Minimum variable-length launch payload: header + stream(8) + name_len(2) + co_hash(16)
# + grid/block/shared(28) + counts(4). Kernel name and argument data add more.
_VARIABLE_KERNEL_LAUNCH_MIN_PAYLOAD_EXPR = (
    "static_cast<uint32_t>(sizeof(hrr_event_header) + 8u + 2u + 16u + 12u + 12u + 4u + 2u + 2u)"
)

# hipModuleGetFunction needs no capture: function handles are identified by name at launch time.
PASSTHROUGH_ONLY: Set[str] = {
    "hipModuleGetFunction",
}

# amdhip.def.in is the Windows amdhip64 export list. An API missing from it has no import library entry, so hrr-playback.exe cannot link it.
# The list is read rather than hard-coded, so the guard disappears once clr exports the symbol.

_AMDHIP_DEF_IN = (Path(__file__).resolve().parent.parent.parent
                  / "clr" / "hipamd" / "src" / "amdhip.def.in")


def _windows_exported_symbols() -> Set[str]:
    if not _AMDHIP_DEF_IN.is_file():
        sys.exit(f"ERROR: Windows export list not found at {_AMDHIP_DEF_IN}. "
                 f"The playback shims need it to know which APIs cannot be "
                 f"called on Windows.")
    out: Set[str] = set()
    for line in _AMDHIP_DEF_IN.read_text().splitlines():
        sym = line.strip()
        if sym and sym != "EXPORTS" and not sym.startswith(";"):
            out.add(sym)
    return out


WINDOWS_EXPORTED_APIS: Set[str] = _windows_exported_symbols()



MANUAL_PLAYBACK_APIS: Set[str] = {
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    "hipLaunchKernel_spt",
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernel_spt",
    "hipDrvLaunchKernelEx",
    "hipLaunchKernelExC",
    # Must go through the cooperative entry point, or a grid-wide barrier hangs.
    "hipModuleLaunchCooperativeKernel",
    "hipMemcpy",
    "hipMemcpyAsync",
    "hipMemcpyHtoD",
    "hipMemcpyHtoDAsync",
    "hipMemcpyWithStream",
    "hipModuleLoadData",
    "hipModuleLoadDataEx",
    "hipModuleLoad",
    "hipModuleGetFunction",
    "hipMalloc",
    "hipMallocAsync",
    "hipMallocFromPoolAsync",
    "hipMallocManaged",
    "hipFree",
    "hipFreeAsync",
    "hipStreamCreate",
    "hipStreamCreateWithFlags",
    "hipStreamCreateWithPriority",
    "hipStreamDestroy",
    "hipEventCreate",
    "hipEventCreateWithFlags",
    "hipEventDestroy",
    # Archives written before the marker events were recorded lack a hipEventRecord for the pair, and HIP returns hipErrorInvalidHandle.
    # See playback_hipEventElapsedTime.
    "hipEventElapsedTime",
    # Capture logs only hipSuccess for queries. Replay may reach the call sooner, so it spins on hipErrorNotReady.
    # See DESIGN.md § HIP-Specific Polling.
    "hipEventQuery",
    "hipStreamQuery",
    "hipStreamQuery_spt",
    "__hipRegisterFatBinary",
    # Maps the host shadow and the capture-time device address onto the symbol found by name in the loaded modules.
    "__hipRegisterVar",
    # Symbol queries — answered from that map rather than from a host shadow
    # address this process does not have.
    "hipGetSymbolAddress",
    "hipGetSymbolSize",
    # Charged to the host budget (--max-host-bytes). A replaced entry is freed first; hipFreeHost and hipHostFree release it.
    "hipHostMalloc",
    "hipMallocHost",
    "hipFreeHost",
    "hipHostFree",
    "hipHostRegister",
    "hipHostUnregister",
    "hipHostGetDevicePointer",
    "hipMemPoolCreate",
    "hipMemPoolSetAttribute",
    "hipMemPoolGetAttribute",
    # hipStreamEndCapture records the output graph handle; hipGraphInstantiate uses the recorded graph and records the exec handle.
    # hipGraphLaunch needs offset-correct rd64/rd32 reads, since reinterpret_cast is broken.
    "hipStreamBeginCapture",
    "hipStreamEndCapture",
    "hipGraphInstantiate",
    # Fails loudly instead of passing a null graph to HIP when the graph came from the unsupported explicit node API.
    "hipGraphInstantiateWithFlags",
    "hipGraphLaunch",
    "hipMemcpyDtoH",
    "hipMemcpyDtoHAsync",
    "hipMemcpy3D",
    "hipMemcpy3DAsync",
    "hipMemcpy3D_spt",
    "hipMemcpy3DAsync_spt",
    # H2D substitutes the captured blob because the host source is a stale address.
    # D2H checks the device result against the captured expected blob.
    "hipMemcpy2D",
    "hipMemcpy2DAsync",
    "hipArrayCreate",
    "hipArray3DCreate",
    # Skips handles missing from array_map, such as arrays from the no-op hipMallocArray.
    "hipFreeArray",
    "hipStreamSetAttribute",
    "hipMemGetAllocationGranularity",
    "hipMemPoolSetAccess",
    "hipMemSetAccess",
    # The recorded op list needs a per-op device-pointer rewrite that depends on each entry's op type.
    "hipStreamBatchMemOp",
    # A legacy archive stores the handle bytes; replay pairs the live handle with the recorded one so a later import resolves.
    # Capture now leaves the field zero, which pairs with nothing.
    "hipIpcGetMemHandle",
    "hipIpcOpenMemHandle",
    "hipMemAddressReserve",
    "hipMemAddressFree",
    "hipMemCreate",
    "hipMemRelease",
    "hipMemMap",
    "hipMemUnmap",
    # Allocates a real buffer in alloc_map, with padding and zero-init matching hipMalloc.
    "hipExtMallocWithFlags",
    "hipDrvMemcpy3D",
    "hipDrvMemcpy3DAsync",
    "hipDrvMemcpy2DUnaligned",
    "hipMemcpyParam2D",
    "hipMemcpyParam2DAsync",
    # Each operand's address is in a union member chosen by the tag beside it.
    "hipMemcpy3DBatchAsync",
    "hipGraphAddKernelNode",
    "hipGraphKernelNodeSetParams",
    "hipGraphExecKernelNodeSetParams",
    # Per-op device address rewrite, as in hipStreamBatchMemOp.
    "hipGraphAddBatchMemOpNode",
    "hipGraphBatchMemOpNodeSetParams",
    "hipGraphExecBatchMemOpNodeSetParams",
    "hipGraphAddMemcpyNodeToSymbol",
    "hipGraphAddMemcpyNodeFromSymbol",
    "hipGraphMemcpyNodeSetParamsToSymbol",
    "hipGraphMemcpyNodeSetParamsFromSymbol",
    "hipGraphExecMemcpyNodeSetParamsToSymbol",
    "hipGraphExecMemcpyNodeSetParamsFromSymbol",
    "hipLinkAddData",
    # The address comes from the replay's pool, so it must land in alloc_map under the recorded one.
    "hipGraphAddMemAllocNode",
    # HIP_MEMCPY3D tells host from device memory by a member, and a host operand was never recorded.
    "hipDrvGraphAddMemcpyNode",
    "hipDrvGraphMemcpyNodeSetParams",
    "hipDrvGraphExecMemcpyNodeSetParams",
}

# Graph work whose arguments the archive cannot rebuild: a semaphore only another process can produce, a device symbol with no recorded name, a union with no recoverable active member.
# Handlers return hipSuccess but call ctx.mark_graph_incomplete, so hipGraphInstantiate and hipGraphInstantiateWithFlags refuse the graph with hipErrorNotSupported.
# Only calls that build or mutate graph contents belong here; harmless queries stay in NOOP_PLAYBACK_APIS.
ERROR_STUB_PLAYBACK_APIS: Set[str] = {
    "hipGraphAddExternalSemaphoresSignalNode",
    "hipGraphAddExternalSemaphoresWaitNode",
    # hipGraphNodeParams is a union over every node kind, with its active member known only from a type tag.
    # The typed spellings above are replayed instead.
    "hipGraphAddNode",
    # hipGraphInstantiateParams carries an out node pointer and an error-log buffer of the capturing process.
    "hipGraphInstantiateWithParams",
}

# APIs whose effect replay cannot reproduce: host callbacks, fabric/IPC handles and multi-device launch descriptors point into the capturing process.
# Each gets a handler that returns hipErrorNotSupported (fatal unless --continue-on-error) and is listed in the replay summary; the capture shim also warns once.
# Maps API name to the reason, printed verbatim at both ends.
_HOST_CALLBACK_REASON = (
    "the callback is a host function pointer belonging to the capturing "
    "process, so there is no function here to enqueue"
)

_MULTI_DEVICE_LAUNCH_REASON = (
    "each hipLaunchParams entry names its kernel by a host function address in "
    "the capturing process, and a cooperative multi-device launch cannot be "
    "decomposed into per-device launches without breaking the grid-wide "
    "barrier it exists for"
)

_SHAREABLE_IMPORT_REASON = (
    "the recorded argument is an OS handle (a POSIX fd or a Win32 HANDLE) "
    "belonging to the process that exported it, and the same number in the "
    "replaying process names a different object or nothing at all"
)

UNREPLAYABLE_PLAYBACK_APIS: Dict[str, str] = {
    "hipLaunchHostFunc":        _HOST_CALLBACK_REASON,
    "hipLaunchHostFunc_spt":    _HOST_CALLBACK_REASON,
    "hipStreamAddCallback":     _HOST_CALLBACK_REASON,
    "hipStreamAddCallback_spt": _HOST_CALLBACK_REASON,
    "hipLaunchCooperativeKernelMultiDevice": _MULTI_DEVICE_LAUNCH_REASON,
    "hipExtLaunchMultiKernelMultiDevice":    _MULTI_DEVICE_LAUNCH_REASON,
    # A host node is the graph form of the same problem: a function pointer and userData into the capturing process.
    # The graph is marked incomplete so instantiation refuses it.
    "hipGraphAddHostNode":            _HOST_CALLBACK_REASON,
    "hipGraphHostNodeSetParams":      _HOST_CALLBACK_REASON,
    "hipGraphExecHostNodeSetParams":  _HOST_CALLBACK_REASON,
    # Replay re-runs the matching export into its own buffer, so the exported handle is new and differs from the recorded one.
    # Passing the recorded number crashed hipMemPoolImportFromShareableHandle in the pool import, so replay refuses it by name.
    "hipMemImportFromShareableHandle":     _SHAREABLE_IMPORT_REASON,
    "hipMemPoolImportFromShareableHandle": _SHAREABLE_IMPORT_REASON,
    # A user object exists to run its destructor, a hipHostFn_t in the capturing process.
    # Creating one without it would silently drop that.
    "hipUserObjectCreate": _HOST_CALLBACK_REASON,
}


# APIs replay refuses that capture does not annotate.
# UNREPLAYABLE_PLAYBACK_APIS also makes the capture shim call writer::note_unreplayable(), which changes a file outside projects/hrr; this table keeps the change inside it.
_UNTRANSLATED_HANDLE_REASON = (
    "the recorded argument is a handle recorded as an address from the "
    "capturing process; replay has no translation for it, so passing it to "
    "HIP would let the archive choose what the runtime dereferences"
)

PLAYBACK_ONLY_UNREPLAYABLE_APIS: Dict[str, str] = {
    "hipLinkAddFile": (
        "the recorded path names a file the archive chose, and replay would "
        "open whatever that names on this machine; an archive must not decide "
        "which file the replayer reads"),
    # These take an opaque handle that replay does not translate, so they are refused rather than handed to HIP.
    "hipDestroyExternalMemory": _UNTRANSLATED_HANDLE_REASON,
    "hipDestroyExternalSemaphore": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxDestroy": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxGetDevResource": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxGetDevice": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxGetId": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxRecordEvent": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxStreamCreate": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxSynchronize": _UNTRANSLATED_HANDLE_REASON,
    "hipExecutionCtxWaitEvent": _UNTRANSLATED_HANDLE_REASON,
    "hipExternalMemoryGetMappedBuffer": _UNTRANSLATED_HANDLE_REASON,
    "hipExternalMemoryGetMappedMipmappedArray": _UNTRANSLATED_HANDLE_REASON,
    "hipGraphicsResourceGetMappedPointer": _UNTRANSLATED_HANDLE_REASON,
    "hipGraphicsSubResourceGetMappedArray": _UNTRANSLATED_HANDLE_REASON,
    "hipGraphicsUnregisterResource": _UNTRANSLATED_HANDLE_REASON,
    "hipGreenCtxCreate": _UNTRANSLATED_HANDLE_REASON,
    "hipImportExternalMemory": _UNTRANSLATED_HANDLE_REASON,
    "hipImportExternalSemaphore": _UNTRANSLATED_HANDLE_REASON,
    "hipLibraryGetGlobal": _UNTRANSLATED_HANDLE_REASON,
    "hipLibraryGetManaged": _UNTRANSLATED_HANDLE_REASON,
    "hipSignalExternalSemaphoresAsync": _UNTRANSLATED_HANDLE_REASON,
    "hipWaitExternalSemaphoresAsync": _UNTRANSLATED_HANDLE_REASON,
}

# Filled by generate_playback_shim with APIs refused because an argument would reach HIP as a raw cast (see RawSlotError).
# The value is the reason.
AUTO_UNREPLAYABLE_PLAYBACK_APIS: Dict[str, str] = {}

# APIs taking a JIT/link option array. Option values are void*: an integer for some options, a pointer into the capturing process for others.
# The handler refuses when any recorded option carries a pointer (hrr_jit_options.h), reading _d_options and _d_options_n from the deref table.
# Other option-array APIs are hand-written, refused, or ignore the options.
JIT_OPTION_ARRAY_APIS: Set[str] = {"hipLinkCreate"}

# Handlers whose record carries hipKernelNodeAttrValue by value. Its access policy window holds a capturing-process device address (base_ptr).
# The handler translates base_ptr when the attribute id selects the window and passes other attributes through. CLR rejects only num_bytes above accessPolicyMaxWindowSize (0 on AMD), so recorded windows have num_bytes == 0 and replay.
# Maps the API to (attribute-id field, enumerator that selects the window).
ATTR_UNION_WINDOW_APIS: Dict[str, Tuple[str, str]] = {
    "hipGraphKernelNodeSetAttribute": ("attr", "hipKernelNodeAttributeAccessPolicyWindow"),
}

# APIs given an inline no-op playback body instead of an extern declaration: missing from the SDK headers, or output-parameter types the generator cannot cast.
NOOP_PLAYBACK_APIS: Set[str] = {
    "hipExtHostAlloc",
    "hipGLGetDevices",
    "hipGraphicsGLRegisterBuffer",
    "hipGraphicsGLRegisterImage",
    "hipHccModuleLaunchKernel",
    "hipOccupancyMaxActiveClusters",
    "hipOccupancyMaxPotentialClusterSize",
    "hipPointerSetAttribute",
    "hipMemAllocPitch",
    "hipMemGetAddressRange",
    "hipModuleGetGlobal",
    "hipTexRefGetAddress",
    "hipTexRefGetFormat",
    "hipTexRefSetFormat",
    "hipMemMapArrayAsync",
    "hipMipmappedArrayGetMemoryRequirements",
    "hipGraphExecUpdate",
    "hipGraphGetEdges",
    "hipGraphGetNodes",
    "hipGraphGetRootNodes",
    "hipGraphNodeGetDependencies",
    "hipGraphNodeGetDependentNodes",
    "hipGraphRemoveDependencies",
    "hipStreamGetCaptureInfo_v2",
    "hipStreamGetCaptureInfo_v2_spt",
    "hipStreamUpdateCaptureDependencies",
    # hipCtxGetDevice stays a no-op: it writes a hipDevice_t replay does not use, and hipSetDevice already chose the device.
    "hipCtxGetDevice",
    "hipDeviceGet",
    "hipDrvGetErrorName",
    "hipDrvGetErrorString",
    "hipGetTextureReference",
    "hipKernelGetName",
    "hipMemRetainAllocationHandle",
    "hipMemGetAllocationPropertiesFromHandle",
    "hipModuleGetTexRef",
    "hipStreamGetDevice",
    "hipDeviceGetByPCIBusId",
    # These write into a caller-sized char buffer; the generated `char _out{}` is 1 byte and smashes the stack on Linux.
    "hipDeviceGetName",
    "hipDeviceGetPCIBusId",
    "hipDeviceGetGraphMemAttribute",
    "hipDeviceSetGraphMemAttribute",
    "hipDevicePrimaryCtxGetState",
    "hipPointerGetAttribute",
    "hipDrvPointerGetAttributes",
    "hipMemPoolGetAccess",
    "hipMemPoolExportPointer",
    "hipMemPoolImportPointer",
    "hipMallocArray",
    "hipMalloc3DArray",
    "hipMalloc3D",
    # hipMallocHost and hipFreeHost are replayed by hand (MANUAL_PLAYBACK_APIS).
    "hipMemAllocHost",
    "hipHostAlloc",
    "hipMemAllocPitch",
    "hipHostGetFlags",
    # H2D data is handled by the hipMemcpy blobs; these have no blob fields.
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
    "hipMemcpyToArray",
    "hipMemcpyFromArray",
    "hipMemcpy2DToArray",
    "hipMemcpy2DFromArray",
    "hipMemcpy2DToArrayAsync",
    "hipMemcpy2DFromArrayAsync",
    "hipMemcpyAtoH",
    "hipMemcpyHtoA",
    "hipMemcpyToSymbol",
    "hipMemcpyFromSymbol",
    "hipMemcpyToSymbolAsync",
    "hipMemcpyFromSymbolAsync",
    "hipMemset3D",
    "hipMemset3DAsync",
    "hipMemset3D_spt",
    "hipMemset3DAsync_spt",
    # hipExtMallocWithFlags is a real device allocation handled manually, so its pointer lands in alloc_map.
    "hipGetDevicePropertiesR0000",
    "hipGetErrorName",
    "hipGetErrorString",
    "hipCreateChannelDesc",
    "hipApiName",
    "hipKernelNameRef",
    "hipKernelNameRefByPtr",
    "hipOccupancyMaxPotentialBlockSize",
    "hipOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "hipFuncSetCacheConfig",
    "hipFuncSetSharedMemConfig",
    "hipFuncGetAttributes",
    "hipFuncSetAttribute",
    "hipDeviceGetUuid",
    "hipDevicePrimaryCtxRelease",
    "hipDevicePrimaryCtxReset",
    "hipDevicePrimaryCtxSetFlags",
    "hipExtStreamCreateWithCUMask",
    "hipExtStreamGetCUMask",
    "hipExtGetLinkTypeAndHopCount",
    # hipStreamWaitValue32/64 stay no-ops: the wait is satisfied by a writer that may not be replayed, so replay could hang.
    # hipStreamWriteValue32/64 are replayed, translating ptr via ctx.translate_ptr and the stream via ctx.translate_stream.
    # An unmapped destination follows the policy in playback/hip_playback.h.
    "hipStreamWaitValue32",
    "hipStreamWaitValue64",
    "hipStreamAttachMemAsync",
    "hipGetStreamDeviceId",
    "hipCtxGetFlags",
    "hipCtxGetCacheConfig",
    "hipCtxGetSharedMemConfig",
    "hipCtxGetApiVersion",
    "hipCtxSetCurrent",
    "hipCtxEnablePeerAccess",
    "hipCtxDisablePeerAccess",
    "hipModuleUnload",
    "hipModuleGetFunctionCount",
    "hipModuleEnumerateFunctions",
    # The fat binary pointer is stale; the code object is already loaded via the registered binary.
    "hipModuleLoadFatBinary",
    "hipModuleOccupancyMaxPotentialBlockSize",
    "hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
    "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
    "hipFuncGetAttribute",
    "hipGetFuncBySymbol",
    "hipExtLaunchKernel",
    "hipLibraryLoadData",
    "hipLibraryLoadFromFile",
    "hipLibraryUnload",
    "hipLibraryGetKernel",
    "hipLibraryGetKernelCount",
    "hipLibraryGetModule",
    "hipLibraryEnumerateKernels",
    "hipKernelGetLibrary",
    "hipKernelGetFunction",
    "hipKernelGetParamInfo",
    "hipKernelGetAttribute",
    "hipKernelSetAttribute",
    # hipGetProcAddress and hipGetProcAddress_spt carry the symbol name and are looked up for real.
    "hipGetDriverEntryPoint",
    "hipGetDriverEntryPoint_spt",
    "hipOccupancyAvailableDynamicSMemPerBlock",
    "hipSetValidDevices",
    "hipMemPtrGetInfo",
    # Segfaults on Linux ROCm 7.13 with a stale dev_ptr.
    "hipMemRangeGetAttribute",
    "hipMemRangeGetAttributes",
    # void** dev_ptrs is a stale device address array, and the generated shim does not translate it.
    "hipMemDiscardBatchAsync",
    "hipDrvMemDiscardBatchAsync",
    "hipMemDiscardAndPrefetchBatchAsync",
    "hipDrvMemDiscardAndPrefetchBatchAsync",
    "hipMemcpy3DPeer",
    "hipMemcpy3DPeerAsync",
    "hipMemcpy2DArrayToArray",
    "hipMemcpyAtoD",
    "hipMemcpyDtoA",
    "hipMemcpyAtoA",
    "hipMemcpyAtoHAsync",
    "hipMemcpyHtoAAsync",
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
    # Node construction and mutation are replayed. The remaining no-ops are Get* queries, which change nothing in the replayed graph,
    # and calls whose arguments name state only the capturing process had.
    "hipGraphDestroy",
    "hipGraphUpload",
    "hipGraphDebugDotPrint",
    "hipGraphNodeGetType",
    "hipGraphNodeGetEnabled",
    "hipGraphKernelNodeGetParams",
    "hipGraphKernelNodeGetAttribute",
    "hipGraphMemcpyNodeGetParams",
    "hipGraphMemsetNodeGetParams",
    "hipGraphHostNodeGetParams",
    "hipGraphExecDestroy",
    "hipGraphExecGetFlags",
    # See hipGraphAddNode.
    "hipGraphNodeSetParams",
    "hipGraphExecNodeSetParams",
    "hipGraphChildGraphNodeGetGraph",
    "hipGraphEventRecordNodeGetEvent",
    "hipGraphEventWaitNodeGetEvent",
    "hipGraphMemAllocNodeGetParams",
    "hipGraphMemFreeNodeGetParams",
    "hipUserObjectRetain",
    "hipUserObjectRelease",
    "hipGraphRetainUserObject",
    "hipGraphReleaseUserObject",
    "hipGraphBatchMemOpNodeGetParams",
    "hipDrvGraphMemcpyNodeGetParams",
# In the internal dispatch table but not declared in the public headers playback builds against, so a direct call would not compile.
# hipDeviceGetLuid is Windows-only: a device-identity query, like the no-op hipDeviceGetUuid.
    "hipDeviceGetLuid",
    # Absent from the ROCm 7.15 nightly headers; the harness handles device initialization.
    "hipInitDevice",
    # The hipMemLocation* pointee cannot be rebuilt safely on all SDKs, and a null location can return hipErrorInvalidValue and abort replay.
    "hipMemGetDefaultMemPool",
}

# APIs passing an opaque handle playback does not translate, but already replaying through a generated call.
# A new API with such a parameter type replays as a no-op until added here or given another class.
# A hand-written handler belongs in MANUAL_PLAYBACK_APIS, not here.
DIRECT_PLAYBACK_APIS: Set[str] = {
    "hipDevResourceGenerateDesc",
    "hipDeviceGetExecutionCtx",
    "hipGraphicsMapResources",
    "hipGraphicsUnmapResources",
}


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
    "hipHostRegister":    [("uint64_t", "blob_hash_lo", "sysmem blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "sysmem blob hash hi")],
    # The to-symbol spellings read a recorded host buffer, so its bytes go in a blob.
    # The from-symbol spellings write one, and the blob is the expected output.
    "hipGraphAddMemcpyNodeToSymbol":
        [("uint64_t", "blob_hash_lo", "host src blob hash lo, 0 if src is a device pointer"),
         ("uint64_t", "blob_hash_hi", "host src blob hash hi")],
    "hipGraphAddMemcpyNodeFromSymbol":
        [("uint64_t", "blob_hash_lo", "unused; the destination is written at replay"),
         ("uint64_t", "blob_hash_hi", "unused")],
    "hipGraphMemcpyNodeSetParamsToSymbol":
        [("uint64_t", "blob_hash_lo", "host src blob hash lo, 0 if src is a device pointer"),
         ("uint64_t", "blob_hash_hi", "host src blob hash hi")],
    "hipGraphExecMemcpyNodeSetParamsToSymbol":
        [("uint64_t", "blob_hash_lo", "host src blob hash lo, 0 if src is a device pointer"),
         ("uint64_t", "blob_hash_hi", "host src blob hash hi")],
    # The linked code object can reach hundreds of kilobytes, so it is a blob.
    "hipLinkAddData":     [("uint64_t", "blob_hash_lo", "linker input image blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "linker input image blob hash hi")],
    # Device address of the symbol in the capturing process. The live shim records zero because it runs at static-init time before any device exists.
    # The post-registration sweep fills it in.
    "__hipRegisterVar":   [("uint64_t", "dev_addr",
                            "capture-time device address of the symbol, 0 if unresolved")],
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
    "hipMemcpyWithStream": [("uint64_t", "blob_hash_lo", "H2D blob hash lo"),
                            ("uint64_t", "blob_hash_hi", "H2D blob hash hi")],
    # sizeof(hipMemPoolProps) == 88 bytes (4+4+8+8+8+56 reserved).
    "hipMemPoolCreate": [("uint8_t", "pool_props_bytes[88]", "hipMemPoolProps inline copy")],
    # value is a void* to a uint64_t or uint32_t; 8 bytes are stored inline.
    "hipMemPoolSetAttribute": [("uint64_t", "value_u64", "attribute value stored inline")],
    "hipMemPoolGetAttribute": [("uint64_t", "value_u64", "attribute value stored inline (unused at capture)")],
    # sizeof(hipMemcpy3DParms) == 160 bytes.
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
    # The region is height rows of `width` bytes spaced by spitch/dpitch.
    # The blob holds those rows packed (width * height bytes), per HRR_FILE_FLAG_PACKED_HOST_RECTS.
    "hipMemcpy2D":      [("uint64_t", "blob_hash_lo", "H2D blob hash lo (0 if not H2D)"),
                         ("uint64_t", "blob_hash_hi", "H2D blob hash hi"),
                         ("uint64_t", "d2h_hash_lo",  "D2H expected-output blob hash lo (0 if not D2H)"),
                         ("uint64_t", "d2h_hash_hi",  "D2H expected-output blob hash hi")],
    "hipMemcpy2DAsync": [("uint64_t", "blob_hash_lo", "H2D blob hash lo (0 if not H2D)"),
                         ("uint64_t", "blob_hash_hi", "H2D blob hash hi"),
                         ("uint64_t", "d2h_hash_lo",  "D2H expected-output blob hash lo (0 if not D2H)"),
                         ("uint64_t", "d2h_hash_hi",  "D2H expected-output blob hash hi")],
    "hipDrvMemcpy3D":      [("uint8_t", "drv3d_bytes[192]", "HIP_MEMCPY3D inline copy"),
                            ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                            ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                            ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                            ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipDrvMemcpy3DAsync": [("uint8_t", "drv3d_bytes[192]", "HIP_MEMCPY3D inline copy"),
                            ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                            ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                            ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                            ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipDrvMemcpy2DUnaligned": [("uint8_t", "drv2d_bytes[136]", "hip_Memcpy2D inline copy"),
                            ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                            ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                            ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                            ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipMemcpyParam2D": [("uint8_t", "drv2d_bytes[136]", "hip_Memcpy2D inline copy"),
                            ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                            ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                            ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                            ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipMemcpyParam2DAsync": [("uint8_t", "drv2d_bytes[136]", "hip_Memcpy2D inline copy"),
                            ("uint64_t", "blob_hash_lo",    "H2D blob hash lo (0 if not H2D)"),
                            ("uint64_t", "blob_hash_hi",    "H2D blob hash hi"),
                            ("uint64_t", "d2h_hash_lo",     "D2H expected-output blob hash lo (0 if not D2H)"),
                            ("uint64_t", "d2h_hash_hi",     "D2H expected-output blob hash hi")],
    "hipArrayCreate":   [("uint8_t", "array_desc_bytes[24]", "HIP_ARRAY_DESCRIPTOR inline copy")],
    "hipArray3DCreate": [("uint8_t", "array3d_desc_bytes[40]", "HIP_ARRAY3D_DESCRIPTOR inline copy")],
    "hipStreamSetAttribute": [("uint8_t", "stream_attr_bytes[64]", "hipStreamAttrValue inline copy")],
    "hipMemGetAllocationGranularity": [("uint8_t", "alloc_prop_bytes[32]", "hipMemAllocationProp inline copy")],
    "hipMemPoolSetAccess": [("uint8_t", "access_desc_bytes[12]", "hipMemAccessDesc[0] inline copy")],
    "hipMemSetAccess":     [("uint8_t", "access_desc_bytes[12]", "hipMemAccessDesc[0] inline copy")],
}

# Dereferenced pointer arguments. normalise_field_type() lowers pointers to the capture-time address, so the pointee never reaches the archive.
# A Deref makes capture copy the pointee (`size` bytes, or `count` elements) and set <param>_present; playback rebuilds a local copy.
# A pointee holding device pointers, handles or nested pointers needs `playback="manual"` and a handler in MANUAL_PLAYBACK_APIS.

@dataclass(frozen=True)
class Deref:
    param: str            # parameter name in the hip_api_trace.hpp typedef
    ctype: str            # pointee type, e.g. "hipMemAllocationProp"
    size: int             # inline bytes per element; static_assert'd against sizeof(ctype)
    direction: str = "in"  # "in" | "out" | "inout"
    count: str = ""       # element-count parameter name (array arguments only)
    max_count: int = 0    # inline capacity in elements, for array/string arguments
    # A NUL-terminated string has no count or fixed size: capture copies up to max_count - 1 characters plus the terminator.
    string: bool = False
    playback: str = "auto"  # "auto" (generic reconstruction) | "manual"
    # Members holding recorded device addresses; replay runs each through ctx.translate_ptr.
    ptr_members: Tuple[str, ...] = ()
    # Members holding recorded HIP handles, as (member, handle type), translated through that type's map.
    handle_members: Tuple[Tuple[str, str], ...] = ()
    elem_handle: str = ""
    # Array of bare device addresses (a batch copy's dsts/srcs); each element goes through translate_ptr.
    elem_ptr: bool = False
    # Capability pointee such as an IPC handle: capture leaves the bytes zero and records only whether the pointer was non-null.
    capability: bool = False
    # Members naming the machine that the API never reads; capture zeroes them, so replay passes zeros.
    blank_members: Tuple[str, ...] = ()

    @property
    def is_array(self) -> bool:
        return bool(self.count)

    @property
    def bytes_field(self) -> str:
        return f"{self.param}_bytes"

    @property
    def present_field(self) -> str:
        return f"{self.param}_present"

    @property
    def count_field(self) -> str:
        return f"{self.param}_n"

    @property
    def total_bytes(self) -> int:
        if self.string:
            return self.max_count
        return self.size * self.max_count if self.is_array else self.size


DEREF_FIELDS: Dict[str, List[Deref]] = {
    # The allocation property decides the memory type and, for cross-node heaps, the source device.
    "hipMemCreate": [
        Deref("prop", "hipMemAllocationProp", 32, playback="manual"),
    ],
    # The device pointer sits in an op-type-dependent member, so reconstruction is hand-written.
    "hipStreamBatchMemOp": [
        Deref("paramArray", "hipStreamBatchMemOpParams", 48, count="count",
              max_count=16, playback="manual"),
    ],
    # The IPC handle is 64 bytes and lets any process on the machine open the allocation while its exporter lives, so the archive does not carry it.
    # An importer never has its export in the same archive.
    "hipIpcGetMemHandle": [
        Deref("handle", "hipIpcMemHandle_t", 64, direction="out",
              playback="manual", capability=True),
    ],
    # The location selects which device's view is queried.
    "hipMemGetAccess": [
        Deref("location", "hipMemLocation", 8),
        Deref("flags", "unsigned long long", 8, direction="out"),
    ],
    # The pre-chevron launch ABI pushes `size` bytes from a host address; 256 bytes covers any argument.
    "hipSetupArgument": [
        Deref("arg", "unsigned char", 1, count="size", max_count=256),
    ],
    # The symbol name is the whole question; a bare address caused a SIGSEGV in hip::hipGetProcAddress_common.
    "hipGetProcAddress": [
        Deref("symbol", "char", 1, max_count=256, string=True),
    ],
    "hipGetProcAddress_spt": [
        Deref("symbol", "char", 1, max_count=256, string=True),
    ],
    # deviceVar is the only part meaningful in another process (`var` is a host shadow, `modules` a fat-binary handle).
    # Replay resolves it with hipModuleGetGlobal and maps the recorded device address onto it.
    "__hipRegisterVar": [
        Deref("deviceVar", "char", 1, max_count=256, string=True),
    ],
    # 32 options is more than any caller passes.
    "hipLinkCreate": [
        Deref("options", "hipJitOption", 4, count="numOptions", max_count=32),
        Deref("optionValues", "void*", 8, count="numOptions", max_count=32),
    ],
    "hipLinkAddData": [
        Deref("name", "char", 1, max_count=256, string=True),
        Deref("options", "hipJitOption", 4, count="numOptions", max_count=32),
        Deref("optionValues", "void*", 8, count="numOptions", max_count=32),
    ],
    "hipLinkAddFile": [
        Deref("path", "char", 1, max_count=256, string=True),
        Deref("options", "hipJitOption", 4, count="numOptions", max_count=32),
        Deref("optionValues", "void*", 8, count="numOptions", max_count=32),
    ],
    # 1472 bytes is large, but hipChooseDevice runs once at startup. CLR matches only numeric limits, so UUID, LUID and PCI location are blanked.
    "hipChooseDevice": [
        Deref("prop", "hipDeviceProp_t", 1472,
              blank_members=("uuid", "luid", "pciDomainID", "pciBusID", "pciDeviceID")),
    ],
    "hipChooseDeviceR0000": [
        Deref("properties", "hipDeviceProp_tR0000", 792,
              blank_members=("pciDomainID", "pciBusID", "pciDeviceID")),
    ],
    "hipDeviceGetTexture1DLinearMaxWidth": [
        Deref("fmtDesc", "hipChannelFormatDesc", 20),
        Deref("maxWidthInElements", "size_t", 8, direction="out"),
    ],
    "hipGraphKernelNodeSetAttribute": [
        Deref("value", "hipKernelNodeAttrValue", 64),
    ],
    # The pool handle beside it translates through mempool_map.
    "hipMemSetMemPool": [
        Deref("location", "hipMemLocation", 8),
    ],
    "hipMemGetMemPool": [
        Deref("location", "hipMemLocation", 8),
    ],
    # Sixteen inline entries cover this suite's batches; capture warns beyond that.
    "hipMemcpyBatchAsync": [
        Deref("dsts", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("srcs", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("sizes", "size_t", 8, count="count", max_count=16),
        Deref("attrs", "hipMemcpyAttributes", 24, count="numAttrs",
              max_count=16),
        Deref("attrsIdxs", "size_t", 8, count="numAttrs", max_count=16),
        Deref("failIdx", "size_t", 8, direction="out"),
    ],
    # Adds swap-partner sizes and per-entry waits/signals, whose addr is a recorded device address.
    "hipExtMemcpyBatchAsync": [
        Deref("dsts", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("srcs", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("sizes", "size_t", 8, count="count", max_count=16),
        Deref("sizesDst", "size_t", 8, count="count", max_count=16),
        Deref("waits", "hipExtMemcpyWait", 64, count="count", max_count=16,
              ptr_members=("addr",)),
        Deref("signals", "hipExtMemcpySignal", 64, count="count", max_count=16,
              ptr_members=("addr",)),
        Deref("attrs", "hipExtMemcpyAttributes", 56, count="numAttrs",
              max_count=16),
        Deref("attrsIdxs", "size_t", 8, count="numAttrs", max_count=16),
    ],
    # Each operand is a tagged union of a pointer and an array handle, so the address member depends on the tag.
    "hipMemcpy3DBatchAsync": [
        Deref("opList", "hipMemcpy3DBatchOp", 112, count="numOps",
              max_count=16, playback="manual"),
        Deref("failIdx", "size_t", 8, direction="out", playback="manual"),
    ],
}

# Every hipGraphAdd*Node call takes a dependency array of hipGraphNode_t and a node-parameter struct.
# The array is an elem_handle Deref; parameter structs are single pointees rewritten at replay.
# _HANDLE_CREATE_APIS registers each new node in graph_node_map. 16 inline dependencies cover real graphs; overflow warns at capture.
_MAX_DEPS = 16

def _deps(param: str, count: str = "numDependencies") -> Deref:
    return Deref(param, "hipGraphNode_t", 8, count=count, max_count=_MAX_DEPS,
                 elem_handle="hipGraphNode_t")


# Recorded-address members inside each node-parameter struct.
_MEMSET_PARAMS = dict(ctype="hipMemsetParams", size=48, ptr_members=("dst",))
_MEMCPY3D_PARAMS = dict(
    ctype="hipMemcpy3DParms", size=160,
    ptr_members=("srcPtr.ptr", "dstPtr.ptr"),
    handle_members=(("srcArray", "hipArray_t"), ("dstArray", "hipArray_t")),
)

DEREF_FIELDS.update({
    "hipGraphAddEmptyNode":       [_deps("pDependencies")],
    "hipGraphAddEventRecordNode": [_deps("pDependencies")],
    "hipGraphAddEventWaitNode":   [_deps("pDependencies")],
    "hipGraphAddChildGraphNode":  [_deps("pDependencies")],
    "hipGraphAddMemcpyNode1D":    [_deps("pDependencies")],
    # Host buffers are carried as blobs by the manual shim, so only the dependency array is declarative.
    "hipGraphAddMemcpyNodeToSymbol":   [_deps("pDependencies")],
    "hipGraphAddMemcpyNodeFromSymbol": [_deps("pDependencies")],
    "hipGraphAddMemFreeNode":     [_deps("pDependencies")],
    "hipDrvGraphAddMemFreeNode":  [_deps("dependencies")],
    "hipGraphAddDependencies":    [_deps("from"), _deps("to")],
    "hipGraphAddMemsetNode": [
        _deps("pDependencies"),
        Deref("pMemsetParams", **_MEMSET_PARAMS),
    ],
    "hipDrvGraphAddMemsetNode": [
        _deps("dependencies"),
        Deref("memsetParams", **_MEMSET_PARAMS),
    ],
    "hipGraphAddMemcpyNode": [
        _deps("pDependencies"),
        Deref("pCopyParams", **_MEMCPY3D_PARAMS),
    ],
    # A kernel node needs the kernel-launch encoder: a host function address and a void** argument array.
    "hipGraphAddKernelNode": [
        _deps("pDependencies"),
        Deref("pNodeParams", "hipKernelNodeParams", 64, playback="manual"),
    ],
    # dptr must land in alloc_map and poolProps names a device, so replay is hand-written.
    "hipGraphAddMemAllocNode": [
        _deps("pDependencies"),
        Deref("pNodeParams", "hipMemAllocNodeParams", 120, direction="inout",
              playback="manual"),
    ],
    # A host source needs a blob rather than a pointer rewrite.
    "hipDrvGraphAddMemcpyNode": [
        _deps("dependencies"),
        Deref("copyParams", "HIP_MEMCPY3D", 184, playback="manual"),
    ],
    "hipGraphAddBatchMemOpNode": [
        _deps("dependencies"),
        Deref("nodeParams", "hipBatchMemOpNodeParams", 32, playback="manual"),
    ],
    "hipGraphMemsetNodeSetParams":     [Deref("pNodeParams", **_MEMSET_PARAMS)],
    "hipGraphExecMemsetNodeSetParams": [Deref("pNodeParams", **_MEMSET_PARAMS)],
    "hipDrvGraphExecMemsetNodeSetParams": [Deref("memsetParams", **_MEMSET_PARAMS)],
    "hipGraphMemcpyNodeSetParams":     [Deref("pNodeParams", **_MEMCPY3D_PARAMS)],
    "hipGraphExecMemcpyNodeSetParams": [Deref("pNodeParams", **_MEMCPY3D_PARAMS)],
    "hipGraphKernelNodeSetParams": [
        Deref("pNodeParams", "hipKernelNodeParams", 64, playback="manual"),
    ],
    "hipGraphExecKernelNodeSetParams": [
        Deref("pNodeParams", "hipKernelNodeParams", 64, playback="manual"),
    ],
    "hipDrvGraphMemcpyNodeSetParams": [
        Deref("nodeParams", "HIP_MEMCPY3D", 184, playback="manual"),
    ],
    "hipDrvGraphExecMemcpyNodeSetParams": [
        Deref("copyParams", "HIP_MEMCPY3D", 184, playback="manual"),
    ],
    "hipGraphBatchMemOpNodeSetParams": [
        Deref("nodeParams", "hipBatchMemOpNodeParams", 32, direction="inout",
              playback="manual"),
    ],
    "hipGraphExecBatchMemOpNodeSetParams": [
        Deref("nodeParams", "hipBatchMemOpNodeParams", 32, playback="manual"),
    ],
    # As a no-op, the begin never happened at replay, so hipStreamEndCapture returned 401.
    "hipStreamBeginCaptureToGraph": [
        _deps("dependencies", count="numDependencies"),
        Deref("dependencyData", "hipGraphEdgeData", 8, count="numDependencies",
              max_count=_MAX_DEPS),
    ],
})


def deref_specs(api: str) -> List[Deref]:
    return DEREF_FIELDS.get(api, [])


def deref_for_param(api: str, param: str) -> Optional[Deref]:
    for d in deref_specs(api):
        if d.param == param:
            return d
    return None


def deref_covered_params(api: str) -> Set[str]:
    """Parameters whose pointee reaches the archive.

    Read by the matrix's payload-loss detector (derive_manifest.py): a
    const-struct-pointer argument that is carried inline is no longer a loss.
    """
    return {d.param for d in deref_specs(api)}


# Structs passed by value that do not fit in a uint64_t. normalise_field_type() would lower them to a uint64_t recorded as zero.
# They are recorded as inline bytes `uint8_t <param>_bytes[N]`; ptr_members names members holding device addresses, translated through ctx.translate_ptr at replay.
# The IPC handle types are capabilities: fields keep their size but stay zero (capability=True).

@dataclass(frozen=True)
class ByValueStruct:
    size: int
    ptr_members: Tuple[str, ...] = ()
    capability: bool = False


BY_VALUE_STRUCTS: Dict[str, ByValueStruct] = {
    "hipExtent":           ByValueStruct(24),
    "hipPos":              ByValueStruct(24),
    "hipPitchedPtr":       ByValueStruct(32, ptr_members=("ptr",)),
    "hipMemLocation":      ByValueStruct(8),
    "hipChannelFormatDesc": ByValueStruct(20),
    "hipIpcMemHandle_t":   ByValueStruct(64, capability=True),
    "hipIpcEventHandle_t": ByValueStruct(64, capability=True),
}


def by_value_struct(raw_type: str) -> Optional[ByValueStruct]:
    """The by-value inline-bytes spec for a parameter type, if it has one."""
    t = re.sub(r"\b(const|volatile|restrict|struct)\b", " ", raw_type)
    if "*" in t:
        return None
    parts = t.split()
    return BY_VALUE_STRUCTS.get(parts[0]) if parts else None


# Maps the base name of an inline uint8_t array field to the C struct whose sizeof must fit.
# Emits static_assert(sizeof(CType) == N) after each struct containing such a field.
_INLINE_STRUCT_ASSERTS: Dict[str, str] = {
    "pool_props_bytes":   "hipMemPoolProps",
    "parms_bytes":        "hipMemcpy3DParms",
    "drv3d_bytes":        "HIP_MEMCPY3D",
    "drv2d_bytes":        "hip_Memcpy2D",
    "array_desc_bytes":   "HIP_ARRAY_DESCRIPTOR",
    "array3d_desc_bytes": "HIP_ARRAY3D_DESCRIPTOR",
    "stream_attr_bytes":  "hipStreamAttrValue",
    "alloc_prop_bytes":   "hipMemAllocationProp",
    "access_desc_bytes":  "hipMemAccessDesc",
}



_HANDLE_TYPES = {
    "hipStream_t", "hipEvent_t", "hipModule_t", "hipFunction_t",
    "hipCtx_t", "hipDevice_t", "hipDeviceptr_t",
    "hipArray_t", "hipArray_const_t", "hipMipmappedArray_t",
    "hipMipmappedArray_const_t", "hipSurfaceObject_t", "hipTextureObject_t",
    "hipMemPool_t", "hipGraph_t", "hipGraphNode_t", "hipGraphExec_t",
    "hipUserObject_t", "hipMemGenericAllocationHandle_t",
    "hipExternalMemory_t", "hipExternalSemaphore_t",
    "hipKernel_t", "hipLibrary_t", "hipLinkState_t",
    "hipExecutionCtx_t", "hipDevResourceDesc_t",
}

# Types that cannot be cast to uint64_t and are recorded as zero: function pointers and opaque handles meaningless outside the capturing process.
# An API that depends on one belongs in UNREPLAYABLE_PLAYBACK_APIS.
_NON_CASTABLE_TYPES = {
    "hipGraphicsResource_t",  # opaque GL interop resource
    "hipStreamCallback_t",    # function pointer
    "hipHostFn_t",            # function pointer
}

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
    # Without this an enum lowers to `uint64_t /* T */`, which the payload-loss detector reads as a dropped struct.
    # Each is a 4-byte enum.
    "hipArray_Format", "hipDevResourceType", "hipFunction_attribute",
    "hipGraphMemAttributeType", "hipJitInputType",
    "hipMemAllocationHandleType", "hipMemRangeHandleType",
    "hipPointer_attribute", "hipStreamAttrID", "hipGLDeviceList",
}

_SCALAR_MAP = [
    # Longer, more specific names first.
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
    # OpenGL spellings of unsigned int; the GL headers are not included.
    ("GLenum",             "uint32_t"),
    ("GLuint",             "uint32_t"),
]


def normalise_field_type(raw_type: str) -> str:
    """
    Convert a C parameter type string to the field type used in the struct.
    Returns a sentinel "__DIM3__" for dim3/uint3 (caller expands to x/y/z), and
    "__BYVAL__" for a by-value struct carried as inline bytes (BY_VALUE_STRUCTS).
    """
    t = raw_type.strip()
    t_nc = re.sub(r'\b(const|volatile|restrict|struct)\b', '', t).strip()
    t_nc = re.sub(r'\s+', ' ', t_nc)

    if '*' in t_nc:
        return "uint64_t"

    base = t_nc.split()[0]
    if base in ("dim3", "uint3"):
        return "__DIM3__"

    if base in BY_VALUE_STRUCTS:
        return "__BYVAL__"

    if base in _HANDLE_TYPES:
        return "uint64_t"

    if base in _ENUM_TYPES:
        return "int32_t"

    for key, mapped in _SCALAR_MAP:
        if t_nc == key:
            return mapped

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
        if ft == "__BYVAL__":
            bv = by_value_struct(self.raw_type)
            assert bv is not None
            return [f"uint8_t {safe}_bytes[{bv.size}];"]
        return [f"{ft} {safe};"]


@dataclass
class ApiEntry:
    name:     str
    ret_type: str
    params:   List[Param]
    table:    str   # "runtime" | "compiler"
    # Retired or unparsable slot: keeps its hrr_api_id_t so later IDs do not shift; no capture shim is installed.
    reserved: bool = False
    reserved_reason: str = ""



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

    before = text[:idx]
    typedef_start = before.rfind('typedef')
    if typedef_start == -1:
        return None

    after_needle = idx + len(needle)
    i = after_needle
    while i < len(text) and text[i].isspace():
        i += 1
    if i >= len(text) or text[i] != '(':
        return None
    end, _inner = _extract_balanced_parens(text, i)
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
    raw = re.sub(r'\[.*?\]', '*', raw)

    raw = re.sub(r'\s+', ' ', raw)

    m = re.match(r'^(.*?)\s*(\**)([a-zA-Z_]\w*)$', raw)
    if m:
        type_prefix = m.group(1).strip()
        stars        = m.group(2)
        potential_name = m.group(3)
        if type_prefix:
            full_type = (type_prefix + stars).strip()
            return Param(raw_type=full_type, name=potential_name)
        else:
            return Param(raw_type=raw, name='')

    return Param(raw_type=raw, name='')


def _parse_typedef_text(full_text: str, func_name: str) -> Optional[ApiEntry]:
    """
    Parse a complete typedef string into an ApiEntry.
    full_text looks like:
        typedef hipError_t (*t_hipMalloc)(void** ptr, size_t size);
    """
    m = re.match(r'typedef\s+(.+?)\s*\(\s*\*', full_text, re.DOTALL)
    if not m:
        return None
    ret_type = re.sub(r'\s+', ' ', m.group(1)).strip()

    needle = f"(*t_{func_name})"
    idx = full_text.find(needle)
    if idx == -1:
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


def _dispatch_table_slots(text: str, struct_name: str) -> List[Tuple[str, bool]]:
    """(func_name, reserved) in dispatch-table member declaration order.

    hip_api_trace.hpp mandates that new members are appended to the end of a
    dispatch table and that existing ones are never re-ordered or removed (a
    retired slot becomes a nulled void*), because anything else breaks the ABI.
    That makes member order append-only, which is what hrr_api_id_t needs: the
    IDs are written into every captured event, so an ID that shifts silently
    re-interprets existing archives. Typedef declaration order carries no such
    guarantee — it is maintained roughly alphabetically, so a new API lands in
    the middle and pushes every later ID up by one.

    A member may span two lines when the typedef name is long, so the type and
    the member name are matched across whitespace rather than within one line.
    Both `t_hipFoo hipFoo_fn;` and a retired `void* hipFoo_fn;` occupy a slot;
    the latter is returned with reserved=True.
    """
    m = re.search(r'struct\s+' + struct_name + r'\s*\{(.*?)\n\}\s*;', text, re.S)
    if not m:
        return []
    slots: List[Tuple[str, bool]] = []
    for tm in re.finditer(
            r'^\s*(?:t_(\w+)\s+(\w+)|void\s*\*\s*(\w+))\s*;', m.group(1), re.M):
        if tm.group(1) is not None:
            slots.append((tm.group(1), False))
        else:
            member = tm.group(3)
            name = member[:-3] if member.endswith('_fn') else member
            slots.append((name, True))
    return slots


def parse_hip_api_trace(path: Path) -> List[ApiEntry]:
    text = path.read_text(encoding='utf-8')
    text = _strip_comments(text)

    entries: List[ApiEntry] = []

    def reserve(func_name: str, table: str, why: str) -> None:
        print(f"WARNING: {why} for {func_name}; reserving ID so later slots "
              f"do not shift", file=sys.stderr)
        entries.append(ApiEntry(name=func_name, ret_type="void", params=[],
                                table=table, reserved=True, reserved_reason=why))

    def add_entry(func_name: str, table: str, reserved: bool = False) -> None:
        if reserved:
            reserve(func_name, table, "retired void* dispatch slot")
            return
        typedef_name = "t_" + func_name   # e.g. t___hipRegisterFatBinary
        full = find_typedef_for(text, typedef_name)
        if not full:
            reserve(func_name, table, "typedef not found")
            return
        entry = _parse_typedef_text(full, func_name)
        if not entry:
            reserve(func_name, table, "failed to parse typedef")
            return
        entry.table = table
        entries.append(entry)

    # Runtime first: compiler-first would renumber all runtime IDs whenever HipCompilerDispatchTable grows. A new HipDispatchTable member shifts only the compiler-ID tail (playback no-ops).
    runtime_slots = _dispatch_table_slots(text, "HipDispatchTable")
    if not runtime_slots:
        sys.exit("ERROR: HipDispatchTable missing or has no dispatch slots in "
                 "hip_api_trace.hpp; refusing to invent hrr_api_id_t order")

    compiler_slots = _dispatch_table_slots(text, "HipCompilerDispatchTable")
    if not compiler_slots:
        print("WARNING: HipCompilerDispatchTable missing or empty; "
              "using built-in compiler API list", file=sys.stderr)
        compiler_slots = [(n, False) for n in _COMPILER_APIS]

    for func_name, reserved in runtime_slots:
        add_entry(func_name, "runtime", reserved=reserved)
    for func_name, reserved in compiler_slots:
        add_entry(func_name, "compiler", reserved=reserved)

    # A t_hipXxx typedef with no HipDispatchTable member goes last, so table-ordered IDs keep their values. None exist today.
    in_table = {n for n, _ in runtime_slots} | {n for n, _ in compiler_slots}
    stray_pattern = re.compile(r'\(\s*\*\s*t_(hip\w+)\s*\)')
    seen = set()
    for m in stray_pattern.finditer(text):
        func_name = m.group(1)
        if func_name in in_table or func_name in seen:
            continue
        seen.add(func_name)
        print(f"WARNING: {func_name} has no HipDispatchTable member; "
              f"appending it after the table-ordered APIs", file=sys.stderr)
        add_entry(func_name, "runtime")

    return entries


def validate_capture_coverage(entries: List[ApiEntry],
                              manual_capture_source: Path,
                              silent: bool = False) -> None:
    """Validate reserved slots and hand-written capture shim definitions."""
    errors: List[str] = []
    if not manual_capture_source.is_file():
        errors.append(f"manual capture source is missing: {manual_capture_source}")
        manual_capture_text = ""
    else:
        manual_capture_text = manual_capture_source.read_text(encoding="utf-8")

    for entry in entries:
        if entry.reserved:
            if entry.reserved_reason != "retired void* dispatch slot":
                errors.append(f"{entry.table}:{entry.name}: {entry.reserved_reason}")
            continue
        if entry.name not in MANUAL_CAPTURE_APIS:
            continue
        definition = re.compile(
            r"(?m)^[^#\n]*\bcapture_" + re.escape(entry.name) + r"\s*\(")
        if not definition.search(manual_capture_text):
            errors.append(
                f"{entry.table}:{entry.name}: manual capture shim definition is "
                f"missing from {manual_capture_source}")

    if errors:
        sys.exit("ERROR: HRR capture coverage check failed:\n  " + "\n  ".join(errors))
    if not silent:
        manual_count = sum(
            not entry.reserved and entry.name in MANUAL_CAPTURE_APIS for entry in entries)
        print(f"  HRR capture coverage: {manual_count} manual shims verified")



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
 * Archive format (v6):
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
 *   - payload_length uint32_t  total record size in bytes (incl. header)
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
 * Dereferenced pointer arguments (DEREF_FIELDS in gen_hrr_api_args.py) add,
 * for a pointer argument whose pointee must survive into the archive:
 *   - uint8_t  <param>_bytes[N]   the pointee's bytes, copied at capture
 *   - uint8_t  <param>_present    1 when the argument was non-null
 *   - uint32_t <param>_n          element count, for array arguments only
 * Without them a pointer argument reaches the archive as a capture-time host
 * address and nothing else, which is the payload-loss class.
 *
 * The structs use #pragma pack(1) so layout is identical on all platforms.
 * ============================================================================
 */
#pragma once

#include <stdint.h>
#include <string.h>

/* ---- Archive format constants ---- */
#define HRR_MAGIC   ((uint32_t)0x52524845u)  /* "HRRE" */
/* v4: payload_length widened from uint16_t to uint32_t so kernel-launch events
 * larger than 65535 bytes (many args / long mangled names / large by-value
 * structs) are no longer dropped.
 * v5: hrr_api_id_t is assigned from HipDispatchTable member order, then
 * HipCompilerDispatchTable member order, instead of typedef declaration
 * order, which renumbered 496 of the 552 IDs once. Every event stores its
 * ID, so a pre-v5 archive names the wrong API when decoded against this
 * table and needs an ID translation to be read back.
 * Runtime IDs come first and compiler IDs occupy the tail, so appending a
 * HipCompilerDispatchTable member moves no existing ID, while appending a
 * HipDispatchTable member takes the first compiler ID and shifts the whole
 * compiler tail up by one. Compiler APIs do write events, so that second
 * case still needs a version bump. A retired dispatch-table slot (nulled
 * void*) still occupies an ID.
 * v6: pointer arguments whose pointee used to be dropped now carry it inline
 * (DEREF_FIELDS). Event payloads grew for ~50 APIs, so an archive written
 * before v6 cannot be read by a v6 reader: re-capture rather than replay an
 * old recording.
 * v7: the host blobs of the pitched copies are packed, and the writer sets
 * HRR_FILE_FLAG_PACKED_HOST_RECTS. A v6 reader ignored the flag and replayed
 * a packed blob with the recorded pitch, reading past its end.
 * v8: hipDeviceFlushGPUDirectRDMAWrites and hipLibraryGetModule were
 * appended to HipDispatchTable. Each took a compiler ID, so the compiler
 * API IDs moved up by two. develop still records both under v7. A v7
 * reader decodes those events as the wrong API.
 * v9: hipKernelSetAttributeForDevice was appended to HipDispatchTable,
 * taking the first compiler ID and shifting the compiler API IDs up by one.
 * A v8 reader would decode those events as the wrong API.
 * v10: a kernel launch event carries pinned host snapshot records after its
 * arguments (num_snapshots was always 0). A v9 reader would not restore them.
 * v11: hipKernelSetCacheConfig was appended to HipDispatchTable, taking the
 * first compiler ID and shifting the compiler API IDs up by one.
 * A v10 reader would decode those events as the wrong API.
 * v12: hipExtMemcpyBatchAsync was appended to HipDispatchTable,
 * shifting the compiler API IDs up by one. A v11 reader would decode
 * those events as the wrong API. */
#define HRR_VERSION ((uint16_t)12u)

/* Written once at byte 0 of events.bin. */
#pragma pack(push, 1)
typedef struct {
    uint32_t magic;    /* HRR_MAGIC                */
    uint16_t version;  /* HRR_VERSION              */
    uint16_t reserved; /* HRR_FILE_FLAG_* bits      */
} hrr_file_header;
#pragma pack(pop)

/* Bits of hrr_file_header.reserved in events.bin. A reader before v7 ignores
 * them, which is why the packed layout came with the v7 bump.
 * HRR_FILE_FLAG_PACKED_HOST_RECTS: the host blobs of the pitched copies
 * (hipMemcpy2D, the hipMemcpy3D family and the driver 2D/3D copies) hold only
 * the copied rows, packed end to end: width*height*depth bytes, whatever the
 * pitch. Without it, a blob spans the host rect from its base pointer through
 * the last copied byte, or holds the flat width*height*depth bytes from the
 * base pointer (3D and driver D2H, hipMemcpy3D H2D).
 * HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS: hipExtModuleLaunchKernel records
 * its start and stop events as hipEventRecord events around the launch.
 * Without it, replay cannot tell a missing marker from a replay that failed to
 * apply one, so only an archive without the flag may tolerate a
 * hipEventElapsedTime that returns hipErrorInvalidHandle. No version bump: an
 * older reader replays the extra hipEventRecord events like any others. */
#define HRR_FILE_FLAG_PACKED_HOST_RECTS ((uint16_t)0x0001u)
#define HRR_FILE_FLAG_EXT_LAUNCH_EVENT_MARKERS ((uint16_t)0x0002u)

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
    uint32_t payload_length; /* total record size in bytes (incl. header) */
    uint8_t  reserved[2];    /* padding to 32 bytes; zero on write        */
} hrr_event_header;

#ifdef __cplusplus
static_assert(sizeof(hrr_event_header) == 32, "hrr_event_header must be 32 bytes");
#endif

/* ---- Clean-shutdown trailer ----
 * Written once at the end of events.bin by the capture writer ONLY on a clean
 * shutdown (writer::flush). Its ABSENCE tells the reader the capture was
 * interrupted (e.g. the recorded process crashed) and the trailing record may
 * be torn — the reader then recovers all complete records instead of failing.
 * event_type uses a sentinel (HRR_EOF_MARKER) far outside the hrr_api_id_t
 * range, so playback dispatch and name lookup treat it as unknown/no-op if it
 * is ever fed to them. */
#define HRR_EOF_MARKER ((uint16_t)0xFFFFu)     /* hrr_event_header.event_type sentinel */
#define HRR_EOF_MAGIC  ((uint32_t)0x464F4548u) /* "HEOF" trailer payload magic         */

typedef struct {
    hrr_event_header hdr;   /* event_type = HRR_EOF_MARKER, payload_length = sizeof(hrr_eof_record) */
    uint64_t total_events;  /* count of real events written before this trailer */
    uint32_t eof_magic;     /* HRR_EOF_MAGIC                                     */
} hrr_eof_record;

#ifdef __cplusplus
static_assert(sizeof(hrr_eof_record) == 44, "hrr_eof_record must be 44 bytes");
#endif

/* Build a clean-shutdown trailer record. Single source of truth for the trailer
 * layout, shared by the capture writer (writer::flush) and the offline repair
 * tool (hrr-playback --repair) so the two cannot drift. The caller may overwrite
 * hdr.timestamp_ns / hdr.thread_id afterwards; the offline tool leaves them 0. */
static inline hrr_eof_record hrr_make_eof_record(uint64_t sequence_id,
                                                 uint64_t total_events) {
    hrr_eof_record rec;
    memset(&rec, 0, sizeof(rec));
    rec.hdr.event_type     = HRR_EOF_MARKER;
    rec.hdr.sequence_id    = sequence_id;
    rec.hdr.payload_length = (uint16_t)sizeof(hrr_eof_record);
    rec.total_events       = total_events;
    rec.eof_magic          = HRR_EOF_MAGIC;
    return rec;
}

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

    if entry.reserved:
        lines.append(f"/* retired/unparsable dispatch slot {entry.name}; ID placeholder */")
    else:
        param_sig = ', '.join(
            (p.raw_type + ' ' + p.name).strip()
            for p in entry.params
        )
        lines.append(f"/* {entry.ret_type} {entry.name}({param_sig}) */")
    lines.append("typedef struct {")
    lines.append("    hrr_event_header hdr;")

    if not _is_void_return(entry.ret_type):
        if '*' in entry.ret_type:
            lines.append("    uint64_t ret;")
        else:
            ft = normalise_field_type(entry.ret_type)
            if ft == "__BYVAL__":
                # By-value struct return (hipCreateChannelDesc): never recorded, but the field needs a round-trippable size.
                bv = by_value_struct(entry.ret_type)
                lines.append(f"    uint8_t ret_bytes[{bv.size}];")
                ft = None
            elif ft == "__DIM3__":
                ft = "uint64_t"
            if ft is not None:
                lines.append(f"    {ft} ret;")

    if entry.name == "__hipRegisterFatBinary":
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
            elif ft == "__BYVAL__":
                bv = by_value_struct(param.raw_type)
                base = _get_base_type(param.raw_type)
                kept = "not recorded, zeros" if bv.capability else "inline copy"
                lines.append(f"    uint8_t {safe}_bytes[{bv.size}];"
                             f"  /* {base} passed by value, {kept} */")
            else:
                lines.append(f"    {ft} {safe};")

    for d in deref_specs(entry.name):
        what = (f"{d.ctype}[{d.max_count}] inline copy" if d.is_array
                else f"{d.ctype} not recorded, zeros" if d.capability
                else f"{d.ctype} inline copy")
        lines.append(f"    uint8_t {d.bytes_field}[{d.total_bytes}];  /* {what} */")
        lines.append(f"    uint8_t {d.present_field};  /* 1 when {d.param} was non-null */")
        if d.is_array:
            lines.append(f"    uint32_t {d.count_field};  /* elements copied from {d.param} */")

    for (ftype, fname, fcomment) in EXTRA_FIELDS.get(entry.name, []):
        lines.append(f"    {ftype} {fname};  /* {fcomment} */")

    lines.append(f"}} {sname};")

    # Guarded by HIP_INCLUDE_HIP_HIP_RUNTIME_H: the referenced types exist only with the full HIP runtime header, and hrr_api_args.h is also included by TUs with minimal headers.
    asserts = []
    for (ftype, fname, _fcomment) in EXTRA_FIELDS.get(entry.name, []):
        if not ftype.startswith("uint8_t") or '[' not in fname:
            continue
        base = fname[:fname.index('[')]
        ctype = _INLINE_STRUCT_ASSERTS.get(base)
        if not ctype:
            continue
        n = fname[fname.index('[')+1 : fname.index(']')]
        asserts.append(
            f"static_assert(sizeof({ctype}) <= {n},"
            f' "hrr_args_{entry.name}::{fname} too small for {ctype}");'
        )
    # A field reserving fewer bytes than the value needs would truncate silently at capture; catch it in the compiler.
    for d in deref_specs(entry.name):
        asserts.append(
            f"static_assert(sizeof({d.ctype}) <= {d.size},"
            f' "hrr_args_{entry.name}::{d.bytes_field} too small for {d.ctype}");'
        )
    for param in entry.params:
        if normalise_field_type(param.raw_type) != "__BYVAL__":
            continue
        bv = by_value_struct(param.raw_type)
        base = _get_base_type(param.raw_type)
        asserts.append(
            f"static_assert(sizeof({base}) <= {bv.size},"
            f' "hrr_args_{entry.name}::{param.name or "unnamed"}_bytes'
            f' too small for {base}");'
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

    parts.append("/* ---- API id enumeration ---- */")
    parts.append("typedef enum hrr_api_id {")
    for idx, e in enumerate(entries):
        enum_name = "HRR_API_" + e.name.lstrip('_').upper()
        parts.append(f"    {enum_name} = {idx},")
    parts.append(f"    HRR_API_COUNT = {len(entries)}")
    parts.append("} hrr_api_id_t;\n")

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

// hrr_api_args.h now lives in projects/hrr/include/hrr (shared wire format)
#include "hrr/hrr_api_args.h"

#include "hip/amd_detail/hip_api_trace.hpp"
// LogPrintfWarning — capture diagnostics go through amd's log-level machinery,
// never raw stderr. debug.hpp's ClPrint macro reads the AMD_LOG_LEVEL and
// AMD_LOG_MASK flag variables but does not declare them, so flags.hpp must come
// with it, and flags.hpp spells its defaults with top.hpp's unit suffixes (Mi)
// and typedefs (uint). All three, in this order.
#include "top.hpp"
#include "utils/flags.hpp"
#include "utils/debug.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
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
    # A single-level pointer to const is something the API reads, whatever it
    # points at. Without this, a `const hipGraphNode_t*` dependency array reads
    # as an output handle and both ends treat an array as one node.
    if t.count('*') == 1 and re.match(r'^\s*const\b', t):
        return False
    if t.count('*') >= 2:
        return True
    _OUTPUT_HANDLE_TYPES = (
        'hipStream_t', 'hipEvent_t', 'hipModule_t', 'hipFunction_t',
        'hipMemPool_t', 'hipGraph_t', 'hipGraphExec_t', 'hipGraphNode_t',
        'hipArray_t', 'hipMipmappedArray_t',
        'hipSurfaceObject_t', 'hipTextureObject_t',
        'hipCtx_t', 'hipUserObject_t',
        'hipMemGenericAllocationHandle_t',
        'hipLinkState_t',
    )
    for handle in _OUTPUT_HANDLE_TYPES:
        if handle + '*' in t or handle + ' *' in t:
            return True
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
    elif ft == "__BYVAL__":
        bv = by_value_struct(t)
        if bv is not None and bv.capability:
            lines.append(f"    // {name} is a capability and is not recorded.")
        else:
            lines.append(f"    std::memcpy(a.{name}_bytes, &{name}, sizeof({name}));")
    elif base in _NON_CASTABLE_TYPES:
        lines.append(f"    a.{name} = 0;  // non-castable type skipped")
    elif base == 'hipDevice_t':
        lines.append(f"    a.{name} = static_cast<uint64_t>(static_cast<int>({name}));")
    elif base == 'hipDeviceptr_t' or t == 'hipDeviceptr_t':
        lines.append(f"    a.{name} = static_cast<uint64_t>(reinterpret_cast<uintptr_t>({name}));")
    elif ft == "uint64_t" and '*' in t:
        lines.append(f"    a.{name} = reinterpret_cast<uint64_t>({name});")
    elif ft == "uint64_t" and base in _HANDLE_TYPES:
        lines.append(f"    a.{name} = reinterpret_cast<uint64_t>({name});")
    else:
        lines.append(f"    a.{name} = static_cast<decltype(a.{name})>({name});")


def _fill_derefs(lines: List[str], entry: ApiEntry) -> None:
    """Emit the memcpy that carries each dereferenced pointee into the event.

    Runs after the real call, alongside the other fills: the arguments a HIP
    API reads are still valid there, and an output argument has been written
    by then.

    A pointee too large for its fixed-size field is clipped. The warning alone
    is not enough: AMD_LOG_LEVEL defaults to 0, so nothing durable would record
    that the payload is partial and replay would treat it as faithful. Each
    truncation therefore also calls note_unreplayable(), which lists the API,
    with the clipped argument as the reason, under manifest.unreplayable_apis
    where replay can see it. The warning is printed once per process, but the
    note is made on every truncation: the writer lists notes per archive, and
    a forked child or a fresh archive starts with none.

    note_unreplayable(), not mark_incomplete(): the event is written and the
    archive stays well-formed — only the clipped argument makes the call
    impossible to reproduce exactly. mark_incomplete() means an event was
    dropped or torn and suppresses the clean-shutdown trailer, which is a
    different and stronger claim. See hip_capture_writer.h.
    """
    for d in deref_specs(entry.name):
        if d.string:
            lines.append(f"    if ({d.param}) {{")
            lines.append(f"      size_t _n = std::strlen({d.param});")
            lines.append(f"      if (_n > {d.max_count - 1}u) {{")
            lines.append(f"        static bool warned_{d.param} = false;")
            lines.append(f"        if (!warned_{d.param}) {{")
            lines.append(f"          warned_{d.param} = true;")
            lines.append(f"          LogPrintfWarning(")
            lines.append(f"              \"[HRR] {entry.name}: {d.param} is %zu characters; \"")
            lines.append(f"              \"recording the first {d.max_count - 1} only\", _n);")
            lines.append(f"        }}")
            lines.append(f"        hrr_cap::writer::note_unreplayable(\"{entry.name}\",")
            lines.append(f"            \"{d.param} was truncated to {d.max_count - 1} characters at capture \"")
            lines.append(f"            \"time; replay would pass a shortened string\");")
            lines.append(f"        _n = {d.max_count - 1}u;")
            lines.append(f"      }}")
            lines.append(f"      std::memcpy(a.{d.bytes_field}, {d.param}, _n);")
            lines.append(f"      a.{d.bytes_field}[_n] = 0;")
            lines.append(f"      a.{d.present_field} = 1;")
            lines.append(f"    }}")
        elif d.is_array:
            lines.append(f"    if ({d.param} && {d.count} > 0) {{")
            lines.append(f"      uint32_t _n = static_cast<uint32_t>({d.count});")
            lines.append(f"      if (_n > {d.max_count}u) {{")
            lines.append(f"        static bool warned_{d.param} = false;")
            lines.append(f"        if (!warned_{d.param}) {{")
            lines.append(f"          warned_{d.param} = true;")
            lines.append(f"          LogPrintfWarning(")
            lines.append(f"              \"[HRR] {entry.name}: recording only the first {d.max_count} \"")
            lines.append(f"              \"of %u {d.param} entries; replay of this call will be \"")
            lines.append(f"              \"incomplete\", _n);")
            lines.append(f"        }}")
            lines.append(f"        hrr_cap::writer::note_unreplayable(\"{entry.name}\",")
            lines.append(f"            \"only the first {d.max_count} {d.param} entries were recorded at \"")
            lines.append(f"            \"capture time; replay would pass a partial array\");")
            lines.append(f"        _n = {d.max_count}u;")
            lines.append(f"      }}")
            lines.append(f"      std::memcpy(a.{d.bytes_field}, {d.param},"
                         f" static_cast<size_t>(_n) * sizeof({d.ctype}));")
            lines.append(f"      a.{d.count_field}   = _n;")
            lines.append(f"      a.{d.present_field} = 1;")
            lines.append(f"    }}")
        elif d.capability:
            lines.append(f"    // *{d.param} is a capability and is not recorded.")
            lines.append(f"    if ({d.param}) a.{d.present_field} = 1;")
        else:
            lines.append(f"    if ({d.param}) {{")
            lines.append(f"      std::memcpy(a.{d.bytes_field}, {d.param}, sizeof({d.ctype}));")
            for m in d.blank_members:
                lines.append(f"      std::memset(a.{d.bytes_field} + offsetof({d.ctype}, {m}), 0,"
                             f" sizeof({d.param}->{m}));")
            lines.append(f"      a.{d.present_field} = 1;")
            lines.append(f"    }}")


# Output parameters whose value is a capturing-process address that replay does not use, such as a symbol lookup's resolved entry point.
# Replay looks the name up again, so the field stays 0 and no load address leaks.
UNRECORDED_OUTPUTS: Dict[str, Set[str]] = {
    "hipGetProcAddress":          {"pfn"},
    "hipGetProcAddress_spt":      {"pfn"},
    "hipGetDriverEntryPoint":     {"funcPtr"},
    "hipGetDriverEntryPoint_spt": {"funcPtr"},
}


def _fill_output_param_post(lines: List[str], p: Param, name: str,
                            api: str = "") -> None:
    """Emit lines that fill a.name AFTER the real call for output pointer params."""
    if name in UNRECORDED_OUTPUTS.get(api, ()):
        lines.append(f"    // {name} is not recorded, see UNRECORDED_OUTPUTS in the generator.")
        return
    t = p.raw_type.strip()
    if 'hipDevice_t' in t and 'hipDeviceptr_t' not in t:
        lines.append(f"    if ({name}) a.{name} = static_cast<uint64_t>(static_cast<int>(*{name}));")
    else:
        # Guard with a null check: some output params are optional (e.g. pErrorNode in hipGraphInstantiate).
        lines.append(f"    if ({name}) a.{name} = reinterpret_cast<uint64_t>(*{name});")


# Per-API shim/handler bodies the default emitter cannot express. Kept here, not in the MANUAL_* sets, so the function stays in the generated files and the dispatch tables wire it automatically.
CUSTOM_CAPTURE_SHIMS: Dict[str, str] = {
    # hipThreadExchangeStreamCaptureMode is an in/out swap: input *mode is the desired capture mode, output *mode is the previous one.
    # The generic emitter records the pointer, so replay cannot restore the mode and a hipMalloc inside a graph capture fails with 900.
    # Record the input enum value instead.
    "hipThreadExchangeStreamCaptureMode": (
        "// Generated shim (custom: record input capture-mode VALUE, not the pointer)\n"
        "static hipError_t capture_hipThreadExchangeStreamCaptureMode(hipStreamCaptureMode* mode) {\n"
        "  hipStreamCaptureMode desired = mode ? *mode : hipStreamCaptureModeGlobal;\n"
        "  hipError_t r = g_real_table.hipThreadExchangeStreamCaptureMode_fn(mode);\n"
        "  if (r == hipSuccess) {\n"
        "    hrr_args_hipThreadExchangeStreamCaptureMode a{};\n"
        "    a.ret         = static_cast<int32_t>(r);\n"
        "    a.mode = static_cast<uint64_t>(desired);\n"
        "    hrr_cap::writer::write_event_raw(HRR_API_HIPTHREADEXCHANGESTREAMCAPTUREMODE, &a.hdr, sizeof(a));\n"
        "  }\n"
        "  return r;\n"
        "}\n"
    ),
    # hipStreamEndCapture on an invalidated or unjoined capture fails, yields no
    # graph, and still ends the capture. Record those failures too, or replay
    # never ends its own capture and treats every later call as captured.
    "hipStreamEndCapture": (
        "// Generated shim (custom: also record the failures that end the capture)\n"
        "static hipError_t capture_hipStreamEndCapture(hipStream_t stream, hipGraph_t* pGraph) {\n"
        "  hipError_t r = g_real_table.hipStreamEndCapture_fn(stream, pGraph);\n"
        "  if (r == hipSuccess || r == hipErrorStreamCaptureInvalidated ||\n"
        "      r == hipErrorStreamCaptureUnjoined) {\n"
        "    hrr_args_hipStreamEndCapture a{};\n"
        "    a.ret         = static_cast<int32_t>(r);\n"
        "    a.stream = reinterpret_cast<uint64_t>(stream);\n"
        "    if (pGraph) a.pGraph = reinterpret_cast<uint64_t>(*pGraph);\n"
        "    hrr_cap::writer::write_event_raw(HRR_API_HIPSTREAMENDCAPTURE, &a.hdr, sizeof(a));\n"
        "  }\n"
        "  return r;\n"
        "}\n"
    ),
}

CUSTOM_PLAYBACK_BODIES: Dict[str, str] = {
    # Restore the recorded desired thread capture mode (stored as enum VALUE) so
    # allocations bracketed by these calls during a graph capture are permitted.
    "hipThreadExchangeStreamCaptureMode": (
        "static hipError_t playback_hipThreadExchangeStreamCaptureMode(PlaybackContext& ctx, const uint8_t* payload) {\n"
        "  (void)ctx;\n"
        "  const auto* a = reinterpret_cast<const hrr_args_hipThreadExchangeStreamCaptureMode*>(payload);\n"
        "  hipStreamCaptureMode mode = static_cast<hipStreamCaptureMode>(a->mode);\n"
        "  hipError_t _r = (hipError_t)hipThreadExchangeStreamCaptureMode(&mode);\n"
        "  return _r;\n"
        "}\n"
    ),
}


def generate_shim(entry: ApiEntry) -> str:
    """Generate a single capture shim function.
    MANUAL_CAPTURE_APIS: returns empty string — hand-written in hip_capture.cpp.
    """
    if entry.reserved:
        return ""
    if entry.name in CUSTOM_CAPTURE_SHIMS:
        return CUSTOM_CAPTURE_SHIMS[entry.name]
    is_manual   = entry.name in MANUAL_CAPTURE_APIS
    is_passonly = entry.name in PASSTHROUGH_ONLY
    is_compiler = entry.table == "compiler"
    void_ret    = _is_void_return(entry.ret_type)
    is_const_char_ret = entry.ret_type.strip() in ('const char*', 'const char *')
    # A non-hipError_t struct/scalar return (e.g. hipCreateChannelDesc) cannot be compared with hipSuccess, so it is forwarded without capture.
    ret_base = _get_base_type(entry.ret_type)
    is_non_hiperrort_ret = (not void_ret and not is_const_char_ret
                            and ret_base not in ('hipError_t',)
                            and '*' not in entry.ret_type)

    if is_manual:
        return ""

    param_decl = _cpp_param_decl(entry)
    fwd_args   = _cpp_passthrough_args(entry)
    sname      = f"hrr_args_{entry.name}"

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
        lines.append(f"  return {table_name}.{fn_field}({fwd_args});")
        lines.append(f"}}")
        return '\n'.join(lines) + '\n'

    if void_ret:
        lines.append(f"  {table_name}.{fn_field}({fwd_args});")
        if not is_passonly:
            lines.append(f"  {{")
            lines.append(f"    {sname} a{{}};")
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
            for p, name, ft in output_params:
                _fill_output_param_post(lines, p, name, entry.name)
            _fill_derefs(lines, entry)
            enum_name = "HRR_API_" + entry.name.lstrip('_').upper()
            _note_unreplayable(lines, entry)
            lines.append(f"    hrr_cap::writer::write_event_raw({enum_name}, &a.hdr, sizeof(a));")
            lines.append(f"  }}")
    else:
        lines.append(f"  {entry.ret_type} r = {table_name}.{fn_field}({fwd_args});")
        if not is_passonly:
            if is_const_char_ret:
                lines.append(f"  {{")
            else:
                lines.append(f"  if (r == hipSuccess) {{")
            lines.append(f"    {sname} a{{}};")
            if is_const_char_ret:
                lines.append(f"    a.ret = reinterpret_cast<uint64_t>(r);")
            else:
                lines.append(f"    a.ret         = static_cast<int32_t>(r);")
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
            for p, name, ft in output_params:
                _fill_output_param_post(lines, p, name, entry.name)
            _fill_derefs(lines, entry)
            enum_name = "HRR_API_" + entry.name.lstrip('_').upper()
            _note_unreplayable(lines, entry)
            lines.append(f"    hrr_cap::writer::write_event_raw({enum_name}, &a.hdr, sizeof(a));")
            lines.append(f"  }}")
        lines.append(f"  return r;")

    lines.append(f"}}")
    lines.append("")
    return "\n".join(lines)


def _note_unreplayable(lines: List[str], entry: ApiEntry) -> None:
    """An API replay cannot reproduce is worth knowing about while the recording
    is being made, not only when someone later tries to replay it. Emitted next
    to write_event_raw() so manifest.json lists the API only when an event was
    actually recorded — a failed call writes no event."""
    if entry.name in UNREPLAYABLE_PLAYBACK_APIS:
        lines.append(f'    hrr_cap::writer::note_unreplayable("{entry.name}",')
        lines.append(f'        "{UNREPLAYABLE_PLAYBACK_APIS[entry.name]}");')


def generate_build_table(entries: List[ApiEntry]) -> str:
    """Generate hip_capture_build_table() and hip_capture_build_compiler_table()."""
    runtime_entries  = [e for e in entries if e.table == "runtime"]
    compiler_entries = [e for e in entries if e.table == "compiler"]

    lines = []

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
        if e.reserved:
            # No shim is generated for a reserved slot, so the real table's entry stays.
            # That is nullptr for a retired void* slot, and a live function pointer for a typedef that could not be parsed, which keeps pass-through.
            continue
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
        if e.reserved:
            continue
        lines.append(f"  cap.{e.name}_fn = capture_{e.name};")
    lines.append("  hip_capture_install_compiler_table(cap);")
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
        if e.reserved:
            continue
        parts.append(generate_shim(e))

    parts.append("// ============================================================")
    parts.append("// Table builders")
    parts.append("// ============================================================")
    parts.append("")
    parts.append(generate_build_table(entries))

    return "\n".join(parts)



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
#include "hrr/hrr_api_args.h"
#include <hip/hip_runtime.h>
#include <cstring>

// Manual playback implementations (extern'd below) are in hip_playback.cpp
// Compiler APIs are no-ops during playback

"""

_PLAYBACK_HANDLE_INFO: Dict[str, Tuple[str, str, str]] = {
    'hipStream_t':          ('ctx.translate_stream',    'ctx.record_stream',    'ctx.remove_stream'),
    'hipEvent_t':           ('ctx.translate_event',     'ctx.record_event',     'ctx.remove_event'),
    'hipModule_t':          ('ctx.translate_module',    'ctx.record_module',    'ctx.remove_module'),
    'hipFunction_t':        ('ctx.translate_func',      'ctx.record_func',      'ctx.remove_func'),
    'hipMemPool_t':         ('ctx.translate_mempool',   'ctx.record_mempool',   'ctx.remove_mempool'),
    'hipArray_t':           ('ctx.translate_array',     'ctx.record_array',     'ctx.remove_array'),
    'hipMipmappedArray_t':  ('ctx.translate_mipmapped', 'ctx.record_mipmapped', 'ctx.remove_mipmapped'),
    'hipGraph_t':           ('ctx.translate_graph',     'ctx.record_graph',     'ctx.remove_graph'),
    'hipGraphNode_t':       ('ctx.translate_graph_node','ctx.record_graph_node','ctx.remove_graph_node'),
    'hipGraphExec_t':       ('ctx.translate_graph_exec','ctx.record_graph_exec','ctx.remove_graph_exec'),
    'hipSurfaceObject_t':   ('ctx.translate_surface',   'ctx.record_surface',   'ctx.remove_surface'),
    'hipTextureObject_t':   ('ctx.translate_texture',   'ctx.record_texture',   'ctx.remove_texture'),
    'hipCtx_t':             ('ctx.translate_ctx',       'ctx.record_ctx',       'ctx.remove_ctx'),
    'hipLinkState_t':       ('ctx.translate_link_state','ctx.record_link_state','ctx.remove_link_state'),
}
_PLAYBACK_HANDLE_TRANSLATE = {k: v[0] for k, v in _PLAYBACK_HANDLE_INFO.items()}
# The VMM handle has its own map (vmm_handle_map, filled by the hand-written hipMemCreate handler) but no record/remove pair.
# Generated handlers still translate it, since the recorded value names an allocation in the capturing process.
_PLAYBACK_HANDLE_TRANSLATE['hipMemGenericAllocationHandle_t'] = \
    'ctx.translate_vmm_handle'

# Opaque values playback cannot turn back into a live object. A generated
# call would pass the recorded address from the capturing process, or a zero
# for a type that cannot be recorded. hipDevice_t is an int and is not here.
_UNTRANSLATED_PLAYBACK_TYPES = {
    "hipKernel_t",
    "hipLibrary_t",
    "hipExternalMemory_t",
    "hipExternalSemaphore_t",
    "hipExecutionCtx_t",
    "hipDevResourceDesc_t",
    "hipUserObject_t",
    "hipGraphicsResource_t",
    "hipStreamCallback_t",
    "hipHostFn_t",
}

# `void*` parameters that are really out buffers: the callee writes a handle through them, and passing the recorded value would hand the runtime a capturing-process address (SIGSEGV inside hipMemExportToShareableHandle).
# The value is the number of bytes the callee writes; the handler passes a zeroed local of that size.
# What the callee writes stays local: an fd or Win32 HANDLE is meaningful only in the receiving process.
_OUT_BUFFER_PARAMS: Dict[str, Dict[str, int]] = {
    # An fd is an int and a Win32 HANDLE is a pointer; eight bytes covers both.
    'hipMemExportToShareableHandle':     {'shareableHandle': 8},
    'hipMemPoolExportToShareableHandle': {'shared_handle': 8},
    'hipMemGetHandleForAddressRange':    {'handle': 8},
    # A `char*` the callee fills rather than a string it reads: hipDeviceGetLuid writes eight LUID bytes into luid[0..7].
    # Without a size here the pointee rule would give it one char. Only Windows writes; Linux returns hipErrorNotSupported first.
    'hipDeviceGetLuid':                  {'luid': 8},
}

_ALLOC_CREATE_APIS: Dict[str, Tuple[str, str]] = {
    'hipMalloc':             ('ptr', 'size'),
    'hipMallocAsync':        ('dev_ptr', 'size'),
    'hipMallocFromPoolAsync':('dev_ptr', 'size'),
    'hipMallocManaged':      ('dev_ptr', 'size'),
    'hipExtMallocWithFlags': ('ptr', 'sizeBytes'),  # manual playback handler; field is sizeBytes
    'hipMallocPitch':        ('ptr', 'width'),  # approximate; width used as proxy
    'hipHostMalloc':         ('ptr', 'size'),
    'hipHostAlloc':          ('ptr', 'size'),
}

# Subset of _ALLOC_CREATE_APIS whose output is *host* (pinned) memory and must be
# released with hipHostFree, not hipFree. The teardown loop dispatches on the
# AllocKind tag recorded here. APIs not listed default to AllocKind::Device.
_HOST_ALLOC_CREATE_APIS = {'hipHostMalloc', 'hipHostAlloc'}

# APIs that free device allocations: API name -> rec_ptr_param name in struct.
# hipIpcCloseMemHandle releases the pointer hipIpcOpenMemHandle imported, which
# replay skips when no export in the archive matches it.
_ALLOC_FREE_APIS: Dict[str, str] = {
    'hipFree':              'ptr',
    'hipFreeAsync':         'dev_ptr',
    'hipIpcCloseMemHandle': 'devPtr',
}

# Param names must match the typedef parameter names in hip_api_trace.hpp.
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
    'hipGraphDestroyNode':     ('node',          'hipGraphNode_t'),
    'hipCtxDestroy':           ('ctx',           'hipCtx_t'),
    'hipLinkDestroy':          ('state',         'hipLinkState_t'),
}

# APIs that create handles: API name -> list of (output param name, handle type)
# The param name MUST match the actual parameter name in the hip_api_trace.hpp typedef.
_HANDLE_CREATE_APIS: Dict[str, List[Tuple[str, str]]] = {
    'hipStreamCreate':              [('stream',      'hipStream_t')],
    'hipStreamCreateWithFlags':     [('stream',      'hipStream_t')],
    'hipStreamCreateWithPriority':  [('stream',      'hipStream_t')],
    'hipEventCreate':               [('event',       'hipEvent_t')],
    'hipEventCreateWithFlags':      [('event',       'hipEvent_t')],
    # Memory pools. The two query spellings hand out a pool they did not create, usually the device default. Registering it lets a later hipMemSetMemPool or allocation name that pool.
    'hipMemPoolCreate':             [('mem_pool',     'hipMemPool_t')],
    'hipDeviceGetMemPool':          [('mem_pool',     'hipMemPool_t')],
    'hipDeviceGetDefaultMemPool':   [('mem_pool',     'hipMemPool_t')],
    'hipMemGetMemPool':             [('pool',         'hipMemPool_t')],
    # Without this, hipLinkComplete and hipLinkDestroy were passed the capturing process's pointer and failed.
    'hipLinkCreate':                [('stateOut',     'hipLinkState_t')],
    'hipArrayCreate':               [('pHandle',      'hipArray_t')],
    'hipArray3DCreate':             [('array',        'hipArray_t')],
    'hipMallocArray':               [('array',        'hipArray_t')],
    'hipMalloc3DArray':             [('array',        'hipArray_t')],
    'hipMallocMipmappedArray':      [('mipmappedArray', 'hipMipmappedArray_t')],
    'hipGraphCreate':               [('pGraph',       'hipGraph_t')],
    'hipGraphClone':                [('pGraphClone',  'hipGraph_t')],
    'hipGraphInstantiate':          [('pGraphExec',   'hipGraphExec_t')],
    'hipGraphInstantiateWithFlags': [('pGraphExec',   'hipGraphExec_t')],
    # Registering each produced node lets a later dependency list, *NodeSetParams or hipGraphDestroyNode name it.
    'hipGraphAddEmptyNode':         [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddEventRecordNode':   [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddEventWaitNode':     [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddChildGraphNode':    [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddMemsetNode':        [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddMemcpyNode':        [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddMemcpyNode1D':      [('pGraphNode',   'hipGraphNode_t')],
    'hipGraphAddMemFreeNode':       [('pGraphNode',   'hipGraphNode_t')],
    'hipDrvGraphAddMemsetNode':     [('phGraphNode',  'hipGraphNode_t')],
    'hipDrvGraphAddMemFreeNode':    [('phGraphNode',  'hipGraphNode_t')],
    'hipGraphNodeFindInClone':      [('pNode',        'hipGraphNode_t')],
    'hipCreateSurfaceObject':       [('pSurfObject',  'hipSurfaceObject_t')],
    'hipCreateTextureObject':       [('pTexObject',   'hipTextureObject_t')],
    # Every call that hands out a context registers it, including the two that only report one: a recording pushes back what it popped, and the pop is where replay learns the live context.
    'hipCtxCreate':                 [('ctx',          'hipCtx_t')],
    'hipCtxPopCurrent':             [('ctx',          'hipCtx_t')],
    'hipCtxGetCurrent':             [('ctx',          'hipCtx_t')],
    'hipDevicePrimaryCtxRetain':    [('pctx',         'hipCtx_t')],
}


def _deref_translate_lines(var: str, ptr_members: Tuple[str, ...],
                           indent: str = "  ",
                           handle_members: Tuple[Tuple[str, str], ...] = (),
                           ) -> List[str]:
    """Rewrite the recorded device addresses and handles in a restored struct."""
    out = []
    for member in ptr_members:
        out.append(f"{indent}{var}.{member} = ctx.translate_ptr("
                   f"reinterpret_cast<uint64_t>({var}.{member}));")
    for (member, htype) in handle_members:
        fn = _PLAYBACK_HANDLE_TRANSLATE[htype]
        out.append(f"{indent}{var}.{member} = ({htype}){fn}("
                   f"reinterpret_cast<uint64_t>({var}.{member}));")
    return out


def _graph_param(entry: ApiEntry) -> str:
    """The name of the hipGraph_t this API adds to, or "" if it has none."""
    for p in entry.params:
        t = p.raw_type.strip()
        if '*' not in t and _get_base_type(t) == 'hipGraph_t' and p.name:
            return p.name
    return ""


# hipCtx_t arguments of the driver graph node calls. Nothing maps the recorded address, and hipDrvGraphAddMemcpyNode rejects a null context (hip_graph.cpp:1445).
# hrr_live_ctx substitutes the replaying thread's context and preserves a recorded null, so a call that failed at capture fails the same way here.
_LIVE_CTX_PLAYBACK_APIS: Set[str] = {
    "hipDrvGraphAddMemsetNode",
    "hipDrvGraphExecMemsetNodeSetParams",
}

_POST_CALL_EXTRA: Dict[str, List[str]] = {
    # A clone of a graph HRR could not rebuild is equally unbuildable.
    "hipGraphClone": [
        "if (_r == hipSuccess && ctx.graph_is_incomplete(a->originalGraph))",
        "  ctx.mark_graph_incomplete(a->pGraphClone, \"hipGraphClone\");",
    ],
    "hipGraphAddChildGraphNode": [
        "if (_r == hipSuccess && ctx.graph_is_incomplete(a->childGraph))",
        "  ctx.mark_graph_incomplete(a->graph, \"hipGraphAddChildGraphNode\");",
    ],
}


# Playback calls the PUBLIC HIP API by name; capture goes through the dispatch table. The two declarations may diverge: hip_runtime_api.h can re-declare an API with a new signature under a HIP_TARGET_API_VERSION guard.
# validate_playback_signatures() refuses to generate when they disagree. An entry here is the reviewed resolution for one such API, emitted under the public header's guard.
#
#   guard : the #if expression under which the public API takes the NEW type
#   param : dispatch-table parameter name being bridged
#   pre   : lines emitted inside the guard ({name} -> param name)
#   expr  : expression substituted for that argument inside the guard
# APIs declared before any SDK that CI builds playback against exports them. The call goes through a symbol looked up at replay time, so hrr-playback still links against an older amdhip64 and skips the call with a warning.
# Declarations and enum types must come from the in-tree headers (HRR_HIP_INCLUDE_DIRS in projects/hrr/CMakeLists.txt). Add an API here when its first CI build fails to link.
_PLAYBACK_RUNTIME_RESOLVED_APIS = frozenset({
    "hipDeviceFlushGPUDirectRDMAWrites",
    "hipExtMemcpyBatchAsync",
})

_PLAYBACK_ARG_BRIDGES: Dict[str, Dict[str, object]] = {
    # HIP 8.0 replaced `int device` with a `hipMemLocation` struct. The recorded field keeps the int ordinal, so the bridge rebuilds a hipMemLocation from it.
    'hipMemAdvise': {
        'guard': 'HIP_TARGET_API_VERSION >= 800 && !defined(HIP_ABI_IMPL)',
        'param': 'device',
        'pre': [
            '  hipMemLocation _loc_{name}{{}};',
            '  _loc_{name}.type = hipMemLocationTypeDevice;',
            '  _loc_{name}.id   = (int)a->{name};',
        ],
        'expr': '_loc_{name}',
    },
}


def _apply_arg_bridge(bridge: Dict[str, object],
                      entry: 'ApiEntry',
                      call_args: List[str]) -> Tuple[List[str], List[str]]:
    """Return (pre_lines, call_args) with the bridged argument substituted."""
    if len(call_args) != len(entry.params):
        sys.exit(f"ERROR: _PLAYBACK_ARG_BRIDGES[{entry.name!r}] cannot be applied: "
                 f"the generated call has {len(call_args)} arguments for "
                 f"{len(entry.params)} dispatch-table parameters, so arguments no "
                 f"longer map to parameters by position. Update _apply_arg_bridge.")
    pname = str(bridge['param'])
    idx = next((i for i, p in enumerate(entry.params) if p.name == pname), None)
    if idx is None:
        sys.exit(f"ERROR: _PLAYBACK_ARG_BRIDGES[{entry.name!r}] names parameter "
                 f"{pname!r}, which is not a parameter of the dispatch-table "
                 f"typedef. Fix the bridge or remove it.")
    pre = [ln.format(name=pname) for ln in bridge['pre']]      # type: ignore[index]
    args = list(call_args)
    args[idx] = str(bridge['expr']).format(name=pname)
    return pre, args


class RawSlotError(Exception):
    """An argument would reach a HIP call as a raw cast of an archive integer.

    The archive is untrusted input. A pointer or an opaque handle in a record is
    an address from the capturing process: handing it to HIP unchanged lets the
    archive choose what the runtime dereferences. Every such slot is either
    translated through the context maps or the API is refused as unreplayable;
    _playback_arg raises this for the second case and generate_playback_shim
    turns it into a refusing handler.
    """


# Enumerations not in _ENUM_TYPES are recorded as an 8-byte placeholder but are plain numbers, so casting back is a conversion.
# Listing them keeps the raw-slot rule from refusing their APIs without changing the header layout.
_NUMERIC_ENUMS_RECORDED_AS_WIDE = {
    "hipFlushGPUDirectRDMAWritesTarget",
    "hipFlushGPUDirectRDMAWritesScope",
}

_C_ARITHMETIC_TYPE_RE = re.compile(
    r'(?:(?:unsigned|signed|short|long|char|int|float|double|bool|_Bool)(?: |$))+'
    r'|(?:u?int(?:8|16|32|64)_t|size_t|ssize_t|ptrdiff_t|intptr_t|uintptr_t)')


def _raw_slot_kind(raw_type: str) -> Optional[str]:
    """What a raw `(T)a->field` cast of this parameter type would hand to HIP.

    Returns None when the cast is a numeric conversion (an integer, a float, an
    enum, a by-value struct carried as bytes), and "pointer", "handle" or
    "composite" when the recorded integer would be taken as an address, an
    opaque handle or a struct this generator cannot rebuild.
    """
    t = raw_type.strip()
    t_nc = re.sub(r'\b(const|volatile|restrict|struct|enum)\b', '', t).strip()
    t_nc = re.sub(r'\s+', ' ', t_nc)
    if '*' in t_nc:
        return "pointer"
    base = t_nc.split()[0]
    if base in ("dim3", "uint3") or base in BY_VALUE_STRUCTS:
        return None
    if _C_ARITHMETIC_TYPE_RE.fullmatch(t_nc):
        return None
    if base == "hipDevice_t":  # an int ordinal, not a pointer
        return None
    if base in _HANDLE_TYPES or base in _NON_CASTABLE_TYPES:
        return "handle"
    if base in _ENUM_TYPES or base in _NUMERIC_ENUMS_RECORDED_AS_WIDE:
        return None
    for key, _mapped in _SCALAR_MAP:
        if t_nc == key:
            return None
    return "composite"


def _raw_slot_error(entry: ApiEntry, name: str, t: str, kind: str) -> RawSlotError:
    return RawSlotError(
        f"argument {name} ({t.strip()}) is a {kind} recorded as an address from "
        f"the capturing process; replay has no translation for it, so passing it "
        f"to HIP would let the archive choose what the runtime dereferences")


def _playback_arg(entry: ApiEntry, p: Param, name: str,
                  pre_lines: List[str]) -> str:
    """Return the expression to pass to the real HIP API during playback."""
    t = p.raw_type.strip()
    base = _get_base_type(t)
    is_output = _is_output_ptr_param(p)

    # A declared out buffer: the callee writes into a local of the right size, since the recorded address belongs to the capturing process.
    out_bytes = _OUT_BUFFER_PARAMS.get(entry.name, {}).get(name)
    if out_bytes:
        pre_lines.append(f"  alignas(8) unsigned char _outbuf_{name}"
                         f"[{out_bytes}]{{}};")
        return f"({t})_outbuf_{name}"

    bv = by_value_struct(t)
    if bv is not None:
        pre_lines.append(f"  {base} _v_{name}{{}};")
        pre_lines.append(f"  std::memcpy(&_v_{name}, a->{name}_bytes,"
                         f" sizeof(_v_{name}));")
        pre_lines.extend(_deref_translate_lines(f"_v_{name}", bv.ptr_members))
        return f"_v_{name}"

    d = deref_for_param(entry.name, name)
    if d is not None and d.playback == "auto":
        if d.string:
            # A fixed-size character field the archive need not NUL-terminate: copy into a local the field's size and force the last byte to NUL, so the callee reads inside the buffer.
            # The static_assert catches a field of another size, which would make the copy read past the record or stop short.
            pre_lines.append(f'  static_assert(sizeof(a->{d.bytes_field}) == {d.max_count}, '
                             f'"{entry.name}: {d.bytes_field} is not {d.max_count} bytes");')
            pre_lines.append(f"  char _str_{name}[{d.max_count}]{{}};")
            pre_lines.append(f"  if (a->{d.present_field}) {{")
            pre_lines.append(f"    std::memcpy(_str_{name}, a->{d.bytes_field},"
                             f" sizeof(_str_{name}));")
            pre_lines.append(f"    _str_{name}[sizeof(_str_{name}) - 1] = '\\0';")
            pre_lines.append(f"  }}")
            return (f"(a->{d.present_field} ? ({t})_str_{name}"
                    f" : ({t})nullptr)")
        if d.is_array:
            pre_lines.append(f"  {d.ctype} _d_{name}[{d.max_count}]{{}};")
            pre_lines.append(f"  uint32_t _d_{name}_n = a->{d.count_field} >"
                             f" {d.max_count}u ? {d.max_count}u : a->{d.count_field};")
            pre_lines.append(f"  (void)_d_{name}_n;")
            if d.direction != "out":
                pre_lines.append(f"  if (a->{d.present_field})")
                pre_lines.append(f"    std::memcpy(_d_{name}, a->{d.bytes_field},"
                                 f" static_cast<size_t>(_d_{name}_n) * sizeof({d.ctype}));")
                for member in d.ptr_members:
                    pre_lines.append(f"  for (uint32_t _i = 0; _i < _d_{name}_n; ++_i)")
                    pre_lines.append(f"    _d_{name}[_i].{member} = ctx.translate_ptr("
                                     f"reinterpret_cast<uint64_t>(_d_{name}[_i].{member}));")
                for (member, htype) in d.handle_members:
                    hfn = _PLAYBACK_HANDLE_TRANSLATE[htype]
                    pre_lines.append(f"  for (uint32_t _i = 0; _i < _d_{name}_n; ++_i)")
                    pre_lines.append(f"    _d_{name}[_i].{member} = ({htype}){hfn}("
                                     f"reinterpret_cast<uint64_t>(_d_{name}[_i].{member}));")
                if d.elem_ptr:
                    pre_lines.append(f"  for (uint32_t _i = 0; _i < _d_{name}_n; ++_i)")
                    pre_lines.append(f"    _d_{name}[_i] = ctx.translate_ptr("
                                     f"reinterpret_cast<uint64_t>(_d_{name}[_i]));")
                # An array of handles: each element is a recorded handle. A graph dependency list holds capturing-process node addresses; untranslated, the runtime would reject them or wire the node to whatever lives at that address here.
                if d.elem_handle:
                    efn = _PLAYBACK_HANDLE_TRANSLATE[d.elem_handle]
                    gparam = _graph_param(entry)
                    pre_lines.append(f"  for (uint32_t _i = 0; _i < _d_{name}_n; ++_i) {{")
                    pre_lines.append(f"    {d.elem_handle} _live = ({d.elem_handle}){efn}("
                                     f"reinterpret_cast<uint64_t>(_d_{name}[_i]));")
                    pre_lines.append(f"    if (!_live && _d_{name}[_i]) {{")
                    pre_lines.append(f"      fprintf(stderr, \"[HRR] {entry.name}: dependency \"")
                    pre_lines.append(f"              \"0x%llx was never built at replay, so this call is \"")
                    pre_lines.append(f"              \"skipped and the graph it belongs to is marked \"")
                    pre_lines.append(f"              \"incomplete; instantiating that graph fails loudly \"")
                    pre_lines.append(f"              \"rather than running one that is missing an \"")
                    pre_lines.append(f"              \"ordering constraint.\\n\",")
                    pre_lines.append(f"              (unsigned long long)reinterpret_cast<uint64_t>(_d_{name}[_i]));")
                    if gparam:
                        pre_lines.append(f"      ctx.mark_graph_incomplete(a->{gparam},"
                                         f" \"{entry.name}\");")
                    pre_lines.append(f"      return hipSuccess;")
                    pre_lines.append(f"    }}")
                    pre_lines.append(f"    _d_{name}[_i] = _live;")
                    pre_lines.append(f"  }}")
            # An empty array is passed as null: hipDrvGraphAddMemFreeNode (hip_graph.cpp:3502) rejects a non-null dependency pointer with a zero count, and nodes added without dependencies have a zero count.
            return f"({t})(_d_{name}_n ? _d_{name} : nullptr)"
        pre_lines.append(f"  {d.ctype} _d_{name}{{}};")
        # An output pointee is written by the call, so only the buffer size matters; the recorded value is kept for fidelity. A local of the pointee's own type also stops the generic output-pointer path from giving a 64-byte write an 8-byte void*.
        if d.direction == "out":
            return f"({t})&_d_{name}"
        pre_lines.append(f"  if (a->{d.present_field})")
        pre_lines.append(f"    std::memcpy(&_d_{name}, a->{d.bytes_field},"
                         f" sizeof(_d_{name}));")
        pre_lines.extend(_deref_translate_lines(f"_d_{name}", d.ptr_members,
                                                handle_members=d.handle_members))
        if d.direction == "inout":
            return f"({t})&_d_{name}"
        return f"(a->{d.present_field} ? ({t})&_d_{name} : ({t})nullptr)"

    if is_output:
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
        pre_lines.append(f"  void* _out_{name} = nullptr;")
        return f"(void**)&_out_{name}"

    if base == 'hipCtx_t' and entry.name in _LIVE_CTX_PLAYBACK_APIS:
        return f"hrr_live_ctx(a->{name})"

    if base in _PLAYBACK_HANDLE_TRANSLATE:
        fn = _PLAYBACK_HANDLE_TRANSLATE[base]
        return f"({t}){fn}(a->{name})"
    if 'hipArray' in base:
        return f"(hipArray_t)ctx.translate_array(a->{name})"
    if 'hipMipmappedArray' in base:
        return f"(hipMipmappedArray_t)ctx.translate_mipmapped(a->{name})"

    if base == 'hipDeviceptr_t':
        return f"(hipDeviceptr_t)ctx.translate_ptr(a->{name})"

    is_ptr = '*' in t
    if is_ptr and base == 'void':
        if any(kw in name.lower() for kw in ('dst', 'dev', 'ptr', 'buf', 'src')):
            return f"ctx.translate_ptr(a->{name})"

    if base in ('dim3', 'uint3'):
        pre_lines.append(f"  dim3 _dim_{name}(a->{name}_x, a->{name}_y, a->{name}_z);")
        return f"_dim_{name}"

    if base in ('hipIpcEventHandle_t', 'hipIpcMemHandle_t', 'hipPitchedPtr',
                'hipExtent', 'hipPos', 'hipMemLocation', 'hipChannelFormatDesc'):
        pre_lines.append(f"  {t} _s_{name}{{}};")
        return f"_s_{name}"

    # Function pointers are capturing-process addresses with nothing to translate to, so an API taking one is refused.
    if base in ('hipStreamCallback_t', 'hipHostFn_t'):
        raise _raw_slot_error(entry, name, t, "function pointer")

    # Unhandled non-const pointer to a non-void scalar: treat as an output pointer. The recorded value is an invalid address at replay, so pass a zero-initialised local.
    is_ptr = '*' in t
    is_const_ptr = 'const' in t and is_ptr
    if is_ptr and not is_const_ptr and base != 'void':
        inner = t.replace('*', '', 1).strip()
        pre_lines.append(f"  {inner} _out_{name}{{}};")
        return f"&_out_{name}"

    # Anything left is a numeric slot (integer, float, enum): the cast is a conversion. A pointer, handle or struct would reach HIP as an archive integer used as an address, so refuse the API.
    kind = _raw_slot_kind(t)
    if kind is not None:
        raise _raw_slot_error(entry, name, t, kind)
    return f"({t})a->{name}"


def _refusing_playback_shim(entry: ApiEntry, reason: str) -> str:
    """Playback handler that refuses the API by name instead of calling it."""
    sname = f"hrr_args_{entry.name}"
    fname = f"playback_{entry.name}"
    gparam = _graph_param(entry)
    # Unlike ERROR_STUB_PLAYBACK_APIS this returns an error, so replay stops here unless --continue-on-error. Under that flag the graph is now short a node; marking it makes the later instantiate refuse it.
    mark = (f"  const auto* a = reinterpret_cast<const {sname}*>(payload);\n"
            f"  ctx.mark_graph_incomplete(a->{gparam}, \"{entry.name}\");\n"
            if gparam else "  (void)payload;\n")
    return (f"static hipError_t {fname}"
            f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
            f"{mark}"
            f"  hrr_note_unreplayable(ctx, \"{entry.name}\",\n"
            f"                        \"{reason}\");\n"
            f"  return hipErrorNotSupported;\n"
            f"}}\n")


def playback_defaults_to_noop(entry: ApiEntry) -> bool:
    """True when a generated call would pass an untranslated opaque value.

    DIRECT_PLAYBACK_APIS keeps the generated call for APIs that already had
    one. The API matrix (tools/api-matrix/derive_manifest.py) calls this too,
    so its expected replay class cannot drift from what the shim emits.
    """
    return (entry.name not in DIRECT_PLAYBACK_APIS
            and any(_get_base_type(p.raw_type) in _UNTRANSLATED_PLAYBACK_TYPES
                    for p in entry.params))


def _noop_playback_shim(api: str) -> str:
    """Warn once and return success. The call's effect is not reproduced.

    The flag is atomic: --multi-thread replay can enter one handler from two
    threads, because non-ordering events advance next_seq before dispatch.
    """
    return (f"static hipError_t playback_{api}"
            f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
            f"  (void)ctx; (void)payload;\n"
            f"  static std::atomic<bool> warned{{false}};\n"
            f"  if (!warned.exchange(true)) {{\n"
            f"    fprintf(stderr, \"[HRR] NOOP playback handler called for {api} — \"\n"
            f"            \"this API is not replayed; results may differ from capture.\\n\");\n"
            f"  }}\n"
            f"  return hipSuccess;\n"
            f"}}\n")


def generate_playback_shim(entry: ApiEntry) -> str:
    """Generate playback function for one API."""
    sname = f"hrr_args_{entry.name}"
    fname = f"playback_{entry.name}"
    sig   = f"static hipError_t {fname}(PlaybackContext& ctx, const uint8_t* payload)"

    if entry.reserved:
        return (f"static hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
                f"  (void)ctx; (void)payload;\n"
                f"  return hipSuccess;\n"
                f"}}\n")

    # Error-stub APIs: graph work whose arguments the archive cannot rebuild. Warn once per API, mark the graph incomplete, return hipSuccess.
    # The hard failure is at hipGraphInstantiate/hipGraphInstantiateWithFlags, which refuse an incomplete graph (finding H1); programs that never instantiate keep replaying.
    if entry.name in ERROR_STUB_PLAYBACK_APIS:
        gparam = _graph_param(entry)
        mark = (f"  const auto* a = reinterpret_cast<const {sname}*>(payload);\n"
                f"  ctx.mark_graph_incomplete(a->{gparam}, \"{entry.name}\");\n"
                if gparam else "  (void)ctx; (void)payload;\n")
        return (f"static hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
                f"{mark}"
                f"  static bool warned = false;\n"
                f"  if (!warned) {{\n"
                f"    warned = true;\n"
                f"    fprintf(stderr, \"[HRR] {entry.name}: not reconstructable at \"\n"
                f"            \"replay, so the call is skipped and the graph it belongs \"\n"
                f"            \"to is marked incomplete; instantiating that graph fails \"\n"
                f"            \"loudly rather than running a graph that is missing \"\n"
                f"            \"work.\\n\");\n"
                f"  }}\n"
                f"  return hipSuccess;\n"
                f"}}\n")

    if entry.name in UNREPLAYABLE_PLAYBACK_APIS:
        return _refusing_playback_shim(entry, UNREPLAYABLE_PLAYBACK_APIS[entry.name])
    if entry.name in PLAYBACK_ONLY_UNREPLAYABLE_APIS:
        return _refusing_playback_shim(entry, PLAYBACK_ONLY_UNREPLAYABLE_APIS[entry.name])

    if entry.name in NOOP_PLAYBACK_APIS:
        return _noop_playback_shim(entry.name)

    if entry.name in CUSTOM_PLAYBACK_BODIES:
        return CUSTOM_PLAYBACK_BODIES[entry.name]

    if entry.name in MANUAL_PLAYBACK_APIS:
        return (f"extern hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload);\n")

    lines = []
    lines.append(f"{sig} {{")

    if entry.table == "compiler":
        lines.append(f"  (void)ctx; (void)payload;")
        lines.append(f"  return hipSuccess;")
        lines.append("}")
        return "\n".join(lines) + "\n"

    # An opaque handle with no playback translation would pass a capturing-process address, so that API replays as a no-op until a reviewed handler is added.
    # DIRECT_PLAYBACK_APIS keeps the generated call for APIs that already had one; editing hip_playback_generated.cpp does not.
    if playback_defaults_to_noop(entry):
        return _noop_playback_shim(entry.name)

    lines.append(f"  const auto* a = reinterpret_cast<const {sname}*>(payload);")

    is_alloc_create  = entry.name in _ALLOC_CREATE_APIS
    is_alloc_free    = entry.name in _ALLOC_FREE_APIS
    is_hdl_create    = entry.name in _HANDLE_CREATE_APIS
    is_hdl_destroy   = entry.name in _HANDLE_DESTROY_APIS

    # A node or executable graph never built here cannot be mutated. A null handle returns an error and dispatch treats any handler error as fatal, so a graph HRR already refused at instantiate would take the replay down through its follow-up mutations.
    # The complaint was already made; skip.
    for p in entry.params:
        pt = p.raw_type.strip()
        if '*' in pt or not p.name:
            continue
        if _get_base_type(pt) not in ('hipGraphNode_t', 'hipGraphExec_t'):
            continue
        tfn = _PLAYBACK_HANDLE_TRANSLATE[_get_base_type(pt)]
        lines.append(f"  if (a->{p.name} != 0 && {tfn}(a->{p.name}) == nullptr) {{")
        lines.append(f"    static bool warned = false;")
        lines.append(f"    if (!warned) {{")
        lines.append(f"      warned = true;")
        lines.append(f"      fprintf(stderr, \"[HRR] {entry.name}: {p.name} 0x%llx was \"")
        lines.append(f"              \"never built at replay, so this call is skipped.\\n\",")
        lines.append(f"              (unsigned long long)a->{p.name});")
        lines.append(f"    }}")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    if is_alloc_free:
        rec_param = _ALLOC_FREE_APIS[entry.name]
        lines.append(f"  uint64_t _rec_ptr = a->{rec_param};")
        # Releasing memory replay never had is not an unmapped pointer in the policy's sense: use lookup_ptr, no note.
        lines.append(f"  void*    _live_ptr = ctx.lookup_ptr(_rec_ptr);")
        lines.append(f"  if (_live_ptr == nullptr) return hipSuccess;")
    if is_hdl_destroy:
        rec_param, hdl_type = _HANDLE_DESTROY_APIS[entry.name]
        translate_fn = _PLAYBACK_HANDLE_TRANSLATE.get(hdl_type, '')
        if translate_fn:
            lines.append(f"  uint64_t _rec_hdl = a->{rec_param};")

    pre_lines: List[str] = []
    call_args: List[str] = []
    unnamed = 0
    try:
        for p in entry.params:
            name = p.name or f"p{unnamed}"
            if not p.name: unnamed += 1

            if is_alloc_free and name == _ALLOC_FREE_APIS[entry.name]:
                call_args.append("_live_ptr")
                continue
            if is_hdl_destroy and name == _HANDLE_DESTROY_APIS[entry.name][0]:
                hdl_type = _HANDLE_DESTROY_APIS[entry.name][1]
                translate_fn = _PLAYBACK_HANDLE_TRANSLATE.get(hdl_type, '')
                if translate_fn:
                    call_args.append(f"({hdl_type}){translate_fn}(_rec_hdl)")
                else:
                    raise _raw_slot_error(entry, name, hdl_type, "handle")
                continue

            # The element count of a deref'd array is the number of elements the event carries, not the recorded count when the array overflowed inline capacity.
            array_owner = next((d for d in deref_specs(entry.name)
                                if d.is_array and d.count == name
                                and d.playback == "auto"), None)
            if array_owner is not None:
                call_args.append(f"({p.raw_type.strip()})_d_{array_owner.param}_n")
                continue

            call_args.append(_playback_arg(entry, p, name, pre_lines))
    except RawSlotError as err:
        AUTO_UNREPLAYABLE_PLAYBACK_APIS[entry.name] = str(err)
        return _refusing_playback_shim(entry, str(err))

    for pl in pre_lines:
        lines.append(pl)

    if entry.name in ATTR_UNION_WINDOW_APIS:
        id_field, enumerator = ATTR_UNION_WINDOW_APIS[entry.name]
        lines.append(f"  if (a->{id_field} == {enumerator}) {{")
        lines.append("    _d_value.accessPolicyWindow.base_ptr = ctx.translate_ptr(")
        lines.append("        reinterpret_cast<uint64_t>(_d_value.accessPolicyWindow.base_ptr));")
        lines.append("  }")

    # A JIT/link option array may carry pointers the runtime dereferences (log buffers, symbol tables), with no translation; such a record is refused, naming the API.
    if entry.name in JIT_OPTION_ARRAY_APIS:
        lines.append("  if (hrr::jit_options_carry_pointer(_d_options, _d_options_n)) {")
        lines.append(f'    hrr_note_unreplayable(ctx, "{entry.name}",')
        lines.append('        "a JIT or link option carries a pointer (a log buffer or a symbol "')
        lines.append('        "table) that points into the capturing process; replay cannot supply "')
        lines.append('        "one, and passing the recorded value would let the archive choose "')
        lines.append('        "what the runtime writes to");')
        lines.append("    return hipErrorNotSupported;")
        lines.append("  }")

    args_str = ", ".join(call_args)
    void_ret = _is_void_return(entry.ret_type)

    resolved = entry.name in _PLAYBACK_RUNTIME_RESOLVED_APIS
    if resolved:
        if entry.name in _PLAYBACK_ARG_BRIDGES:
            sys.exit(f"ERROR: {entry.name} is runtime-resolved and has an arg bridge; "
                     f"the two cannot be combined")
        lines.append(f"  static const auto _fn = reinterpret_cast<decltype(&{entry.name})>(")
        lines.append(f"      hrr_runtime_symbol(\"{entry.name}\"));")
        lines.append(f"  if (_fn == nullptr) {{")
        lines.append(f"    static bool warned = false;")
        lines.append(f"    if (!warned) {{")
        lines.append(f"      warned = true;")
        lines.append(f"      fprintf(stderr, \"[HRR] {entry.name} is not exported by this \"")
        lines.append(f"              \"HIP runtime; skipping it during replay, results may \"")
        lines.append(f"              \"differ from capture.\\n\");")
        lines.append(f"    }}")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    def _emit_call(target_lines: List[str], a_str: str) -> None:
        callee = "_fn" if resolved else entry.name
        if void_ret:
            target_lines.append(f"  {callee}({a_str});")
        else:
            target_lines.append(f"  hipError_t _r = (hipError_t){callee}({a_str});")

    bridge = _PLAYBACK_ARG_BRIDGES.get(entry.name)
    if bridge:
        b_pre, b_args = _apply_arg_bridge(bridge, entry, call_args)
        lines.append(f"#if {bridge['guard']}")
        lines.extend(b_pre)
        _emit_call(lines, ", ".join(b_args))
        lines.append("#else")
        _emit_call(lines, args_str)
        lines.append("#endif")
    else:
        _emit_call(lines, args_str)
    ret_expr = "hipSuccess" if void_ret else "_r"

    # A call that failed at capture and fails identically here was reproduced: return success so dispatch_event does not abort.
    # Only hand-written capture shims record failing calls, so elsewhere a->ret is 0.
    if not void_ret and _get_base_type(entry.ret_type) == 'hipError_t':
        lines.append(f"  if (_r != hipSuccess && a->ret != 0 &&"
                     f" static_cast<int32_t>(_r) == a->ret) {{")
        lines.append(f"    hrr_note_recorded_error(ctx, \"{entry.name}\", a->ret);")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    success_cond = "true" if void_ret else "_r == hipSuccess"

    if is_alloc_create:
        rec_param, sz_param = _ALLOC_CREATE_APIS[entry.name]
        kind = ("AllocKind::HostMalloc"
                if entry.name in _HOST_ALLOC_CREATE_APIS else "AllocKind::Device")
        lines.append(f"  if ({success_cond}) {{")
        lines.append(f"    ctx.record_alloc(a->{rec_param}, _out_{rec_param},"
                     f" static_cast<size_t>(a->{sz_param}), {kind});")
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

    for extra in _POST_CALL_EXTRA.get(entry.name, []):
        lines.append(f"  {extra}")

    lines.append(f"  return {ret_expr};")
    lines.append("}")

    if entry.name not in WINDOWS_EXPORTED_APIS:
        return _win_unexported_shim(entry.name, lines)

    return "\n".join(lines) + "\n"


def _win_unexported_shim(api: str, lines: List[str]) -> str:
    """Wrap a generated playback body so Windows gets a warning instead of a call.

    Same contract as NOOP_PLAYBACK_APIS: warn once per process and report
    success, so one unexportable API does not abort an otherwise fine replay.
    """
    return "\n".join([
        lines[0],                       # signature and opening brace
        "#ifdef _WIN32",
        "  (void)ctx; (void)payload;",
        "  static bool warned = false;",
        "  if (!warned) {",
        "    warned = true;",
        f"    fprintf(stderr, \"[HRR] {api} is not exported by amdhip64 on \"",
        "            \"Windows; skipping it during replay, results may differ \"",
        "            \"from capture.\\n\");",
        "  }",
        "  return hipSuccess;",
        "#else",
        *lines[1:-1],                   # the real call
        "#endif",
        lines[-1],                      # closing brace
    ]) + "\n"


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
        if e.name in VARIABLE_LENGTH_KERNEL_LAUNCH_APIS:
            min_expr = _VARIABLE_KERNEL_LAUNCH_MIN_PAYLOAD_EXPR
            note = "variable-length kernel launch"
        else:
            min_expr = f"static_cast<uint32_t>(sizeof(hrr_args_{e.name}))"
            note = enum_name
        lines.append(f"    {min_expr},  // [{idx}] {note}")
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
# Public API signature validation
# ---------------------------------------------------------------------------
# Capture goes through the dispatch table; playback calls the public API by name. Casts derived from the table are sound only while the two declarations agree.
# The table is append-only ABI, but hip_runtime_api.h may re-declare an API under a HIP_TARGET_API_VERSION guard.
# This pass parses the public header and refuses to generate when a cast would disagree with its callee.


@dataclass
class PublicDecl:
    """One non-template declaration of a public HIP API."""
    types:   Tuple[str, ...]
    guarded: bool          # inside a HIP_TARGET_API_VERSION / HIP_ABI_IMPL #if
    line:    int


def _strip_comments_keep_lines(text: str) -> str:
    """Strip comments while preserving line numbering."""
    def blank(m: 're.Match[str]') -> str:
        return re.sub(r'[^\n]', ' ', m.group(0))
    text = re.sub(r'/\*.*?\*/', blank, text, flags=re.DOTALL)
    text = re.sub(r'//[^\n]*', blank, text)
    return text


def _collect_typedef_aliases(text: str) -> Dict[str, str]:
    """Simple `typedef <existing> <alias>;` pairs, resolved transitively.

    Lets hipDeviceProp_tR0600 (dispatch table) compare equal to
    hipDeviceProp_t (public header) while they remain the same type.
    """
    alias: Dict[str, str] = {}
    for m in re.finditer(r'\btypedef\s+([A-Za-z_][A-Za-z0-9_]*)\s+'
                         r'([A-Za-z_][A-Za-z0-9_]*)\s*;', text):
        existing, name = m.group(1), m.group(2)
        if existing != name:
            alias[name] = existing
    resolved: Dict[str, str] = {}
    for name in alias:
        seen, cur = {name}, alias[name]
        while cur in alias and cur not in seen:
            seen.add(cur)
            cur = alias[cur]
        resolved[name] = cur
    return resolved


def _norm_type(raw: str, aliases: Dict[str, str]) -> str:
    """Canonical spelling of a parameter type, names and defaults removed."""
    t = re.sub(r'__dparm\s*\([^)]*\)', '', raw).split('=')[0]
    t = re.sub(r'\s+', ' ', t).strip()
    t = t.replace('*', ' * ').replace('&', ' & ')
    t = re.sub(r'\s+', ' ', t).strip()
    return ' '.join(aliases.get(tok, tok) for tok in t.split())


def _param_types(inner: str, aliases: Dict[str, str]) -> Optional[Tuple[str, ...]]:
    inner = inner.strip()
    if not inner or inner == 'void':
        return ()
    out: List[str] = []
    for raw in _split_params(inner):
        prm = _parse_param(re.sub(r'__dparm\s*\([^)]*\)', '', raw).split('=')[0])
        if prm is None:
            return None
        out.append(_norm_type(prm.raw_type, aliases))
    return tuple(out)


def _version_guard_stack(text: str) -> Dict[int, List[str]]:
    """line number -> enclosing version/ABI-dependent #if conditions."""
    stack: List[str] = []
    per_line: Dict[int, List[str]] = {}
    for lineno, line in enumerate(text.splitlines(), 1):
        s = line.strip()
        if re.match(r'#\s*if', s):
            stack.append(s)
        elif re.match(r'#\s*el(se|if)', s):
            if stack:
                stack[-1] = s
        elif re.match(r'#\s*endif', s):
            if stack:
                stack.pop()
        per_line[lineno] = [g for g in stack
                            if 'API_VERSION' in g or 'ABI_IMPL' in g]
    return per_line


def parse_public_api_signatures(path: Path) -> Dict[str, List[PublicDecl]]:
    """Parse hip_runtime_api.h for the public declaration of each hip* API.

    Template overloads are skipped: they coexist with the C declaration rather
    than replacing it, so they never change which function a C-typed call
    resolves to.
    """
    text    = _strip_comments_keep_lines(path.read_text(errors='replace'))
    aliases = _collect_typedef_aliases(text)
    guards  = _version_guard_stack(text)

    def line_of(pos: int) -> int:
        return text.count('\n', 0, pos) + 1

    decls: Dict[str, List[PublicDecl]] = {}
    pattern = (r'\b(?:hipError_t|void|int|unsigned|hip[A-Za-z0-9_]+_t)'
               r'\s+\*?\s*(hip[A-Za-z0-9_]+)\s*\(')
    for m in re.finditer(pattern, text):
        name = m.group(1)
        back = text[max(0, m.start() - 400):m.start()]
        cut  = max(back.rfind(';'), back.rfind('}'), back.rfind('#'))
        if 'template' in back[cut + 1:]:
            continue
        try:
            _end, inner = _extract_balanced_parens(text, text.index('(', m.start(1)))
        except ValueError:
            continue
        types = _param_types(inner, aliases)
        if types is None:
            continue
        ln = line_of(m.start())
        decls.setdefault(name, []).append(
            PublicDecl(types=types, guarded=bool(guards.get(ln)), line=ln))
    return decls


def validate_playback_signatures(entries: List[ApiEntry],
                                 public: Dict[str, List[PublicDecl]],
                                 public_path: Path, silent: bool = False) -> None:
    """Fail generation if a playback cast would disagree with its callee.

    An API is safe when the public header offers a declaration that both
    matches the dispatch-table signature and is unconditionally visible, so
    the generated call resolves to it in every build configuration.
    """
    aliases  = _collect_typedef_aliases(_strip_comments_keep_lines(
        public_path.read_text(errors='replace')))
    problems: List[str] = []
    checked  = skipped = 0

    for e in entries:
        if (e.reserved or e.table == "compiler"
                or e.name in MANUAL_PLAYBACK_APIS
                or e.name in NOOP_PLAYBACK_APIS
                or e.name in ERROR_STUB_PLAYBACK_APIS
                or e.name in CUSTOM_PLAYBACK_BODIES):
            continue
        variants = public.get(e.name)
        if not variants:
            skipped += 1
            continue
        table_types = tuple(_norm_type(p.raw_type, aliases) for p in e.params)
        checked += 1

        exact = [d for d in variants if d.types == table_types]
        if any(not d.guarded for d in exact):
            continue                      # always resolvable — safe
        if len(exact) == len(variants) and exact:
            continue                      # only spelling/guard, one signature

        if e.name in _PLAYBACK_ARG_BRIDGES:
            continue                      # explicitly bridged, see the table

        detail = [f"    dispatch table : ({', '.join(table_types) or 'void'})"]
        for d in sorted(variants, key=lambda x: x.line):
            tag = "guarded" if d.guarded else "unconditional"
            detail.append(f"    public L{d.line:<6} ({', '.join(d.types) or 'void'})"
                          f"   [{tag}]")
        problems.append(f"  {e.name}\n" + "\n".join(detail))

    if problems:
        sys.exit(
            "ERROR: playback would cast arguments using the dispatch-table\n"
            "signature, but the public HIP API it calls is declared differently\n"
            f"in {public_path}.\n"
            "Playback calls the public API, not the table, so these casts would\n"
            "be wrong (or silently pick the wrong overload) in some build\n"
            "configuration.\n\n"
            + "\n\n".join(problems) +
            "\n\nResolve each by adding an entry to _PLAYBACK_ARG_BRIDGES with the\n"
            "guard and conversion the public header requires.")

    if not silent:
        print(f"  Public API signatures: {checked} verified against "
              f"{public_path.name}, {skipped} declared elsewhere (skipped)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # The portable header and playback live under projects/hrr; the capture
    # shim stays inside CLR because it compiles into amdhip64.
    tools_dir       = Path(__file__).resolve().parent   # projects/hrr/tools/
    hrr_project_dir = tools_dir.parent                  # projects/hrr/
    projects_dir    = hrr_project_dir.parent            # projects/
    clr_root        = projects_dir / "clr"              # projects/clr/
    clr_hrr_dir     = clr_root / "hipamd" / "src" / "hrr"  # capture stays in CLR
    default_input    = clr_root / "hipamd/include/hip/amd_detail/hip_api_trace.hpp"
    default_public   = projects_dir / "hip/include/hip/hip_runtime_api.h"
    default_header   = hrr_project_dir / "include" / "hrr" / "hrr_api_args.h"
    default_capture  = clr_hrr_dir / "hip_capture_generated.cpp"
    default_playback = hrr_project_dir / "playback" / "hip_playback_generated.cpp"
    default_manual_capture = clr_hrr_dir / "hip_capture.cpp"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input",           default=str(default_input),
                        help="Path to hip_api_trace.hpp")
    parser.add_argument("--public-header",   default=str(default_public),
                        help="Path to hip_runtime_api.h (public API declarations)")
    parser.add_argument("--skip-public-api-check", action="store_true",
                        help="Skip playback/public-API signature validation. "
                             "Unsafe: a playback cast may disagree with the "
                             "API it is passed to.")
    parser.add_argument("--output-header",   default=str(default_header),
                        help="Path to generated hrr_api_args.h")
    parser.add_argument("--output-capture",  default=str(default_capture),
                        help="Path to generated hip_capture_generated.cpp")
    parser.add_argument("--output-playback", default=str(default_playback),
                        help="Path to generated hip_playback_generated.cpp")
    parser.add_argument("--check-hrr-coverage", action="store_true",
                        help="Validate parseable dispatch slots and manual capture shims")
    parser.add_argument("--manual-capture-source", default=str(default_manual_capture),
                        help="Path to hand-written hip_capture.cpp")
    parser.add_argument("--silent", action="store_true",
                        help="Suppress normal progress output; warnings and errors remain visible")
    args = parser.parse_args()

    def log(message: str) -> None:
        if not args.silent:
            print(message)

    in_path       = Path(args.input)
    public_path   = Path(args.public_header)
    header_path   = Path(args.output_header)
    capture_path  = Path(args.output_capture)
    playback_path = Path(args.output_playback)
    manual_capture_path = Path(args.manual_capture_source)

    if not in_path.exists():
        sys.exit(f"ERROR: input file not found: {in_path}")

    log(f"Parsing {in_path} ...")
    entries = parse_hip_api_trace(in_path)
    n_compiler       = sum(1 for e in entries if e.table == "compiler")
    n_runtime        = sum(1 for e in entries if e.table == "runtime")
    n_reserved       = sum(1 for e in entries if e.reserved)
    n_manual_cap     = sum(1 for e in entries if e.name in MANUAL_CAPTURE_APIS)
    n_manual_play    = sum(1 for e in entries if e.name in MANUAL_PLAYBACK_APIS)
    n_noop_play      = sum(1 for e in entries if e.name in NOOP_PLAYBACK_APIS)
    log(f"  Found {n_compiler} compiler + {n_runtime} runtime = {len(entries)} total"
        f" ({n_reserved} reserved slots)")

    # Cross-validate classification sets against parsed API names: an entry absent from hip_api_trace.hpp is silent dead weight (typo or removed API).
    parsed_names: Set[str] = {e.name for e in entries}
    unknown: Dict[str, List[str]] = {}
    for set_name, api_set in [
        ("MANUAL_CAPTURE_APIS",  MANUAL_CAPTURE_APIS),
        ("MANUAL_PLAYBACK_APIS", MANUAL_PLAYBACK_APIS),
        ("NOOP_PLAYBACK_APIS",   NOOP_PLAYBACK_APIS),
        ("DIRECT_PLAYBACK_APIS", DIRECT_PLAYBACK_APIS),
        ("ERROR_STUB_PLAYBACK_APIS", ERROR_STUB_PLAYBACK_APIS),
        ("EXTRA_FIELDS",         set(EXTRA_FIELDS.keys())),
        ("DEREF_FIELDS",         set(DEREF_FIELDS.keys())),
        ("UNREPLAYABLE_PLAYBACK_APIS", set(UNREPLAYABLE_PLAYBACK_APIS.keys())),
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

    # An API's replay class is exactly one of these sets, and generate_playback_shim() tests them in this order, so an API in two gets the earlier handler and the later one is unreachable.
    # A merge that keeps an addition but drops the matching removal produces this, and nothing downstream notices until a replay-class test disagrees.
    replay_classes = [
        ("ERROR_STUB_PLAYBACK_APIS", ERROR_STUB_PLAYBACK_APIS),
        ("UNREPLAYABLE_PLAYBACK_APIS", set(UNREPLAYABLE_PLAYBACK_APIS)),
        ("NOOP_PLAYBACK_APIS", NOOP_PLAYBACK_APIS),
        ("CUSTOM_PLAYBACK_BODIES", set(CUSTOM_PLAYBACK_BODIES)),
        ("MANUAL_PLAYBACK_APIS", MANUAL_PLAYBACK_APIS),
        ("DIRECT_PLAYBACK_APIS", DIRECT_PLAYBACK_APIS),
    ]
    overlaps: List[str] = []
    for i, (first_name, first_set) in enumerate(replay_classes):
        for later_name, later_set in replay_classes[i + 1:]:
            for n in sorted(first_set & later_set):
                overlaps.append(f"'{n}' is in both {first_name} and "
                                f"{later_name}; it will be replayed as "
                                f"{first_name} and the {later_name} handler "
                                f"is dead code")
    if overlaps:
        print("\nERROR: an API may name only one replay classification set:")
        for problem in overlaps:
            print(f"  {problem}")
        sys.exit(1)

    # A Deref naming a parameter the API lacks does nothing, which is indistinguishable from the payload loss it exists to fix; likewise an array whose count parameter is missing.
    by_api = {e.name: e for e in entries}
    deref_problems: List[str] = []
    for api, specs in sorted(DEREF_FIELDS.items()):
        params = {p.name for p in by_api[api].params}
        for d in specs:
            if d.param not in params:
                deref_problems.append(
                    f"{api}: Deref names parameter '{d.param}', which is not in "
                    f"the signature ({', '.join(sorted(params))})")
            if d.direction not in ("in", "out", "inout"):
                deref_problems.append(
                    f"{api}.{d.param}: direction '{d.direction}' is not "
                    "in/out/inout")
            if d.playback not in ("auto", "manual"):
                deref_problems.append(
                    f"{api}.{d.param}: playback '{d.playback}' is not "
                    "auto/manual")
            if d.is_array:
                if d.count not in params:
                    deref_problems.append(
                        f"{api}.{d.param}: count parameter '{d.count}' is not "
                        "in the signature")
                if d.max_count <= 0:
                    deref_problems.append(
                        f"{api}.{d.param}: an array Deref needs max_count > 0")
    if deref_problems:
        print("\nERROR: DEREF_FIELDS does not match the parsed signatures:")
        for p in deref_problems:
            print(f"  {p}")
        sys.exit(1)

    # An unreplayable API whose capture shim is hand-written never emits the
    # capture-time warning, so the gap would only show up at replay.
    silent_unreplayable = sorted(
        set(UNREPLAYABLE_PLAYBACK_APIS) & MANUAL_CAPTURE_APIS)
    if silent_unreplayable:
        print("\nERROR: these UNREPLAYABLE_PLAYBACK_APIS have hand-written "
              "capture shims, so no capture-time warning is emitted. Call "
              "hrr_cap::writer::note_unreplayable() from the shim in "
              "hip_capture.cpp, or drop the API from MANUAL_CAPTURE_APIS:")
        for n in silent_unreplayable:
            print(f"  '{n}'")
        sys.exit(1)

    for a_name, a_set, b_name, b_set in [
        ("UNREPLAYABLE_PLAYBACK_APIS", set(UNREPLAYABLE_PLAYBACK_APIS),
         "ERROR_STUB_PLAYBACK_APIS", ERROR_STUB_PLAYBACK_APIS),
        ("UNREPLAYABLE_PLAYBACK_APIS", set(UNREPLAYABLE_PLAYBACK_APIS),
         "NOOP_PLAYBACK_APIS", NOOP_PLAYBACK_APIS),
        ("UNREPLAYABLE_PLAYBACK_APIS", set(UNREPLAYABLE_PLAYBACK_APIS),
         "MANUAL_PLAYBACK_APIS", MANUAL_PLAYBACK_APIS),
        ("ERROR_STUB_PLAYBACK_APIS", ERROR_STUB_PLAYBACK_APIS,
         "NOOP_PLAYBACK_APIS", NOOP_PLAYBACK_APIS),
    ]:
        both = sorted(a_set & b_set)
        if both:
            print(f"\nERROR: these APIs are in both {a_name} and {b_name}; "
                  "only the first would take effect:")
            for n in both:
                print(f"  '{n}'")
            sys.exit(1)
    log(f"  Manual capture (hand-written in hip_capture.cpp):  {n_manual_cap}")
    log(f"  Manual playback (hand-written in hip_playback.cpp): {n_manual_play}")
    log(f"  No-op playback (inline hipSuccess stubs):           {n_noop_play}")
    log(f"  Generated capture shims:  {len(entries) - n_manual_cap}")
    log(f"  Generated playback shims: {len(entries) - n_manual_play - n_noop_play}")

    if args.skip_public_api_check:
        print("  WARNING: --skip-public-api-check — playback casts are NOT "
              "validated against the public API")
    elif not public_path.exists():
        sys.exit(f"ERROR: public header not found: {public_path}\n"
                 f"       Playback casts cannot be validated against the API "
                 f"they are passed to.\n"
                 f"       Pass --public-header, or --skip-public-api-check to "
                 f"generate anyway (unsafe).")
    else:
        public_decls = parse_public_api_signatures(public_path)
        validate_playback_signatures(entries, public_decls, public_path, args.silent)

    capture_cpp = generate_capture_cpp(entries)
    if args.check_hrr_coverage:
        validate_capture_coverage(entries, manual_capture_path, args.silent)

    header_path.parent.mkdir(parents=True, exist_ok=True)
    header = generate_header(entries)
    header_path.write_text(header, encoding="utf-8", newline="\n")
    log(f"Written header   -> {header_path}")

    capture_path.parent.mkdir(parents=True, exist_ok=True)
    capture_path.write_text(capture_cpp, encoding="utf-8", newline="\n")
    log(f"Written capture  -> {capture_path}")

    playback_path.parent.mkdir(parents=True, exist_ok=True)
    playback_cpp = generate_playback_cpp(entries)
    playback_path.write_text(playback_cpp, encoding="utf-8", newline="\n")
    log(f"Written playback -> {playback_path}")

    if not args.silent:
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
