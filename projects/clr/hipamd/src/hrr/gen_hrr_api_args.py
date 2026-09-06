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
    # The stream-per-thread and cooperative spellings by host stub. Same
    # encoding as hipLaunchKernel, each under its own event id.
    "hipLaunchKernel_spt",
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernel_spt",
    # Extensible launches — the launch descriptor is a const struct pointer
    # whose attribute list is a second pointer; both are carried in the
    # variable-length launch payload rather than as capture-time addresses.
    "hipDrvLaunchKernelEx",
    "hipLaunchKernelExC",
    # Cooperative launch — void** kernelParams, same encoding as the others.
    "hipModuleLaunchCooperativeKernel",
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
    # hipMemcpy2D family — H2D blob snapshot (pitched host src) + D2H expected blob
    "hipMemcpy2D",
    "hipMemcpy2DAsync",
    # Array creation — need handle map (manual capture for output handle)
    "hipArrayCreate",
    "hipArray3DCreate",
    # Struct-pointer capture: value stored inline
    "hipStreamSetAttribute",
    "hipMemGetAllocationGranularity",
    "hipMemPoolSetAccess",
    "hipMemSetAccess",
    # hipDrv driver 3D/2D memcpy (HIP_MEMCPY3D / hip_Memcpy2D struct ptr; inline + blobs)
    "hipDrvMemcpy3D",
    "hipDrvMemcpy3DAsync",
    "hipDrvMemcpy2DUnaligned",
    "hipMemcpyParam2D",
    "hipMemcpyParam2DAsync",
    # Graph kernel nodes — hipKernelNodeParams names a host function and points
    # at a host argument array; both need the kernel-launch encoding as a tail.
    "hipGraphAddKernelNode",
    "hipGraphKernelNodeSetParams",
    "hipGraphExecKernelNodeSetParams",
    # Graph batch-memory-operation nodes — the op list is a second pointer hop
    # off hipBatchMemOpNodeParams, appended as an array tail.
    "hipGraphAddBatchMemOpNode",
    "hipGraphBatchMemOpNodeSetParams",
    "hipGraphExecBatchMemOpNodeSetParams",
    # Symbol copy nodes — the host buffer on the far side of the symbol is a
    # blob, and whether there is one at all depends on the copy kind.
    "hipGraphAddMemcpyNodeToSymbol",
    "hipGraphAddMemcpyNodeFromSymbol",
    "hipGraphMemcpyNodeSetParamsToSymbol",
    "hipGraphExecMemcpyNodeSetParamsToSymbol",
    # JIT linker input — the image is a blob.
    "hipLinkAddData",
}

# Alias for backward compat within the file (some helpers used MANUAL_APIS)
MANUAL_APIS = MANUAL_CAPTURE_APIS

# Kernel launches recorded as variable-length binary (hip_capture.cpp
# serialize_kernel_launch), NOT as sizeof(hrr_args_hipModuleLaunchKernel).
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

# Minimum bytes for a valid variable-length kernel launch payload:
#   header + stream(8) + name_len(2) + co_hash(16) + grid/block/shared(28) + counts(4)
# (kernel name and per-arg data may add more; spin with 0 args is 94 bytes total).
_VARIABLE_KERNEL_LAUNCH_MIN_PAYLOAD_EXPR = (
    "static_cast<uint32_t>(sizeof(hrr_event_header) + 8u + 2u + 16u + 12u + 12u + 4u + 2u + 2u)"
)

# APIs that are pass-through even for the manual path
# (hipModuleGetFunction: function handles identified by name at launch time)
PASSTHROUGH_ONLY: Set[str] = {
    "hipModuleGetFunction",
}

# ---------------------------------------------------------------------------
# Windows exports
#
# amdhip.def.in is the export list the Windows amdhip64 DLL is linked with, so
# an API missing from it has no import library entry and hrr-playback.exe fails
# to link against it — even though the HIP headers declare it. Read the list
# rather than hard-coding the gaps, so the guard disappears on its own once clr
# starts exporting the symbol.
# ---------------------------------------------------------------------------

_AMDHIP_DEF_IN = Path(__file__).resolve().parent.parent / "amdhip.def.in"


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


# ---------------------------------------------------------------------------
# Playback manual APIs — implemented by hand in hip_playback.cpp
# ---------------------------------------------------------------------------

MANUAL_PLAYBACK_APIS: Set[str] = {
    # Kernel launches — variable-length binary payload, kernarg buffer + function lookup
    "hipModuleLaunchKernel",
    "hipExtModuleLaunchKernel",
    "hipLaunchKernel",
    "hipLaunchByPtr",
    "hipLaunchKernel_spt",
    # Cooperative launches by host stub — the cooperative entry point, for the
    # same reason hipModuleLaunchCooperativeKernel needs it.
    "hipLaunchCooperativeKernel",
    "hipLaunchCooperativeKernel_spt",
    # Extensible launches — same payload plus the recorded launch-attribute
    # list, replayed through hipDrvLaunchKernelEx with a rebuilt descriptor.
    "hipDrvLaunchKernelEx",
    "hipLaunchKernelExC",
    # Cooperative launch — must go back through the cooperative entry point, or
    # a grid-wide barrier hangs instead of running.
    "hipModuleLaunchCooperativeKernel",
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
    # Query APIs — capture only logs hipSuccess (event/stream already complete).
    # Replay may reach the same API sooner relative to GPU work; spin on
    # hipErrorNotReady until hipSuccess.  See DESIGN.md § HIP-Specific Polling.
    "hipEventQuery",
    "hipStreamQuery",
    "hipStreamQuery_spt",
    # Fat binary registration — load blob as module so kernel names resolve
    "__hipRegisterFatBinary",
    # Variable registration — resolve the recorded symbol name in the loaded
    # modules and map both the host shadow and the capture-time device address
    # onto it, which is what makes the symbol copy family replayable.
    "__hipRegisterVar",
    # Symbol queries — answered from that map rather than from a host shadow
    # address this process does not have.
    "hipGetSymbolAddress",
    "hipGetSymbolSize",
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
    # hipGraphInstantiateWithFlags — same graph_map guard as hipGraphInstantiate;
    # must fail loudly (not pass a null graph to the real API) when the graph was
    # built via the unsupported explicit node API.
    "hipGraphInstantiateWithFlags",
    "hipGraphLaunch",
    # DtoH driver-style copies — dst is a host pointer (not in alloc_map); need temp buffer
    "hipMemcpyDtoH",
    "hipMemcpyDtoHAsync",
    # hipMemcpy3D family — parms struct stored inline (parms_bytes); must reconstruct
    "hipMemcpy3D",
    "hipMemcpy3DAsync",
    "hipMemcpy3D_spt",
    "hipMemcpy3DAsync_spt",
    # hipMemcpy2D family — H2D substitutes the captured blob (host src is a stale VA);
    # D2H validates the device result against the captured expected blob.
    "hipMemcpy2D",
    "hipMemcpy2DAsync",
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
    # hipStreamBatchMemOp — the recorded op list needs a per-op device-pointer
    # rewrite, which depends on the op type stored in each entry.
    "hipStreamBatchMemOp",
    # IPC handles — the live handle must be paired with the recorded one so a
    # later import in the same archive resolves.
    "hipIpcGetMemHandle",
    "hipIpcOpenMemHandle",
    # VMM (Virtual Memory Management) — output handles / VAs must be tracked
    "hipMemAddressReserve",
    "hipMemAddressFree",
    "hipMemCreate",
    "hipMemRelease",
    "hipMemMap",
    "hipMemUnmap",
    # Device allocation — must allocate a real buffer and record it in alloc_map
    # (with padding + zero-init parity with hipMalloc).
    "hipExtMallocWithFlags",
    # hipDrv driver 3D/2D memcpy (reconstruct struct, translate device ptrs, blob/validate)
    "hipDrvMemcpy3D",
    "hipDrvMemcpy3DAsync",
    "hipDrvMemcpy2DUnaligned",
    "hipMemcpyParam2D",
    "hipMemcpyParam2DAsync",
    # Batch 3D copy — each operand's address lives in a different union member
    # depending on the tag beside it.
    "hipMemcpy3DBatchAsync",
    # Graph kernel nodes — resolve the kernel by name, rebuild the argument
    # array from the recorded bytes, register the node handle.
    "hipGraphAddKernelNode",
    "hipGraphKernelNodeSetParams",
    "hipGraphExecKernelNodeSetParams",
    # Graph batch-memory-operation nodes — per-op device address rewrite, same
    # as hipStreamBatchMemOp.
    "hipGraphAddBatchMemOpNode",
    "hipGraphBatchMemOpNodeSetParams",
    "hipGraphExecBatchMemOpNodeSetParams",
    # Symbol copy nodes — the symbol resolves through the symbol registry and
    # the host side of the copy comes from a blob.
    "hipGraphAddMemcpyNodeToSymbol",
    "hipGraphAddMemcpyNodeFromSymbol",
    "hipGraphMemcpyNodeSetParamsToSymbol",
    "hipGraphMemcpyNodeSetParamsFromSymbol",
    "hipGraphExecMemcpyNodeSetParamsToSymbol",
    "hipGraphExecMemcpyNodeSetParamsFromSymbol",
    # JIT linker input — the image comes back from a blob, and the linker state
    # from the map hipLinkCreate now fills.
    "hipLinkAddData",
    # Graph memory-allocation nodes — the address the node hands out is chosen
    # by the replay's pool, so it must land in alloc_map under the recorded one.
    "hipGraphAddMemAllocNode",
    # Driver-API memcpy nodes — HIP_MEMCPY3D distinguishes host from device
    # memory by a member, and a host operand was never recorded.
    "hipDrvGraphAddMemcpyNode",
    "hipDrvGraphMemcpyNodeSetParams",
    "hipDrvGraphExecMemcpyNodeSetParams",
}

# ---------------------------------------------------------------------------
# Graph work whose arguments the archive does not carry well enough to rebuild.
# Most of the node API now replays for real; what is left here needs something
# the recording does not have — a semaphore only another process can produce, a
# device symbol the archive has no name for, a union whose active member is not
# recoverable. Unlike the generic NOOP handlers (which emit a vague
# once-per-process message), these emit a loud, attributable warning naming the
# specific API (finding H1), so when replay later fails the cause is traceable.
#
# These are intentionally NON-FATAL (they return hipSuccess) but they poison the
# graph: ctx.mark_graph_incomplete records that the graph is now short a node,
# and hipGraphInstantiate / hipGraphInstantiateWithFlags refuse an incomplete
# graph with hipErrorNotSupported. That is the point at which a missing node
# would otherwise turn into a graph launch that silently does less work than the
# recording did, and it leaves programs that merely build graphs they never
# instantiate free to run.
#
# Only calls that build or mutate graph contents belong here. Harmless queries
# (hipGraphGetNodes/Edges, hipStreamGetCaptureInfo_v2, ...) stay in
# NOOP_PLAYBACK_APIS. The stream-capture chain never emits these APIs, so the
# hipStreamBeginCapture/EndCapture path is unaffected.
# ---------------------------------------------------------------------------
ERROR_STUB_PLAYBACK_APIS: Set[str] = {
    # An external semaphore is imported from a handle only its owning process
    # can produce, so a node that signals or waits on one has nothing to point
    # at here.
    "hipGraphAddExternalSemaphoresSignalNode",
    "hipGraphAddExternalSemaphoresWaitNode",
    # hipHostNodeParams is a host function pointer plus its userData, both
    # belonging to the capturing process.
    "hipGraphAddHostNode",
    # hipGraphNodeParams is a union over every node kind, so reconstructing it
    # means reconstructing all of them through one 256-byte blob whose active
    # member is only known from a type tag. The typed spellings above are
    # replayed; this one is not.
    "hipGraphAddNode",
    # hipGraphInstantiateParams carries an out node pointer and an error-log
    # buffer belonging to the capturing process.
    "hipGraphInstantiateWithParams",
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
    # NOTE: the explicit node-construction Add*Node APIs were moved to
    # ERROR_STUB_PLAYBACK_APIS (they now fail loudly, see H1). Only the harmless
    # query/update calls remain NOOP here.
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
    # Category 5: Other type mismatches (output handle params stored as void** by generator)
    # The hipCtx* family used to live here; it now runs against ctx_map, which
    # is what makes hipCtxPushCurrent push a context that exists in this
    # process. hipCtxGetDevice stays: it writes a hipDevice_t the replay has no
    # use for, and the device it reports is the one hipSetDevice already chose.
    "hipCtxGetDevice",
    "hipDeviceGet",
    "hipDrvGetErrorName",
    "hipDrvGetErrorString",
    "hipGetTextureReference",
    "hipKernelGetName",
    "hipMemImportFromShareableHandle",
    # The export side now writes its handle into a local out-buffer instead of
    # the stale capture-time address, so replay really does mint an fd — and
    # the recorded number the import was handed names something else entirely
    # in this process, which faults inside the pool import.
    "hipMemPoolImportFromShareableHandle",
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
    # hipDeviceGetGraphMemAttribute / hipDeviceSetGraphMemAttribute — void* value ptr (stale) — noop
    "hipDeviceGetGraphMemAttribute",
    "hipDeviceSetGraphMemAttribute",
    # hipDevicePrimaryCtxGetState — output pointers (stale) — noop
    "hipDevicePrimaryCtxGetState",
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
    # hipMallocPitch — already in MANUAL_PLAYBACK_APIS for the DrvMemcpy test; these are the _spt wrappers
    # NOTE: hipExtMallocWithFlags is intentionally NOT noop'd. It is a real device
    # allocation (see _ALLOC_CREATE_APIS) and is handled by a manual playback handler
    # (MANUAL_PLAYBACK_APIS) so the returned device pointer lands in alloc_map and any
    # H2D/D2H/kernel-arg use of it translates correctly.
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
    # hipStreamWaitValue32/64 stay no-op because a wait can hang on replay: the
    # condition is satisfied by whatever wrote the value at capture time, and if
    # that writer is not itself replayed the wait never completes and blocks the
    # replay. The void* ptr is not the problem here: it is named `ptr`, so the
    # generated shim would translate it via ctx.translate_ptr like any other.
    # hipStreamWriteValue32/64 are replayed instead: the void* ptr is translated
    # via alloc_map (ctx.translate_ptr) and the stream via ctx.translate_stream.
    # See SKIP_IF_UNMAPPED_PLAYBACK_APIS for the untranslatable-destination case.
    "hipStreamWaitValue32",
    "hipStreamWaitValue64",
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
    # hipGetProcAddress and hipGetProcAddress_spt used to be here and to crash
    # respectively; both now carry the symbol name and look it up for real.
    "hipGetDriverEntryPoint",
    "hipGetDriverEntryPoint_spt",
    # hipOccupancyAvailableDynamicSMemPerBlock — hipFunction_t stale handle
    "hipOccupancyAvailableDynamicSMemPerBlock",
    # hipSetValidDevices — device array ptr stale at playback
    "hipSetValidDevices",
    # hipMemPtrGetInfo — void* output size ptr stale
    "hipMemPtrGetInfo",
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
    # Category 15: Graph explicit APIs.
    # The node-construction and node-mutation calls are replayed now that a
    # recorded hipGraphNode_t resolves through graph_node_map and the node
    # parameter structs reach the archive (DEREF_FIELDS). What is left here is
    # the query half — a Get* that writes into a caller buffer changes nothing
    # about the replayed graph — plus the handful whose arguments still name
    # something only the capturing process had.
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
    "hipGraphHostNodeSetParams",
    "hipGraphExecHostNodeSetParams",
    "hipGraphExecDestroy",
    "hipGraphExecGetFlags",
    # The generic hipGraphNodeParams spelling — see hipGraphAddNode.
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
}

# ---------------------------------------------------------------------------
# Playback APIs whose recorded destination must resolve to a live allocation
# before the real API is called: API name -> destination parameter name.
# ---------------------------------------------------------------------------
#
# ctx.translate_ptr() returns nullptr when the recorded address is in neither
# the alloc map nor the VMM reserved-VA map, for instance a write into memory
# HRR does not track such as a framework sub-allocation carved out of a larger
# hipMalloc. The stream-operation APIs reject a null ptr with
# hipErrorInvalidValue (ihipStreamOperation checks it first), and
# dispatch_event() treats any non-success handler return as fatal, so a single
# untranslatable destination would abort the entire replay. Warn once and skip
# the call instead, which is the unmapped-pointer contract the hand-written
# playback_hipFree / playback_hipFreeAsync / playback_hipHostFree already use.
# Slightly-wrong data beats no replay at all.
SKIP_IF_UNMAPPED_PLAYBACK_APIS: Dict[str, str] = {
    "hipStreamWriteValue32": "ptr",
    "hipStreamWriteValue64": "ptr",
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
    # The graph symbol copies. The to-symbol spellings read a host buffer the
    # recording owned, so the bytes go in a blob; the from-symbol spellings
    # write one, and the blob is the expected output.
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
    # hipLinkAddData — the code object being linked. Sizes run to hundreds of
    # kilobytes, so it is a blob rather than inline bytes.
    "hipLinkAddData":     [("uint64_t", "blob_hash_lo", "linker input image blob hash lo"),
                           ("uint64_t", "blob_hash_hi", "linker input image blob hash hi")],
    # __hipRegisterVar — the symbol's device address in the capturing process.
    # Zero from the live shim, which runs at static-init time before any device
    # exists; filled in by the post-registration sweep, which is where every
    # registration this suite sees is actually recorded.
    "__hipRegisterVar":   [("uint64_t", "dev_addr",
                            "capture-time device address of the symbol, 0 if unresolved")],
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
    # hipMemcpy2D family — H2D blob (pitched host src) + D2H expected-output blob.
    # The copied region is height rows of `width` bytes spaced by spitch/dpitch;
    # the blob is the contiguous host buffer (pitch * (height-1) + width bytes).
    "hipMemcpy2D":      [("uint64_t", "blob_hash_lo", "H2D blob hash lo (0 if not H2D)"),
                         ("uint64_t", "blob_hash_hi", "H2D blob hash hi"),
                         ("uint64_t", "d2h_hash_lo",  "D2H expected-output blob hash lo (0 if not D2H)"),
                         ("uint64_t", "d2h_hash_hi",  "D2H expected-output blob hash hi")],
    "hipMemcpy2DAsync": [("uint64_t", "blob_hash_lo", "H2D blob hash lo (0 if not H2D)"),
                         ("uint64_t", "blob_hash_hi", "H2D blob hash hi"),
                         ("uint64_t", "d2h_hash_lo",  "D2H expected-output blob hash lo (0 if not D2H)"),
                         ("uint64_t", "d2h_hash_hi",  "D2H expected-output blob hash hi")],
    # hipArrayCreate — HIP_ARRAY_DESCRIPTOR is 24 bytes; store inline.
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
    # Same descriptor, same treatment: hipMemcpyParam2D and its async spelling
    # take the identical hip_Memcpy2D that hipDrvMemcpy2DUnaligned does.
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

# ---------------------------------------------------------------------------
# Dereferenced pointer arguments
#
# normalise_field_type() lowers every pointer to a uint64_t holding the
# capture-time address, so by default the pointee never reaches the archive.
# For an input struct that is silent data loss (section 8.3): replay is handed
# an address from a process that no longer exists. A Deref entry tells the
# generator to carry the pointee itself.
#
# Capture: the generated shim memcpy's `size` bytes (or `count` elements) out of
# the pointer into the event, and sets <param>_present.
# Playback: the generated handler rebuilds a local copy from those bytes and
# passes its address, instead of casting the recorded address back to a pointer.
#
# Use direction="in" for an argument the API reads, "out" for one it writes
# (recorded for fidelity; replay still passes a local, since the recorded
# address means nothing here), and "inout" for both.
#
# An API whose pointee contains device pointers, handles or nested pointers
# needs those translated at replay, which no table entry can express: give it a
# Deref for the capture side and a hand-written handler in MANUAL_PLAYBACK_APIS
# for the replay side. `playback="manual"` says so, and suppresses the generic
# reconstruction.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Deref:
    param: str            # parameter name in the hip_api_trace.hpp typedef
    ctype: str            # pointee type, e.g. "hipMemAllocationProp"
    size: int             # inline bytes per element; static_assert'd against sizeof(ctype)
    direction: str = "in"  # "in" | "out" | "inout"
    count: str = ""       # element-count parameter name (array arguments only)
    max_count: int = 0    # inline capacity in elements, for array/string arguments
    # A NUL-terminated string has no count parameter and no fixed pointee size:
    # capture copies up to max_count - 1 characters plus the terminator, and
    # replay passes the recorded bytes straight out of the event.
    string: bool = False
    playback: str = "auto"  # "auto" (generic reconstruction) | "manual"
    # Members of the pointee that hold recorded device addresses. Replay runs
    # each through ctx.translate_ptr before the call; without that the struct
    # would arrive intact but pointing into the capturing process.
    ptr_members: Tuple[str, ...] = ()
    # Members of the pointee that hold recorded HIP handles, as
    # (member, handle type). Same idea as ptr_members, through the handle map
    # for that type instead of alloc_map.
    handle_members: Tuple[Tuple[str, str], ...] = ()
    # For an array argument whose elements are themselves HIP handles (a graph
    # dependency list is an array of hipGraphNode_t): the handle type each
    # element is translated through.
    elem_handle: str = ""
    # For an array of bare device addresses (a batch copy's dsts[] / srcs[]):
    # every element goes through translate_ptr, the way a ptr_member of a
    # struct does.
    elem_ptr: bool = False

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


# API name -> the pointer arguments whose pointee is carried in the event.
DEREF_FIELDS: Dict[str, List[Deref]] = {
    # The allocation property decides the memory type and, for a heap that maps
    # cross-node peers, which device the pages come from. Playback used to
    # hardcode Pinned/device 0, which builds the wrong topology silently.
    "hipMemCreate": [
        Deref("prop", "hipMemAllocationProp", 32, playback="manual"),
    ],
    # The op list is the whole call. Each entry's address is a device pointer,
    # and which member holds it depends on the op type, so the reconstruction
    # is hand-written rather than a ptr_members rewrite.
    "hipStreamBatchMemOp": [
        Deref("paramArray", "hipStreamBatchMemOpParams", 48, count="count",
              max_count=16, playback="manual"),
    ],
    # The IPC handle is 64 bytes. Recorded as one uint64_t it was 56 bytes of
    # nothing, and replay passed a null pointer for the out buffer, which is
    # why the call came back as invalid argument. Carrying the whole handle
    # also gives the archive something an importer in another rank can be
    # matched against (ipc_handle_map).
    "hipIpcGetMemHandle": [
        Deref("handle", "hipIpcMemHandle_t", 64, direction="out",
              playback="manual"),
    ],
    # Which device's view of the mapping is being asked about, and what the
    # answer was. The location was a stale struct pointer, which is why this
    # was a NOOP; carrying its eight bytes makes the query answerable, and the
    # recorded flags make the recording self-describing.
    "hipMemGetAccess": [
        Deref("location", "hipMemLocation", 8),
        Deref("flags", "unsigned long long", 8, direction="out"),
    ],
    # The pre-chevron launch ABI pushes `size` bytes from a host address into
    # the pending kernarg buffer. Recorded as a bare address, replay handed the
    # runtime a pointer into the capturing process and the read segfaulted.
    # 256 bytes covers any single kernel argument, struct arguments included.
    "hipSetupArgument": [
        Deref("arg", "unsigned char", 1, count="size", max_count=256),
    ],
    # The symbol name is the entire question these two ask, and it was recorded
    # as an address with none of the characters behind it: measured as a SIGSEGV
    # inside hip::hipGetProcAddress_common (hip_device.cpp:880).
    "hipGetProcAddress": [
        Deref("symbol", "char", 1, max_count=256, string=True),
    ],
    "hipGetProcAddress_spt": [
        Deref("symbol", "char", 1, max_count=256, string=True),
    ],
    # deviceVar is the symbol's name in the code object, and it is the only
    # part of a registration that means anything in another process: `var` is
    # a host shadow address and `modules` a fat-binary handle. Recording the
    # name is what lets replay resolve the same global through
    # hipModuleGetGlobal and map the recorded device address onto it.
    "__hipRegisterVar": [
        Deref("deviceVar", "char", 1, max_count=256, string=True),
    ],
    # The JIT linker's option arrays. Both were recorded as addresses and then
    # replayed as if they were out parameters, so a link was rebuilt with
    # whatever the uninitialised locals held. Thirty-two options is more than
    # any caller passes.
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
    # The device properties being matched against are the whole question these
    # two ask. 1472 bytes is a large event, but a program calls hipChooseDevice
    # once at startup, not in a loop.
    "hipChooseDevice": [
        Deref("prop", "hipDeviceProp_t", 1472),
    ],
    "hipChooseDeviceR0000": [
        Deref("properties", "hipDeviceProp_tR0000", 792),
    ],
    # The format decides the answer: the maximum width depends on the element
    # size the caller asked about.
    "hipDeviceGetTexture1DLinearMaxWidth": [
        Deref("fmtDesc", "hipChannelFormatDesc", 20),
        Deref("maxWidthInElements", "size_t", 8, direction="out"),
    ],
    # The attribute value is the whole call, and the union is 64 bytes of it.
    "hipGraphKernelNodeSetAttribute": [
        Deref("value", "hipKernelNodeAttrValue", 64),
    ],
    # Which device's pool is being redirected. Recorded as a stale struct
    # pointer this returned hipErrorInvalidValue at replay; the pool handle
    # beside it was already translated through mempool_map.
    "hipMemSetMemPool": [
        Deref("location", "hipMemLocation", 8),
    ],
    "hipMemGetMemPool": [
        Deref("location", "hipMemLocation", 8),
    ],
    # The batch copies are five parallel arrays and an op list. Recorded as
    # addresses they were five stale pointers, which is why the handler
    # returned an error rather than copying anything. Sixteen entries inline
    # covers the batches this suite issues and warns at capture beyond that.
    "hipMemcpyBatchAsync": [
        Deref("dsts", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("srcs", "void*", 8, count="count", max_count=16, elem_ptr=True),
        Deref("sizes", "size_t", 8, count="count", max_count=16),
        Deref("attrs", "hipMemcpyAttributes", 24, count="numAttrs",
              max_count=16),
        Deref("attrsIdxs", "size_t", 8, count="numAttrs", max_count=16),
        Deref("failIdx", "size_t", 8, direction="out"),
    ],
    # Each operand is a tagged union of a pointer and an array handle, so
    # which member holds the recorded address depends on the tag beside it —
    # a rewrite no table entry can express.
    "hipMemcpy3DBatchAsync": [
        Deref("opList", "hipMemcpy3DBatchOp", 112, count="numOps",
              max_count=16, playback="manual"),
        Deref("failIdx", "size_t", 8, direction="out", playback="manual"),
    ],
}

# ---------------------------------------------------------------------------
# Graph node construction and mutation
#
# Every hipGraphAdd*Node call takes the same two things — a dependency array of
# hipGraphNode_t and a node-parameter struct — and until now neither reached
# the archive: the array was recorded as one stale address and the struct not
# at all. That is why explicit graph construction was a blanket exclusion.
#
# Both are ordinary Deref shapes. The dependency array is an array of handles
# (elem_handle), the parameter structs are single pointees whose device
# addresses and array handles are rewritten at replay (ptr_members /
# handle_members). The node handle each call produces is registered in
# graph_node_map by _HANDLE_CREATE_APIS, which is what makes a later
# dependency reference or *NodeSetParams call resolvable.
#
# A recorded graph rarely has many roots feeding one node; 16 inline
# dependencies covers real graphs and the overflow warns at capture.
# ---------------------------------------------------------------------------
_MAX_DEPS = 16

def _deps(param: str, count: str = "numDependencies") -> Deref:
    return Deref(param, "hipGraphNode_t", 8, count=count, max_count=_MAX_DEPS,
                 elem_handle="hipGraphNode_t")


# hipMemsetParams::dst and hipMemcpy3DParms' pitched pointers and array handles
# are the recorded-address members inside each node-parameter struct.
_MEMSET_PARAMS = dict(ctype="hipMemsetParams", size=48, ptr_members=("dst",))
_MEMCPY3D_PARAMS = dict(
    ctype="hipMemcpy3DParms", size=160,
    ptr_members=("srcPtr.ptr", "dstPtr.ptr"),
    handle_members=(("srcArray", "hipArray_t"), ("dstArray", "hipArray_t")),
)

DEREF_FIELDS.update({
    # --- construction: dependency array only ---
    "hipGraphAddEmptyNode":       [_deps("pDependencies")],
    "hipGraphAddEventRecordNode": [_deps("pDependencies")],
    "hipGraphAddEventWaitNode":   [_deps("pDependencies")],
    "hipGraphAddChildGraphNode":  [_deps("pDependencies")],
    "hipGraphAddMemcpyNode1D":    [_deps("pDependencies")],
    # The symbol copies. Their src/dst is a host buffer on the H2D and D2H
    # spellings, so it is carried as a blob by the manual shim rather than
    # dereferenced here; only the dependency array is declarative.
    "hipGraphAddMemcpyNodeToSymbol":   [_deps("pDependencies")],
    "hipGraphAddMemcpyNodeFromSymbol": [_deps("pDependencies")],
    "hipGraphAddMemFreeNode":     [_deps("pDependencies")],
    "hipDrvGraphAddMemFreeNode":  [_deps("dependencies")],
    # from[] and to[] are parallel arrays under one count.
    "hipGraphAddDependencies":    [_deps("from"), _deps("to")],
    # --- construction: dependency array plus a node-parameter struct ---
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
    # A kernel node names its kernel by host function address and carries a
    # void** argument array, which is the kernel-launch encoder's problem, not
    # a memcpy's.
    "hipGraphAddKernelNode": [
        _deps("pDependencies"),
        Deref("pNodeParams", "hipKernelNodeParams", 64, playback="manual"),
    ],
    # dptr is written by the call and must land in alloc_map, and poolProps
    # names a device — hand-written on the replay side.
    "hipGraphAddMemAllocNode": [
        _deps("pDependencies"),
        Deref("pNodeParams", "hipMemAllocNodeParams", 120, direction="inout",
              playback="manual"),
    ],
    # HIP_MEMCPY3D distinguishes host from device source by a member, so the
    # host case needs a blob rather than a pointer rewrite.
    "hipDrvGraphAddMemcpyNode": [
        _deps("dependencies"),
        Deref("copyParams", "HIP_MEMCPY3D", 184, playback="manual"),
    ],
    # nodeParams.paramArray is a second pointer hop.
    "hipGraphAddBatchMemOpNode": [
        _deps("dependencies"),
        Deref("nodeParams", "hipBatchMemOpNodeParams", 32, playback="manual"),
    ],
    # --- mutation: node parameters, no dependency array ---
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
    # Beginning a capture into an explicitly named graph. As a NOOP the begin
    # never happened at replay, so the stream was not capturing when
    # hipStreamEndCapture arrived and that call returned 401 — a silent no-op
    # billing a real handler two events later.
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


# ---------------------------------------------------------------------------
# Structs passed BY VALUE that do not fit in a uint64_t
#
# normalise_field_type() lowers an unrecognised by-value type to a single
# uint64_t and _fill_param() then records a zero for it, so a 24-byte hipExtent
# or a 64-byte hipIpcMemHandle_t reaches the archive as nothing at all. These
# are recorded as inline bytes instead: `uint8_t <param>_bytes[N]`, memcpy'd at
# capture and memcpy'd back into a local at replay.
#
# ptr_members names the members holding recorded device addresses; replay
# translates each through ctx.translate_ptr, because a faithfully restored
# struct still points into the capturing process otherwise.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ByValueStruct:
    size: int
    ptr_members: Tuple[str, ...] = ()


BY_VALUE_STRUCTS: Dict[str, ByValueStruct] = {
    "hipExtent":           ByValueStruct(24),
    "hipPos":              ByValueStruct(24),
    "hipPitchedPtr":       ByValueStruct(32, ptr_members=("ptr",)),
    "hipMemLocation":      ByValueStruct(8),
    "hipChannelFormatDesc": ByValueStruct(20),
    "hipIpcMemHandle_t":   ByValueStruct(64),
    "hipIpcEventHandle_t": ByValueStruct(64),
}


def by_value_struct(raw_type: str) -> Optional[ByValueStruct]:
    """The by-value inline-bytes spec for a parameter type, if it has one."""
    t = re.sub(r"\b(const|volatile|restrict|struct)\b", " ", raw_type)
    if "*" in t:
        return None
    parts = t.split()
    return BY_VALUE_STRUCTS.get(parts[0]) if parts else None


# Maps the base name of a uint8_t inline-array field (e.g. "pool_props_bytes")
# to the C struct type whose sizeof must match the declared array extent.
# Used to emit static_assert(sizeof(CType) == N) immediately after each struct
# that contains such a field, catching size mismatches at compile time.
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
    "hipKernel_t", "hipLibrary_t", "hipLinkState_t",
    # HIP 7.14+ green context / device resource handles
    "hipExecutionCtx_t", "hipDevResourceDesc_t",
}

# Types that cannot be cast to uint64_t and are recorded as zero.
# By-value structs are NOT in here: they are carried as inline bytes instead
# (BY_VALUE_STRUCTS). What is left is genuinely unrecordable — a function
# pointer or an opaque handle whose value means nothing outside the capturing
# process.
_NON_CASTABLE_TYPES = {
    "hipGraphicsResource_t",  # opaque GL interop resource
    "hipStreamCallback_t",    # function pointer
    "hipHostFn_t",            # function pointer
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
    # Enums the generator used to lower to `uint64_t /* T */`, which reads as a
    # dropped by-value struct to anything inspecting the generated types (the
    # matrix's payload-loss detector counted all eight). Every one is a 4-byte
    # enum whose value survives intact; naming them here says so.
    "hipArray_Format", "hipDevResourceType", "hipFunction_attribute",
    "hipGraphMemAttributeType", "hipJitInputType",
    "hipMemAllocationHandleType", "hipMemRangeHandleType",
    "hipPointer_attribute", "hipStreamAttrID", "hipGLDeviceList",
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
    # OpenGL interop spellings of unsigned int (the GL headers are not included
    # here, so these do not resolve through the scalar keywords above).
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
    # strip const/volatile
    t_nc = re.sub(r'\b(const|volatile|restrict|struct)\b', '', t).strip()
    t_nc = re.sub(r'\s+', ' ', t_nc)

    # Any pointer -> uint64_t
    if '*' in t_nc:
        return "uint64_t"

    # dim3 / uint3 -> expand inline
    base = t_nc.split()[0]
    if base in ("dim3", "uint3"):
        return "__DIM3__"

    # By-value struct too large for a uint64_t -> inline bytes
    if base in BY_VALUE_STRUCTS:
        return "__BYVAL__"

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
    # Retired or unparsable dispatch-table slot. Occupies an hrr_api_id_t so
    # later members keep their IDs; no capture shim is installed.
    reserved: bool = False


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
    # t_hipFoo hipFoo_fn;  or  void* hipFoo_fn;  (possibly split across lines)
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
                                table=table, reserved=True))

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

    # Runtime first. Compiler-first put every runtime ID at a
    # compiler-table-size offset, so one new HipCompilerDispatchTable member
    # renumbered all 500+ runtime IDs — the opposite of the append-only scheme
    # this function exists to implement. A new HipDispatchTable member still
    # shifts the compiler-ID tail (playback no-ops); that is cheaper than
    # shifting the IDs written into every captured event.
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

    # A t_hipXxx typedef with no member in HipDispatchTable has no append-only
    # position to take an ID from. None exist today; if one appears, it is still
    # captured rather than dropped, and it goes last so the table-ordered IDs
    # ahead of it keep their values.
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
 * address and nothing else, which is the payload-loss class of section 8.3.
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
 * order, which renumbered 496 of the 552 IDs once. Runtime IDs occupy 0..N-1
 * so a new compiler-table member cannot shift them. Every event stores its
 * ID, so a pre-v5 archive names the wrong API when decoded against this
 * table and needs an ID translation to be read back. From v5 on a new API
 * takes the next free ID in its table and no existing runtime ID moves, so
 * adding APIs no longer needs a version bump. A retired dispatch-table slot
 * (nulled void*) still occupies an ID.
 * v6: pointer arguments whose pointee used to be dropped now carry it inline
 * (DEREF_FIELDS). Event payloads grew for ~50 APIs, so an archive written
 * before v6 cannot be read by a v6 reader: re-capture rather than replay an
 * old recording. */
#define HRR_VERSION ((uint16_t)6u)

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

    # Comment showing original signature
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

    # Return value
    if not _is_void_return(entry.ret_type):
        if '*' in entry.ret_type:
            lines.append("    uint64_t ret;")
        else:
            ft = normalise_field_type(entry.ret_type)
            if ft == "__BYVAL__":
                # A by-value struct return (hipCreateChannelDesc). Never
                # recorded — such an API is a capture pass-through — but the
                # field has to have a size the archive can round-trip.
                bv = by_value_struct(entry.ret_type)
                lines.append(f"    uint8_t ret_bytes[{bv.size}];")
                ft = None
            elif ft == "__DIM3__":
                ft = "uint64_t"
            if ft is not None:
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
            elif ft == "__BYVAL__":
                bv = by_value_struct(param.raw_type)
                base = _get_base_type(param.raw_type)
                lines.append(f"    uint8_t {safe}_bytes[{bv.size}];"
                             f"  /* {base} passed by value, inline copy */")
            else:
                lines.append(f"    {ft} {safe};")

    # Dereferenced pointer arguments — the pointee, carried inline
    for d in deref_specs(entry.name):
        what = (f"{d.ctype}[{d.max_count}] inline copy" if d.is_array
                else f"{d.ctype} inline copy")
        lines.append(f"    uint8_t {d.bytes_field}[{d.total_bytes}];  /* {what} */")
        lines.append(f"    uint8_t {d.present_field};  /* 1 when {d.param} was non-null */")
        if d.is_array:
            lines.append(f"    uint32_t {d.count_field};  /* elements copied from {d.param} */")

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
    # A Deref or by-value field that reserves fewer bytes than the value needs
    # would truncate silently at capture, which is the failure the whole
    # mechanism exists to remove. Catch it in the compiler instead.
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
    # Single pointer to any handle type → output
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
    elif ft == "__BYVAL__":
        lines.append(f"    std::memcpy(a.{name}_bytes, &{name}, sizeof({name}));")
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


def _fill_derefs(lines: List[str], entry: ApiEntry) -> None:
    """Emit the memcpy that carries each dereferenced pointee into the event.

    Runs after the real call, alongside the other fills: the arguments a HIP
    API reads are still valid there, and an output argument has been written
    by then.
    """
    for d in deref_specs(entry.name):
        if d.string:
            lines.append(f"    if ({d.param}) {{")
            lines.append(f"      size_t _n = std::strlen({d.param});")
            lines.append(f"      if (_n > {d.max_count - 1}u) {{")
            lines.append(f"        static bool warned_{d.param} = false;")
            lines.append(f"        if (!warned_{d.param}) {{")
            lines.append(f"          warned_{d.param} = true;")
            lines.append(f"          fprintf(stderr,")
            lines.append(f"                  \"[HRR] {entry.name}: {d.param} is %zu characters; \"")
            lines.append(f"                  \"recording the first {d.max_count - 1} only.\\n\", _n);")
            lines.append(f"        }}")
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
            lines.append(f"          fprintf(stderr,")
            lines.append(f"                  \"[HRR] {entry.name}: recording only the first {d.max_count} \"")
            lines.append(f"                  \"of %u {d.param} entries; replay of this call will be \"")
            lines.append(f"                  \"incomplete.\\n\", _n);")
            lines.append(f"        }}")
            lines.append(f"        _n = {d.max_count}u;")
            lines.append(f"      }}")
            lines.append(f"      std::memcpy(a.{d.bytes_field}, {d.param},"
                         f" static_cast<size_t>(_n) * sizeof({d.ctype}));")
            lines.append(f"      a.{d.count_field}   = _n;")
            lines.append(f"      a.{d.present_field} = 1;")
            lines.append(f"    }}")
        else:
            lines.append(f"    if ({d.param}) {{")
            lines.append(f"      std::memcpy(a.{d.bytes_field}, {d.param}, sizeof({d.ctype}));")
            lines.append(f"      a.{d.present_field} = 1;")
            lines.append(f"    }}")


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


# Per-API custom shim/handler bodies that the default emitter cannot express.
# Used when an API needs special in/out value handling that the generic
# pointer-vs-value heuristics get wrong. Kept here (not in the MANUAL_* sets) so
# the function still lives in the generated files and the dispatch tables wire it
# automatically.
CUSTOM_CAPTURE_SHIMS: Dict[str, str] = {
    # hipThreadExchangeStreamCaptureMode is an in/out swap: input *mode is the
    # desired thread capture mode, output *mode is the previous one. The generic
    # emitter records the POINTER; replay then can't restore the mode and a
    # hipMalloc bracketed by these calls during a graph capture fails with 900.
    # Record the input enum VALUE instead.
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
}

# ---------------------------------------------------------------------------
# Playback: APIs that must warn-and-skip when their destination pointer cannot
# be translated.
#
# dispatch_event() treats ANY non-hipSuccess handler return as fatal and aborts
# the whole replay.  So an API whose only device-pointer argument is a
# destination it writes must not hand a null pointer to the real API: the call
# returns hipErrorInvalidValue and one untranslatable buffer kills the entire
# replay of a customer archive.  A recorded destination legitimately has no
# alloc_map entry whenever the API that produced it is itself a playback no-op,
# e.g. hipMemAllocPitch (in NOOP_PLAYBACK_APIS), the idiomatic driver-API partner
# of hipMemsetD2D*.
#
# Maps API name -> destination parameter name.  Emits the codebase's standard
# untranslatable-pointer idiom (playback_hipFree, playback_hipMemRelease, ... all
# do `if (!live) return hipSuccess;`), plus a once-per-process warning naming the
# API so the lost fidelity is attributable.  Skipping loses one buffer's
# contents; aborting loses the whole replay.
SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS: Dict[str, str] = {
    "hipMemsetD2D8":       "dst",
    "hipMemsetD2D8Async":  "dst",
    "hipMemsetD2D16":      "dst",
    "hipMemsetD2D16Async": "dst",
    "hipMemsetD2D32":      "dst",
    "hipMemsetD2D32Async": "dst",
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
            _fill_derefs(lines, entry)
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
            _fill_derefs(lines, entry)
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
        if e.reserved:
            continue  # retired void* slot; leave the nullptr from the real table
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
        if e.reserved:
            continue
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
    'hipGraphNode_t':       ('ctx.translate_graph_node','ctx.record_graph_node','ctx.remove_graph_node'),
    'hipGraphExec_t':       ('ctx.translate_graph_exec','ctx.record_graph_exec','ctx.remove_graph_exec'),
    'hipSurfaceObject_t':   ('ctx.translate_surface',   'ctx.record_surface',   'ctx.remove_surface'),
    'hipTextureObject_t':   ('ctx.translate_texture',   'ctx.record_texture',   'ctx.remove_texture'),
    'hipCtx_t':             ('ctx.translate_ctx',       'ctx.record_ctx',       'ctx.remove_ctx'),
    'hipLinkState_t':       ('ctx.translate_link_state','ctx.record_link_state','ctx.remove_link_state'),
}
_PLAYBACK_HANDLE_TRANSLATE = {k: v[0] for k, v in _PLAYBACK_HANDLE_INFO.items()}
# The VMM handle has its own map (PlaybackContext::vmm_handle_map, filled by the
# hand-written hipMemCreate handler) but no record/remove pair, so it is not in
# the table above. Generated handlers still have to translate it: passing the
# recorded value names an allocation in the capturing process.
_PLAYBACK_HANDLE_TRANSLATE['hipMemGenericAllocationHandle_t'] = \
    'ctx.translate_vmm_handle'

# `void*` parameters that are really out buffers — the callee writes a handle
# through them. Nothing in the type says so, so the generator used to treat the
# recorded value as an ordinary opaque input and hand the runtime an address in
# the capturing process; measured outcome was a SIGSEGV inside
# hipMemExportToShareableHandle, which also cost every event recorded after it.
# The value is the number of bytes the callee writes, and the handler passes a
# zeroed local of that size.
#
# What the callee writes stays local to the replay: a POSIX fd or a Win32
# HANDLE is meaningful only in the process that received it, and the archive
# has no way to hand it to another one. hipMemImportFromShareableHandle is a
# NOOP for the same reason, so nothing downstream consumes it.
_OUT_BUFFER_PARAMS: Dict[str, Dict[str, int]] = {
    # An fd is an int and a Win32 HANDLE is a pointer; eight bytes covers both.
    'hipMemExportToShareableHandle':     {'shareableHandle': 8},
    'hipMemPoolExportToShareableHandle': {'shared_handle': 8},
    'hipMemGetHandleForAddressRange':    {'handle': 8},
    # A `char*` the callee fills rather than a string it reads: hipDeviceGetLuid
    # memcpys the low and high parts of the adapter LUID into luid[0..7]. Without
    # a size here the pointee rule would give it a single char to write eight
    # bytes into. Only the Windows path writes; Linux returns hipErrorNotSupported
    # before touching the buffer.
    'hipDeviceGetLuid':                  {'luid': 8},
}

# APIs that create device allocations: API name -> (rec_ptr_param, size_param)
# rec_ptr_param is the name of the output param storing the address (in the struct)
# size_param is the name of the size param in the struct
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
    'hipGraphDestroyNode':     ('node',          'hipGraphNode_t'),
    'hipCtxDestroy':           ('ctx',           'hipCtx_t'),
    'hipLinkDestroy':          ('state',         'hipLinkState_t'),
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
    # Memory pools. The two query spellings hand out a pool they did not
    # create — usually the device default — and registering it is what lets a
    # later hipMemSetMemPool or allocation name that pool.
    'hipMemPoolCreate':             [('mem_pool',     'hipMemPool_t')],
    'hipDeviceGetMemPool':          [('mem_pool',     'hipMemPool_t')],
    'hipDeviceGetDefaultMemPool':   [('mem_pool',     'hipMemPool_t')],
    'hipMemGetMemPool':             [('pool',         'hipMemPool_t')],
    # JIT linker state. Without this the state hipLinkCreate hands out was
    # dropped and hipLinkComplete / hipLinkDestroy were passed the capturing
    # process's pointer, which is why both failed however good the inputs were.
    'hipLinkCreate':                [('stateOut',     'hipLinkState_t')],
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
    # Graph nodes. Registering the node each construction call produces is
    # what lets a later dependency list, *NodeSetParams or hipGraphDestroyNode
    # name it; without this the node API replays as a set of unconnected calls.
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
    # Texture / surface objects
    'hipCreateSurfaceObject':       [('pSurfObject',  'hipSurfaceObject_t')],
    'hipCreateTextureObject':       [('pTexObject',   'hipTextureObject_t')],
    # Driver contexts. Every call that hands one out registers it, including
    # the two that only report a context they did not create: a recording
    # pushes back what it popped, and the pop is where replay learns which
    # live context that recorded value stands for.
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


# hipCtx_t arguments of the driver-API graph node calls. The recorded value is
# an address in the capturing process and nothing here maps it, so passing it
# through would hand the runtime a stale pointer. Null is not the answer
# either: hipDrvGraphAddMemcpyNode rejects a null context outright
# (hip_graph.cpp:1445). hrr_live_ctx substitutes the context the replaying
# thread is on, and preserves a recorded null so a call that failed that way at
# capture fails the same way here.
_LIVE_CTX_PLAYBACK_APIS: Set[str] = {
    "hipDrvGraphAddMemsetNode",
    "hipDrvGraphExecMemsetNodeSetParams",
}

# Lines emitted just before the handler returns, for post-call bookkeeping the
# tables above cannot express. `_r` holds the call's result.
_POST_CALL_EXTRA: Dict[str, List[str]] = {
    # A clone of a graph HRR could not rebuild is equally unbuildable, and
    # nothing about the clone call itself would reveal that.
    "hipGraphClone": [
        "if (_r == hipSuccess && ctx.graph_is_incomplete(a->originalGraph))",
        "  ctx.mark_graph_incomplete(a->pGraphClone, \"hipGraphClone\");",
    ],
    # Likewise a parent graph inherits whatever its child could not rebuild.
    "hipGraphAddChildGraphNode": [
        "if (_r == hipSuccess && ctx.graph_is_incomplete(a->childGraph))",
        "  ctx.mark_graph_incomplete(a->graph, \"hipGraphAddChildGraphNode\");",
    ],
}


def _playback_arg(entry: ApiEntry, p: Param, name: str,
                  pre_lines: List[str]) -> str:
    """Return the expression to pass to the real HIP API during playback."""
    t = p.raw_type.strip()
    base = _get_base_type(t)
    is_output = _is_output_ptr_param(p)

    # A declared out buffer — the recorded address belongs to the capturing
    # process, so the callee writes into a local of the right size instead.
    out_bytes = _OUT_BUFFER_PARAMS.get(entry.name, {}).get(name)
    if out_bytes:
        pre_lines.append(f"  alignas(8) unsigned char _outbuf_{name}"
                         f"[{out_bytes}]{{}};")
        return f"({t})_outbuf_{name}"

    # A by-value struct carried as inline bytes — restore it into a local.
    bv = by_value_struct(t)
    if bv is not None:
        pre_lines.append(f"  {base} _v_{name}{{}};")
        pre_lines.append(f"  std::memcpy(&_v_{name}, a->{name}_bytes,"
                         f" sizeof(_v_{name}));")
        pre_lines.extend(_deref_translate_lines(f"_v_{name}", bv.ptr_members))
        return f"_v_{name}"

    # A dereferenced pointer argument — rebuild the pointee and pass its address
    # rather than casting the recorded capture-time address back to a pointer.
    d = deref_for_param(entry.name, name)
    if d is not None and d.playback == "auto":
        if d.string:
            # The recorded characters are already NUL-terminated inside the
            # event, so the string is read straight out of the payload.
            return (f"(a->{d.present_field} ? ({t})a->{d.bytes_field}"
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
                # An array of handles: every element is itself a recorded
                # handle. A graph dependency list arrives as node addresses
                # from the capturing process; passed through untranslated the
                # runtime would either reject them or, worse, wire the node to
                # whatever lives at that address here.
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
            # An empty array is passed as a null pointer, not as the address of
            # an empty local: hipDrvGraphAddMemFreeNode (hip_graph.cpp:3502)
            # rejects a non-null dependency pointer with a zero count outright,
            # and a zero count is what every node added without dependencies
            # has.
            return f"({t})(_d_{name}_n ? _d_{name} : nullptr)"
        pre_lines.append(f"  {d.ctype} _d_{name}{{}};")
        # An output pointee is written by the call, so what matters is that the
        # buffer is the right size — the recorded value is carried for fidelity
        # and comparison, not to be handed back to the runtime. Passing a local
        # of the pointee's own type is also what stops the generic
        # output-pointer path from handing a 64-byte write an 8-byte void*.
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

    if base == 'hipCtx_t' and entry.name in _LIVE_CTX_PLAYBACK_APIS:
        return f"hrr_live_ctx(a->{name})"

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

    if entry.reserved:
        return (f"static hipError_t {fname}"
                f"(PlaybackContext& ctx, const uint8_t* payload) {{\n"
                f"  (void)ctx; (void)payload;\n"
                f"  return hipSuccess;\n"
                f"}}\n")

    # Error-stub playback APIs: graph work whose arguments the archive does not
    # carry well enough to rebuild. Emit a loud, attributable (per-API) warning
    # and poison the graph, but return hipSuccess — these are non-fatal on their
    # own. The HARD failure is at hipGraphInstantiate /
    # hipGraphInstantiateWithFlags, which refuse an incomplete graph (finding
    # H1). This keeps replay alive for programs that merely create, clone or
    # build graphs they never instantiate. Message is once/process.
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

    # Per-API custom handler body (special in/out value handling).
    if entry.name in CUSTOM_PLAYBACK_BODIES:
        return CUSTOM_PLAYBACK_BODIES[entry.name]

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
    skip_param       = SKIP_IF_UNMAPPED_PLAYBACK_APIS.get(entry.name)

    # Translate the destination once up front and skip the call when the
    # recorded address is in no map: returning an error here would abort the
    # whole replay (see SKIP_IF_UNMAPPED_PLAYBACK_APIS). The live pointer is
    # reused for the real call below, so there is no second lookup.
    if skip_param:
        lines.append(f"  void* _live_dst = ctx.translate_ptr(a->{skip_param});")
        lines.append(f"  if (_live_dst == nullptr) {{")
        lines.append(f"    static bool warned = false;")
        lines.append(f"    if (!warned) {{")
        lines.append(f"      warned = true;")
        lines.append(f"      fprintf(stderr, \"[HRR] {entry.name}: recorded destination 0x%llx \"")
        lines.append(f"              \"is not in the alloc map (memory HRR does not track); \"")
        lines.append(f"              \"skipping this write. Results may differ from capture.\\n\",")
        lines.append(f"              (unsigned long long)a->{skip_param});")
        lines.append(f"    }}")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    # Untranslatable destination -> warn once and skip (never abort the replay).
    # See SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS.
    skip_dst = SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS.get(entry.name)
    if skip_dst:
        lines.append(f"  void* _live_{skip_dst} = ctx.translate_ptr(a->{skip_dst});")
        lines.append(f"  if (!_live_{skip_dst}) {{")
        lines.append(f"    static bool warned = false;")
        lines.append(f"    if (!warned) {{")
        lines.append(f"      warned = true;")
        lines.append(f"      fprintf(stderr, \"[HRR] {entry.name}: {skip_dst} 0x%llx has no alloc_map entry \"")
        lines.append(f"              \"(its allocation was not replayed), so this call is skipped; the \"")
        lines.append(f"              \"destination buffer will differ from capture.\\n\",")
        lines.append(f"              (unsigned long long)a->{skip_dst});")
        lines.append(f"    }}")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    # A node or executable graph that was never built here cannot be mutated.
    # Handing the runtime a null handle would come back as an error, and
    # dispatch treats any handler error as fatal — so a graph HRR already
    # refused to instantiate (loudly, at the instantiate call) would take the
    # rest of the replay down with it through its follow-up mutations. The
    # useful complaint has already been made; skip these.
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

        # Destination already translated (and null-checked) above
        if skip_dst and name == skip_dst:
            call_args.append(f"({p.raw_type.strip()})_live_{skip_dst}")
            continue
        # For alloc-free: replace the pointer arg with the translated live ptr
        if is_alloc_free and name == _ALLOC_FREE_APIS[entry.name]:
            call_args.append("_live_ptr")
            continue
        # For skip-if-unmapped: reuse the destination translated above
        if skip_param and name == skip_param:
            call_args.append("_live_dst")
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

        # The element count of a deref'd array is however many elements the
        # event actually carries, which is not the recorded count when the
        # array overflowed the inline capacity.
        array_owner = next((d for d in deref_specs(entry.name)
                            if d.is_array and d.count == name
                            and d.playback == "auto"), None)
        if array_owner is not None:
            call_args.append(f"({p.raw_type.strip()})_d_{array_owner.param}_n")
            continue

        call_args.append(_playback_arg(entry, p, name, pre_lines))

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

    # A call that failed at capture and fails identically here was reproduced,
    # not botched: return success so dispatch_event does not abort the replay
    # over it. Only the hand-written capture shims record failing calls at all,
    # so for everything else a->ret is 0 and this is dead weight.
    if not void_ret and _get_base_type(entry.ret_type) == 'hipError_t':
        lines.append(f"  if (_r != hipSuccess && a->ret != 0 &&"
                     f" static_cast<int32_t>(_r) == a->ret) {{")
        lines.append(f"    hrr_note_recorded_error(ctx, \"{entry.name}\", a->ret);")
        lines.append(f"    return hipSuccess;")
        lines.append(f"  }}")

    # Post-call: register/unregister allocs and handles
    success_cond = "true" if void_ret else "_r == hipSuccess"

    if is_alloc_create:
        rec_param, sz_param = _ALLOC_CREATE_APIS[entry.name]
        # The output ptr is stored in a local _out_{rec_param} by _playback_arg
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
    n_reserved       = sum(1 for e in entries if e.reserved)
    n_manual_cap     = sum(1 for e in entries if e.name in MANUAL_CAPTURE_APIS)
    n_manual_play    = sum(1 for e in entries if e.name in MANUAL_PLAYBACK_APIS)
    n_noop_play      = sum(1 for e in entries if e.name in NOOP_PLAYBACK_APIS)
    print(f"  Found {n_compiler} compiler + {n_runtime} runtime = {len(entries)} total"
          f" ({n_reserved} reserved slots)")

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
        ("SKIP_IF_UNMAPPED_PLAYBACK_APIS", set(SKIP_IF_UNMAPPED_PLAYBACK_APIS)),
        ("ERROR_STUB_PLAYBACK_APIS", ERROR_STUB_PLAYBACK_APIS),
        ("SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS",
         set(SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS.keys())),
        ("EXTRA_FIELDS",         set(EXTRA_FIELDS.keys())),
        ("DEREF_FIELDS",         set(DEREF_FIELDS.keys())),
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

    # generate_playback_shim() returns early for no-op / error-stub / custom /
    # manual APIs, so a skip-if-unmapped entry that is also in one of those sets
    # would have no effect at all. Fail loudly rather than silently.
    shadowed = sorted(
        set(SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS)
        & (NOOP_PLAYBACK_APIS | ERROR_STUB_PLAYBACK_APIS
           | set(CUSTOM_PLAYBACK_BODIES) | MANUAL_PLAYBACK_APIS))
    if shadowed:
        print("\nERROR: SKIP_IF_UNMAPPED_DST_PLAYBACK_APIS entries have no effect "
              "because the API is handled earlier:")
        for n in shadowed:
            print(f"  '{n}'")
        sys.exit(1)

    # A Deref that names a parameter the API does not have would silently do
    # nothing, which is indistinguishable from the payload loss it is there to
    # fix. The same goes for an array whose count parameter does not exist.
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

    # generate_playback_shim() applies the classes in a fixed order, so an API
    # in two of them silently gets whichever comes first.
    for a_name, a_set, b_name, b_set in [
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
