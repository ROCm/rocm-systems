#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Post-hipify transformation: convert ncclShmem references to LDS pointers.
#
# This enables the compiler to emit ds_ (Local Data Share) instructions instead
# of flat_ instructions for shared memory accesses, avoiding the ~40us latency
# penalty from flat address resolution in the split-compiled device functions.
#
# On gfx942/gfx950 hardware, flat and ds instructions accessing the same LDS
# location are NOT coherent. Every pointer derived from LDS must stay in
# addrspace(3) throughout its lifetime to avoid silent data corruption.
#
# Applied to device header files and common.cu.cpp.

import re
import sys
import os

def is_device_header(path):
    return bool(re.search(r'/src/device/.*\.h$', path))

def is_common_cu(path):
    return path.endswith('common.cu.cpp')

def is_generated_device_header(path):
    return bool(re.search(r'/gensrc/device_table.*\.h$', path))

def should_process(path):
    return is_device_header(path) or is_common_cu(path) or is_generated_device_header(path)


# ---------------------------------------------------------------------------
# F. Include guards (#pragma once)
# ---------------------------------------------------------------------------
def add_pragma_once(text, path):
    if '#pragma once' not in text and is_device_header(path):
        text = '#pragma once\n' + text
    return text


# ---------------------------------------------------------------------------
# Add #include "rccl_ptr.h" if not already present
# ---------------------------------------------------------------------------
def add_rccl_ptr_include(text, path):
    if path.endswith('rccl_ptr.h'):
        return text
    if '#include "rccl_ptr.h"' not in text:
        text = re.sub(
            r'(#include "common\.h")',
            r'\1\n#include "rccl_ptr.h"',
            text, count=1
        )
        if '#include "rccl_ptr.h"' not in text:
            text = re.sub(
                r'(#include "device\.h")',
                r'\1\n#include "rccl_ptr.h"',
                text, count=1
            )
        if '#include "rccl_ptr.h"' not in text:
            text = '#include "rccl_ptr.h"\nstruct ncclShmemData;\n' + text
    return text


# ---------------------------------------------------------------------------
# A. LDS type conversions
# ---------------------------------------------------------------------------
def apply_lds_type_conversions(text):
    # A1. ncclShmemData reference -> LDSPtr<ncclShmemData>
    text = text.replace('struct ncclShmemData& ncclShmem', 'LDSPtr<ncclShmemData> ncclShmem')
    text = text.replace('ncclShmemData& ncclShmem', 'LDSPtr<ncclShmemData> ncclShmem')
    text = text.replace('struct ncclShmemData& shmem', 'LDSPtr<ncclShmemData> shmem')
    text = text.replace('ncclShmemData& shmem', 'LDSPtr<ncclShmemData> shmem')

    # A2. void* ncclShmemPerWarp -> ncclShmemPerWarpPtr ncclShmemPerWarp
    text = re.sub(r'void\*\s*ncclShmemPerWarp', 'ncclShmemPerWarpPtr ncclShmemPerWarp', text)

    # A3. Member access: dot -> arrow for LDSPtr variables
    text = text.replace('ncclShmem.', 'ncclShmem->')
    text = text.replace('ncclShmem).', 'ncclShmem)->')
    # Constructor parameter named 'shmem' in prims_simple.h, prims_ll.h, prims_ll128.h
    text = text.replace('shmem.', 'shmem->')

    # A4. Barrier members
    text = text.replace('uint64_t* barriers;', 'LDSPtr<uint64_t> barriers;')
    text = text.replace('uint64_t* barriers_pat;', 'LDSPtr<uint64_t> barriers_pat;')

    # A5. Work pointer types (order: longer names first)
    text = text.replace('struct ncclDevWorkCollReg*', 'LDSPtr<ncclDevWorkCollReg>')
    text = re.sub(r'\(ncclDevWorkCollReg\*\)', '(LDSPtr<ncclDevWorkCollReg>)', text)
    text = text.replace('struct ncclDevWorkColl*', 'LDSPtr<ncclDevWorkColl>')
    text = re.sub(r'\(ncclDevWorkColl\*\)', '(LDSPtr<ncclDevWorkColl>)', text)
    text = text.replace('struct ncclDevWorkP2p*', 'LDSPtr<ncclDevWorkP2p>')
    text = re.sub(r'\(ncclDevWorkP2p\*\)', '(LDSPtr<ncclDevWorkP2p>)', text)

    # A6. void** intermediates
    text = text.replace('void **ptrs', 'LDSPtr<void*> ptrs')
    text = re.sub(r'void\*\* srcs = ncclShmem->', 'LDSPtr<void*> srcs = ncclShmem->', text)
    text = re.sub(r'void\*\* dsts = ncclShmem->', 'LDSPtr<void*> dsts = ncclShmem->', text)

    # A7. ncclPatPeer pointer conversions
    text = re.sub(
        r'struct ncclPatPeer\* peer = \(\(struct ncclPatPeer\*\)recvPeers\)\+tid',
        'LDSPtr<ncclPatPeer> peer = (LDSPtr<ncclPatPeer>)(((ncclPatPeer*)recvPeers)+tid)',
        text
    )
    text = re.sub(
        r'peer = \(\(struct ncclPatPeer\*\)sendPeers\)\+tid',
        'peer = (LDSPtr<ncclPatPeer>)(((ncclPatPeer*)sendPeers)+tid)',
        text
    )
    text = text.replace('struct ncclPatPeer* peer = NULL', 'LDSPtr<ncclPatPeer> peer = nullptr')
    # After blanket ncclDevWorkColl replacement may have mangled ncclPatPeer lines
    text = text.replace('LDSPtr<ncclDevWorkColl> peer = NULL', 'LDSPtr<ncclPatPeer> peer = nullptr')
    text = text.replace('LDSPtr<ncclDevWorkColl> peer = nullptr', 'LDSPtr<ncclPatPeer> peer = nullptr')

    # A8. parallelFactor polling coherence fix
    text = re.sub(
        r'volatile int\* pfPtr = &shmem->parallelFactor',
        'LDSPtr<volatile int> pfPtr = &shmem->parallelFactor',
        text
    )

    # A9. loadMeta pointer (unpack.h)
    text = re.sub(r'loadMeta\*\s*s_meta;', 'LDSPtr<loadMeta> s_meta;', text)
    text = re.sub(r'loadMeta \*s_meta;', 'LDSPtr<loadMeta> s_meta;', text)
    text = re.sub(
        r's_meta = \(loadMeta\*\)\s*ncclScratchForWarp\(',
        's_meta = ncclScratchForWarp<loadMeta>(',
        text
    )

    # A10. shmemCvtPtr -> LDSPtr (unpack.h)
    text = re.sub(
        r'shmemCvtPtr\(\(uint64_t \*\)\(s_meta \+ \(w \* PPW \+ t\)\)\)',
        'LDSPtr<uint64_t>(s_meta + (w * PPW + t))',
        text
    )
    text = re.sub(
        r'shmemCvtPtr\(\(uint64_t\*\) \(s_meta \+ meta_idx\)\)',
        'LDSPtr<uint64_t>(s_meta + meta_idx)',
        text
    )

    # A11a. op128.h: remove shmemCvtPtr function (replaced by macro in rccl_ptr.h)
    text = re.sub(
        r'inline __device__ uint64_t\* shmemCvtPtr\(volatile uint64_t\* shmemGenericPtr\) \{\n'
        r'  return \(uint64_t\*\)shmemGenericPtr;\n'
        r'\}\n',
        '',
        text
    )

    # A11. op128.h: loadShmem128/storeShmem128/loadShmemMisaligned128
    text = re.sub(
        r'inline __device__ void loadShmem128\(uint64_t\* shmemAsmPtr',
        'inline __device__ void loadShmem128(LDSPtr<uint64_t> ptr',
        text
    )
    text = re.sub(
        r'inline __device__ void storeShmem128\(uint64_t\* shmemAsmPtr',
        'inline __device__ void storeShmem128(LDSPtr<uint64_t> ptr',
        text
    )
    text = re.sub(
        r'inline __device__ void loadShmemMisaligned128\(T \*ptr',
        'inline __device__ void loadShmemMisaligned128(LDSPtr<T> ptr',
        text
    )
    # shmemAsmPtr dereferences -> array subscript
    text = text.replace('*(shmemAsmPtr)', 'ptr[0]')
    text = text.replace('*(shmemAsmPtr+1)', 'ptr[1]')

    # ncclScratchForWarp definition is now transformed in apply_common_h_structural.

    # ncclScratchForWarp type conversions in call-site files
    # prims_ll128.h
    text = re.sub(
        r'uint64_t \*shm8 = shmemCvtPtr\(\(uint64_t\*\)ncclScratchForWarp\(([^;]*)\)\);',
        r'LDSPtr<uint64_t> shm8 = ncclScratchForWarp<uint64_t>(\1);',
        text
    )
    text = re.sub(
        r'T \*shm = \(T\*\)ncclScratchForWarp\(',
        'LDSPtr<T> shm = ncclScratchForWarp<T>(',
        text
    )
    text = re.sub(
        r'T \*shm = \(T\*\)shm8',
        'LDSPtr<T> shm = LDSPtr<T>(shm8)',
        text
    )
    # sendrecv.h
    text = re.sub(
        r'Shared\* shared = \(Shared\*\)ncclScratchForWarp\(',
        'LDSPtr<Shared> shared = ncclScratchForWarp<Shared>(',
        text
    )
    # all_gather.h, reduce_scatter.h: ncclPatShmem
    text = re.sub(
        r'struct ncclPatShmem\* shmem = \(struct ncclPatShmem\*\)ncclScratchForWarp\(',
        'LDSPtr<ncclPatShmem> shmem = ncclScratchForWarp<ncclPatShmem>(',
        text
    )
    # reduce_scatter.h: void** srcPtrs
    text = re.sub(
        r'void\*\* srcPtrs = \(void\*\*\)ncclScratchForWarp\(',
        'LDSPtr<void*> srcPtrs = ncclScratchForWarp<void*>(',
        text
    )
    # common.h: uint8_t* fnsOfBitset
    text = re.sub(
        r'uint8_t\* fnsOfBitset = \(uint8_t\*\)ncclScratchForWarp\(',
        'LDSPtr<uint8_t> fnsOfBitset = ncclScratchForWarp<uint8_t>(',
        text
    )
    # patReduce/patCopy parameter
    text = re.sub(
        r'struct ncclPatShmem\* shmem\)',
        'LDSPtr<ncclPatShmem> shmem)',
        text
    )

    return text


# ---------------------------------------------------------------------------
# B. Function signature additions
# ---------------------------------------------------------------------------
def apply_function_signature_additions(text):
    # B1. runRing and similar collective entry points
    # Pattern: runRing(int tid, int nthreads, struct ncclDevWorkColl* work)
    # The ncclDevWorkColl* was already converted to LDSPtr above
    text = re.sub(
        r'void runRing\(int tid, int nthreads, (LDSPtr<ncclDevWorkColl>) work\)',
        r'void runRing(int tid, int nthreads, \1 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )
    # runTreeSplit, runTreeUpDown - same pattern
    text = re.sub(
        r'void runTreeSplit\(int tid, int nthreads, (LDSPtr<ncclDevWorkColl>) work\)',
        r'void runTreeSplit(int tid, int nthreads, \1 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'void runTreeUpDown\(int tid, int nthreads, (LDSPtr<ncclDevWorkColl>) work\)',
        r'void runTreeUpDown(int tid, int nthreads, \1 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # RunWorkColl::run -- match various parameter name styles (nthreads, /*nthreads*/, tn)
    text = re.sub(
        r'void run\(int tid, (int[^,]*), (LDSPtr<ncclDevWorkColl>) work\)',
        r'void run(int tid, \1, \2 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # sendrecv.h: runSend / runRecv
    text = re.sub(
        r'void runSend\(int tid, int tn, int group, (LDSPtr<ncclDevWorkP2p>) work\)',
        r'void runSend(int tid, int tn, int group, \1 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'void runRecv\(int tid, int tn, int group, (LDSPtr<ncclDevWorkP2p>) work\)',
        r'void runRecv(int tid, int tn, int group, \1 work, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # RunWorkBatch::run() (no params version)
    text = re.sub(
        r'(void run\(\))\s*\{',
        r'void run(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp) {',
        text
    )

    # loadWorkBatchToShmem: add ncclShmem, ncclShmemPerWarp at start of params
    text = re.sub(
        r'void loadWorkBatchToShmem\(\s*int tid,',
        'void loadWorkBatchToShmem(\n    LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp,\n    int tid,',
        text
    )

    # insert_random_delay_per_warp: add ncclShmem param
    text = re.sub(
        r'void insert_random_delay_per_warp\(\)',
        'void insert_random_delay_per_warp(LDSPtr<ncclShmemData> ncclShmem)',
        text
    )

    # profilerEnabled: add ncclShmem param
    text = re.sub(
        r'bool profilerEnabled\(int workItemIdx\)',
        'bool profilerEnabled(LDSPtr<ncclShmemData> ncclShmem, int workItemIdx)',
        text
    )

    # profiler: add ncclShmem param
    text = re.sub(
        r'void profiler\(int action\)',
        'void profiler(LDSPtr<ncclShmemData> ncclShmem, int action)',
        text
    )

    # sendPeerNotify: add ncclShmem param
    text = re.sub(
        r'void sendPeerNotify\(int peer, int connIndex, int steps\)',
        'void sendPeerNotify(int peer, int connIndex, int steps, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )

    # recvPeerNotify: add ncclShmem param
    text = re.sub(
        r'void recvPeerNotify\(int peer, int connIndex, int steps\)',
        'void recvPeerNotify(int peer, int connIndex, int steps, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )

    # checkAbort: add ncclShmem param after spins
    text = re.sub(
        r'(int checkAbort\([^)]*int\s*&\s*spins)\)',
        r'\1, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )

    # GenericOp callback operator(): add ncclShmem at end
    # These are nested struct operator() functions used as callbacks from genericOp.
    # Pattern: "uint32_t sendDirectFlag, uint32_t recvDirectFlag\n      ) {"
    text = re.sub(
        r'(uint32_t sendDirectFlag, uint32_t recvDirectFlag)\n(\s*)\) \{',
        r'\1,\n\2    LDSPtr<ncclShmemData> ncclShmem\n\2) {',
        text
    )

    # ncclNetDevice* definitions: use specific parameter types to match only definitions
    text = re.sub(
        r'void ncclNetDeviceUnpackSetup\(void\* ohandle, const int group, const int index\)',
        r'void ncclNetDeviceUnpackSetup(void* ohandle, const int group, const int index, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )
    text = re.sub(
        r'void ncclNetDeviceIncrementHead\(const int group, const int index\)',
        r'void ncclNetDeviceIncrementHead(const int group, const int index, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )
    text = re.sub(
        r'void ncclNetDeviceSaveHead\(void\* ohandle, const int group, const int index\)',
        r'void ncclNetDeviceSaveHead(void* ohandle, const int group, const int index, LDSPtr<ncclShmemData> ncclShmem)',
        text
    )
    # ncclNetDeviceUnpack: match by unique closing parameter pattern
    # This matches the template declaration, and both Recv=0 and Recv=1 specializations
    text = re.sub(
        r'(void ncclNetDeviceUnpack[^(]*\([^)]*int Src, int workSize)\)',
        r'\1, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # ncclNetDeviceUnpackInner: match declaration and definition
    text = re.sub(
        r'(void ncclNetDeviceUnpackInner\(\s*'
        r'const int tid, const int tidInBlock, const int nworkers, const int group, const int index,\s*'
        r'void \*src, const int nbytes, const uint64_t step)\)',
        r'\1, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # ncclKernelMain signature is transformed in apply_common_h_structural.

    # Proto::calcBytePerStep: add ncclShmem
    text = re.sub(
        r'(static int calcBytePerStep\()\)',
        r'\1LDSPtr<ncclShmemData> ncclShmem)',
        text
    )

    return text


# ---------------------------------------------------------------------------
# C. Macro definition parameterization
# ---------------------------------------------------------------------------
def apply_macro_parameterization(text):
    # Macros like INC_COLL_TRACE, traceKernelLaunch, traceData, traceKernelEnd,
    # traceAbort, __insert_timestamp, and barrier_generic all reference ncclShmem
    # directly. Since ncclShmem is always in scope (as a function parameter or
    # class member) at every call site, NO parameterization is needed -- the A3
    # dot-to-arrow conversion handles the macro bodies automatically.
    #
    # Only rccl*RunRingSimpleProtoImpl macros need parameterization because they
    # pass ncclShmem/ncclShmemPerWarp to function calls.
    text = re.sub(
        r'#define rcclAllGatherRunRingSimpleProtoImpl\(tid, nthreads, work\)',
        '#define rcclAllGatherRunRingSimpleProtoImpl(tid, nthreads, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'#define rcclReduceScatterRunRingSimpleProtoImpl\(tid, nthreads, work\)',
        '#define rcclReduceScatterRunRingSimpleProtoImpl(tid, nthreads, work, ncclShmem, ncclShmemPerWarp)',
        text
    )

    return text


# ---------------------------------------------------------------------------
# D. Primitives class changes
# ---------------------------------------------------------------------------
def apply_primitives_class_changes(text):
    # D1. Add members: ncclShmem and ncclShmemPerWarp
    # Use "uint64_t barrier_next = 0;" as anchor - present in all three
    # Primitives class variants (prims_simple.h, prims_ll.h, prims_ll128.h).
    # For prims_simple.h which has barriers_pat, anchor after barrier_next_pat.
    text = re.sub(
        r'(uint64_t barrier_next_pat = 0;\n)',
        r'\1  LDSPtr<ncclShmemData> ncclShmem;\n  ncclShmemPerWarpPtr ncclShmemPerWarp;\n',
        text
    )
    # For ll / ll128 which don't have barrier_next_pat, anchor after barrier_next
    if 'LDSPtr<ncclShmemData> ncclShmem;' not in text:
        text = re.sub(
            r'(uint64_t barrier_next = 0;\n)',
            r'\1  LDSPtr<ncclShmemData> ncclShmem;\n  ncclShmemPerWarpPtr ncclShmemPerWarp;\n',
            text
        )

    # D2. Constructor: insert shmem, ncclShmemPerWarp_ after redOpArg
    text = re.sub(
        r'(void const \*inputBuf, void \*outputBuf, uint64_t redOpArg),\s*'
        r'(uint8_t group=0)',
        r'\1,\n      LDSPtr<ncclShmemData> shmem, ncclShmemPerWarpPtr ncclShmemPerWarp_,\n      \2',
        text
    )

    # D2b. prims_ll.h secondary constructor: also needs shmem params
    # Pattern: uint64_t redOpArg, uint8_t group, (no default =0)
    text = re.sub(
        r'(void const \*inputBuf, void \*outputBuf, uint64_t redOpArg), (uint8_t group,)',
        r'\1,\n      LDSPtr<ncclShmemData> shmem, ncclShmemPerWarpPtr ncclShmemPerWarp_,\n      \2',
        text
    )

    # D3. Constructor initializer: add ncclShmem(shmem), ncclShmemPerWarp(ncclShmemPerWarp_)
    # For prims_simple.h and prims_ll128.h: init list ends with threadsPerBlock(blockDim.x){
    text = re.sub(
        r'threadsPerBlock\(blockDim\.x\)\{',
        'threadsPerBlock(blockDim.x),\n    ncclShmem(shmem), ncclShmemPerWarp(ncclShmemPerWarp_){',
        text
    )
    # For prims_ll.h: init list ends with sizeof(ncclLLFifoLine)) {
    text = re.sub(
        r'(sizeof\(ncclLLFifoLine\)\)) \{',
        r'\1,\n    ncclShmem(shmem), ncclShmemPerWarp(ncclShmemPerWarp_) {',
        text
    )

    # D3b. Use shmem (parameter) instead of ncclShmem for barrier init
    text = re.sub(
        r'barriers = &ncclShmem->groups\[group\]\.barrier;',
        'barriers = &shmem->groups[group].barrier;',
        text
    )
    text = re.sub(
        r'barriers_pat = &ncclShmem->barrier_pat;',
        'barriers_pat = &shmem->barrier_pat;',
        text
    )

    # D3c. stepSize init: use shmem instead of ncclShmem in initializer lists
    # prims_simple.h: stepSize(stepSize_ == 0 ? ncclShmem->comm...)
    text = re.sub(
        r'stepSize\(stepSize_ == 0 \? ncclShmem->comm',
        'stepSize(stepSize_ == 0 ? shmem->comm',
        text
    )
    # prims_ll128.h: stepSize(ncclShmem->comm.buffSizes[NCCL_PROTO_LL128]...)
    text = re.sub(
        r'stepSize\(ncclShmem->comm\.buffSizes',
        'stepSize(shmem->comm.buffSizes',
        text
    )
    # prims_ll.h: stepLines(ncclShmem->comm.buffSizes[NCCL_PROTO_LL]...)
    text = re.sub(
        r'stepLines\(ncclShmem->comm\.buffSizes',
        'stepLines(shmem->comm.buffSizes',
        text
    )
    text = re.sub(
        r'group\(ncclShmem->warpComm',
        'group(shmem->warpComm',
        text
    )

    # D3d. channel init in constructors: use shmem instead of ncclShmem
    # These lines are inside the constructor body but reference ncclShmem before
    # the member is fully initialized. However, since they come AFTER the
    # initializer list, the member IS initialized by then, so this is actually
    # OK. The member ncclShmem is initialized in the init list.
    # We do NOT convert these - the member ncclShmem-> is correct here.

    return text


# ---------------------------------------------------------------------------
# E. Function/macro call site updates
# ---------------------------------------------------------------------------
def apply_call_site_updates(text):
    # E1. ncclScratchForWarp(warp) -> ncclScratchForWarp<T>(ncclShmemPerWarp, warp)
    # (this was already handled by the individual file patterns in A section)

    # E3. checkAbort call sites: add ncclShmem
    # Only match call sites (args don't start with type keywords like 'int', 'const')
    text = re.sub(
        r'checkAbort\((\w+), (\w+), (\w+)\)',
        r'checkAbort(\1, \2, \3, ncclShmem)',
        text
    )

    # E5. ncclNetDevice* call sites: add ncclShmem
    # Use negative lookahead to avoid matching definitions (which have 'void' before the name)
    text = re.sub(
        r'(?<!void )ncclNetDeviceIncrementHead\((\w+), (\w+)\)',
        r'ncclNetDeviceIncrementHead(\1, \2, ncclShmem)',
        text
    )
    text = re.sub(
        r'(?<!void )ncclNetDeviceUnpackSetup\((\w+), (\w+), (\w+)\)',
        r'ncclNetDeviceUnpackSetup(\1, \2, \3, ncclShmem)',
        text
    )
    text = re.sub(
        r'(?<!void )ncclNetDeviceSaveHead\((\w+), (\w+), (\w+)\)',
        r'ncclNetDeviceSaveHead(\1, \2, \3, ncclShmem)',
        text
    )
    # ncclNetDeviceUnpack template call: add ncclShmem, ncclShmemPerWarp
    text = re.sub(
        r'(ncclNetDeviceUnpack<[^>]+>\([^)]+), (workSize)\)',
        r'\1, \2, ncclShmem, ncclShmemPerWarp)',
        text
    )
    # ncclNetDeviceUnpackInner call site: add ncclShmem, ncclShmemPerWarp
    text = re.sub(
        r'(ncclNetDeviceUnpackInner\([^)]+devicePlugin\.unpack\.head\[\w+\])\)',
        r'\1, ncclShmem, ncclShmemPerWarp)',
        text
    )

    # E6. sendPeerNotify / recvPeerNotify call sites: add ncclShmem
    # Avoid matching definitions (use negative lookbehind for 'void ')
    text = re.sub(
        r'(?<!void )sendPeerNotify\((\w+), (\w+), (\w+)\)',
        r'sendPeerNotify(\1, \2, \3, ncclShmem)',
        text
    )
    text = re.sub(
        r'(?<!void )recvPeerNotify\((\w+), (\w+), (\w+)\)',
        r'recvPeerNotify(\1, \2, \3, ncclShmem)',
        text
    )

    # E7. Primitives constructor: insert ncclShmem, ncclShmemPerWarp after redOpArg
    # The D2 rule added these as constructor params right after redOpArg.
    # Match work->redOpArg followed by a digit or 'group' (the group arg). This avoids
    # matching inside reduceCopy calls where redOpArg is followed by &, false, etc.
    text = re.sub(
        r'work->redOpArg,(?! ncclShmem) (\d|group)',
        r'work->redOpArg, ncclShmem, ncclShmemPerWarp, \1',
        text
    )
    # sendrecv.h / alltoall_pivot.h: Primitives constructor with literal 0 for redOpArg
    # Two patterns: "0, group" (sendrecv) and "0)" (alltoall_pivot, no more args)
    text = re.sub(
        r'/\*redOpArg\(ignored\)=\*/0, (group)',
        r'/*redOpArg(ignored)=*/0, ncclShmem, ncclShmemPerWarp, \1',
        text
    )
    text = re.sub(
        r'/\*redOpArg\(ignored\)=\*/0\)',
        '/*redOpArg(ignored)=*/0, ncclShmem, ncclShmemPerWarp)',
        text
    )
    # prims_ll.h delegating constructor: use parameter names (shmem, ncclShmemPerWarp_)
    # not member names (ncclShmem, ncclShmemPerWarp) to avoid uninitialized access
    text = re.sub(
        r'outputBuf, redOpArg, group,\n\s+connIndexRecv',
        'outputBuf, redOpArg, shmem, ncclShmemPerWarp_, group,\n                  connIndexRecv',
        text
    )

    # E8. fn.template operator()(): add ncclShmem at end
    text = re.sub(
        r'(sendDirectFlag, recvDirectFlag)\)',
        r'\1,\n            ncclShmem)',
        text
    )

    # E9. insert_random_delay_per_warp() -> insert_random_delay_per_warp(ncclShmem)
    text = re.sub(
        r'insert_random_delay_per_warp\(\)',
        'insert_random_delay_per_warp(ncclShmem)',
        text
    )

    # profiler(action) -> profiler(ncclShmem, action)
    text = re.sub(
        r'profiler\((START|STOP|FINI)\)',
        r'profiler(ncclShmem, \1)',
        text
    )

    # profilerEnabled(idx++) -> profilerEnabled(ncclShmem, idx++)
    # Avoid matching the definition (which contains 'int' or 'LDSPtr')
    # Negative lookahead (?!ncclShmem) prevents double-application
    text = re.sub(
        r'(?<!bool )profilerEnabled\((?!ncclShmem)([^)]*\b(?:idx|workItemIdx)[^)]*)\)',
        r'profilerEnabled(ncclShmem, \1)',
        text
    )

    # Macro call sites (traceKernelLaunch, traceKernelEnd, traceData,
    # __insert_timestamp, barrier_generic) don't need updating -- they use
    # ncclShmem from the enclosing scope.

    # E1. ncclScratchForWarp: add ncclShmemPerWarp as first arg
    # After the A section, calls look like ncclScratchForWarp<T>(expr)
    # Template arg can be void*, T, uint64_t, etc. -- use [^>]+ not \w+
    text = re.sub(
        r'ncclScratchForWarp<([^>]+)>\((?!ncclShmemPerWarp)([^)]+)\)',
        r'ncclScratchForWarp<\1>(ncclShmemPerWarp, \2)',
        text
    )

    # E10. loadWorkBatchToShmem: add ncclShmem, ncclShmemPerWarp at start
    text = re.sub(
        r'loadWorkBatchToShmem\(([^,]+), ([^,]+), (args)',
        r'loadWorkBatchToShmem(ncclShmem, ncclShmemPerWarp, \1, \2, \3',
        text
    )

    # E11. RunWorkColl::run and SpecializedRunWorkBatch::run: add params
    # Match all call patterns including WarpSpeed variant (tid % WARP_SIZE, WARP_SIZE, work)
    text = re.sub(
        r'RunWorkColl<([^>]+)>\(\)\.run\(([^)]+), ([^)]+), work\)',
        r'RunWorkColl<\1>().run(\2, \3, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'SpecializedRunWorkBatch\(\)\.run\(\)',
        'SpecializedRunWorkBatch().run(ncclShmem, ncclShmemPerWarp)',
        text
    )

    # E12. device_table.h forward declarations: add LDS pointer params so the
    # mangled names match the DEFINE_ncclDevFunc definitions.
    # When IFC is disabled, generate.py emits __attribute__((noinline)) between
    # __device__ and void, so the regex must allow for optional attributes.
    text = re.sub(
        r'(__device__\s+(?:__attribute__\(\([^)]*\)\)\s+)?void\s+ncclDevFunc_\w+)\(\s*\)',
        r'\1(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp)',
        text
    )

    # E12b. ncclDevFuncPtr_t typedef: match the new function signature
    text = re.sub(
        r'typedef\s+void\s*\(\s*\*\s*ncclDevFuncPtr_t\s*\)\s*\(\s*\)',
        r'typedef void(*ncclDevFuncPtr_t)(LDSPtr<ncclShmemData>, ncclShmemPerWarpPtr)',
        text
    )

    # E12c. Indirect function table calls: pass LDS pointer args
    text = re.sub(
        r'(ncclDevFuncTable_\d+\[.*?\])\(\s*\)',
        r'\1(ncclShmem, ncclShmemPerWarp)',
        text
    )

    # E12d. Non-IFC binary-search dispatch: thread LDS pointers through the
    # Caller template and NCCL_CALL_FUNCTIONS wrapper generated by generate.py.
    # Definitions: add LDS pointer parameters to call*/NCCL_CALL_FUNCTIONS_*.
    text = re.sub(
        r'(void\s+(?:call\d+|NCCL_CALL_FUNCTIONS_\d+)\(unsigned short funcIndex)\)\s*noexcept',
        r'\1, LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp) noexcept',
        text
    )
    # Caller recursive calls and NCCL_CALL_FUNCTIONS -> Caller delegation:
    # pass LDS pointer args alongside funcIndex.
    text = re.sub(
        r'(Caller\d+<[^>]+>::call\d+\(funcIndex)\)',
        r'\1, ncclShmem, ncclShmemPerWarp)',
        text
    )
    # NCCL_CALL_FUNCTIONS_* call sites (common.h): pass LDS pointer args.
    text = re.sub(
        r'(NCCL_CALL_FUNCTIONS_\d+)\((?!unsigned\b)([^)]+)\)',
        r'\1(\2, ncclShmem, ncclShmemPerWarp)',
        text
    )

    # E13. Proto::calcBytePerStep() -> Proto::calcBytePerStep(ncclShmem)
    text = re.sub(
        r'Proto::calcBytePerStep\(\)',
        'Proto::calcBytePerStep(ncclShmem)',
        text
    )

    # runRing call sites: add ncclShmem, ncclShmemPerWarp
    # These calls may have template arguments with nested <> (e.g. ProtoSimple<1,1,0,8>).
    # Use [^(]+ to match everything up to the opening paren.
    text = re.sub(
        r'runRing(<[^(]+>)?\(tid, (\w+), work\)',
        r'runRing\1(tid, \2, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'runTreeSplit(<[^(]+>)?\(tid, (\w+), work\)',
        r'runTreeSplit\1(tid, \2, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'runTreeUpDown(<[^(]+>)?\(tid, (\w+), work\)',
        r'runTreeUpDown\1(tid, \2, work, ncclShmem, ncclShmemPerWarp)',
        text
    )

    # sendrecv.h: runSend/runRecv call sites: add ncclShmem, ncclShmemPerWarp
    # Template args can have nested <> (e.g. ProtoSimple<1,1,0,8>), use [^(]+ to match
    text = re.sub(
        r'runSend(<[^(]+>)?\(subtid, subtn, group, work\)',
        r'runSend\1(subtid, subtn, group, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'runRecv(<[^(]+>)?\(subtid, subtn, group, work\)',
        r'runRecv\1(subtid, subtn, group, work, ncclShmem, ncclShmemPerWarp)',
        text
    )

    # rcclAllGatherRunRingSimpleProtoImpl call: add params
    text = re.sub(
        r'rcclAllGatherRunRingSimpleProtoImpl\(tid, nthreads, work\)',
        'rcclAllGatherRunRingSimpleProtoImpl(tid, nthreads, work, ncclShmem, ncclShmemPerWarp)',
        text
    )
    text = re.sub(
        r'rcclReduceScatterRunRingSimpleProtoImpl\(tid, nthreads, work\)',
        'rcclReduceScatterRunRingSimpleProtoImpl(tid, nthreads, work, ncclShmem, ncclShmemPerWarp)',
        text
    )

    return text


# ---------------------------------------------------------------------------
# G. IFC / noinline guards
# ---------------------------------------------------------------------------
def apply_ifc_noinline_guards(text):
    # G1. Collective runRing: change IFC guard
    # #if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)
    # ->
    # #ifdef USE_INDIRECT_FUNCTION_CALL
    # and also add __forceinline__ to the IFC branch
    text = re.sub(
        r'#if defined\(USE_INDIRECT_FUNCTION_CALL\) && !defined\(__gfx942__\) && !defined\(__gfx950__\)',
        '#ifdef USE_INDIRECT_FUNCTION_CALL',
        text
    )
    # Some files have partial guards
    text = re.sub(
        r'#if defined\(USE_INDIRECT_FUNCTION_CALL\) && !defined\(__gfx950__\)',
        '#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx942__) && !defined(__gfx950__)',
        text
    )

    # G2. IFC branch: add __forceinline__ to device functions
    # The pattern is: after #ifdef USE_INDIRECT_FUNCTION_CALL, the next line has __device__
    # Add __forceinline__ if not already present
    text = re.sub(
        r'(#ifdef USE_INDIRECT_FUNCTION_CALL\n\s*)__device__\s+void\s+(run(?:Ring|TreeSplit|TreeUpDown))',
        r'\1__device__ __forceinline__ void \2',
        text
    )

    # G3. genericOp: force noinline
    text = re.sub(
        r'__device__ __forceinline__ void genericOp\(',
        '__device__ __attribute__((noinline)) void genericOp(',
        text
    )

    # Remove __forceinline__ from IFC+gfx guard branches for sendrecv
    text = re.sub(
        r'(#if defined\(USE_INDIRECT_FUNCTION_CALL\) && !defined\(__gfx942__\) && !defined\(__gfx950__\)\n\s*)(__device__) __forceinline__',
        r'\1\2',
        text
    )

    return text


# ---------------------------------------------------------------------------
# H. copyToShmem16 overloads
# ---------------------------------------------------------------------------
def apply_copytoshmem16_overloads(text):
    # The original copyToShmem16(tid, void* dst, void const* src, bytes) needs new overloads.
    # Replace the existing function definition with typed versions.
    old_def = (
        '// Copy 16-byte aligned data. You must call with at least `(bytes+15)/16` threads.\n'
        'inline __device__ void copyToShmem16(int tid, void* dst, void const* src, int bytes) {\n'
        '  int offset = 16*tid;\n'
        '  if (offset < bytes) {\n'
        '    ulong2 *src2, *dst2;\n'
        '    src2 = (ulong2*)((char const*)src + offset);\n'
        '    dst2 = (ulong2*)((char*)dst + offset);\n'
        '    dst2->x = src2->x;\n'
        '    dst2->y = src2->y;\n'
        '  }\n'
        '}'
    )
    new_defs = (
        '// Copy 16-byte aligned data from global to LDS. You must call with at least `(bytes+15)/16` threads.\n'
        'inline __device__ void copyToShmem16(int tid, LDSPtr<uint8_t> dst, u8_gptr src, int bytes) {\n'
        '  int offset = 16*tid;\n'
        '  if (offset < bytes) {\n'
        '    u64_gptr src2 = u64_gptr(src + offset);\n'
        '    LDSPtr<uint64_t> dst2 = LDSPtr<uint64_t>(dst + offset);\n'
        '    dst2[0] = src2[0]; dst2[1] = src2[1];\n'
        '  }\n'
        '}\n'
        '\n'
        '// Copy 16-byte aligned data within LDS.\n'
        'inline __device__ void copyToShmem16(int tid, LDSPtr<uint8_t> dst, LDSPtr<uint8_t> src, int bytes) {\n'
        '  int offset = 16*tid;\n'
        '  if (offset < bytes) {\n'
        '    LDSPtr<uint64_t> src2 = LDSPtr<uint64_t>(src + offset);\n'
        '    LDSPtr<uint64_t> dst2 = LDSPtr<uint64_t>(dst + offset);\n'
        '    dst2[0] = src2[0]; dst2[1] = src2[1];\n'
        '  }\n'
        '}\n'
        '\n'
        '// Copy 16-byte aligned data from LDS to global (used by profiling).\n'
        'inline __device__ void copyToShmem16(int tid, u8_gptr dst, LDSPtr<uint8_t> src, int bytes) {\n'
        '  int offset = 16*tid;\n'
        '  if (offset < bytes) {\n'
        '    LDSPtr<uint64_t> src2 = LDSPtr<uint64_t>(src + offset);\n'
        '    u64_gptr dst2 = u64_gptr(dst + offset);\n'
        '    dst2[0] = src2[0]; dst2[1] = src2[1];\n'
        '  }\n'
        '}'
    )
    text = text.replace(old_def, new_defs)

    # Update call sites of copyToShmem16 in common.h (ncclKernelMain)
    # Global -> LDS: copyToShmem16(tid, dst, src, bytes) where dst = &ncclShmem->...
    text = re.sub(
        r'copyToShmem16\(([^,]+), &ncclShmem->(\w+), (ncclShmem->args\.\w+|&\(\([^)]+\)ncclShmem->args\.\w+\)->[^,]+), (\w+)\)',
        r'copyToShmem16(\1, LDSPtr<uint8_t>(&ncclShmem->\2), u8_gptr(\3), \4)',
        text
    )
    # More specific patterns for ncclKernelMain
    text = re.sub(
        r'copyToShmem16\(([^,]+), (void\* dst = )&ncclShmem->(\w+)',
        r'copyToShmem16(\1, LDSPtr<uint8_t>(&ncclShmem->\3)',
        text
    )
    # Replace the "void* dst/src" block patterns
    text = re.sub(
        r'\{ void\* dst = &ncclShmem->comm;\s*\n\s*void\* src = ncclShmem->args\.comm;',
        '{ u8_gptr src = u8_gptr(ncclShmem->args.comm);',
        text
    )
    text = re.sub(
        r'copyToShmem16\(tid, dst, src, bytes\);',
        'copyToShmem16(tid, LDSPtr<uint8_t>(&ncclShmem->comm), src, bytes);',
        text, count=1
    )
    text = re.sub(
        r'\{ // Get address of channel.*\n\s*void\* dst = &ncclShmem->channel;\s*\n\s*void\* src = &\(\(ncclKernelCommAndChannels\*\)ncclShmem->args\.comm\)->channels\[ncclShmem->channelId\];',
        '{ // Get address of channel without incurring indirect load from ncclKernelComm::channels\n      u8_gptr src = u8_gptr(&((ncclKernelCommAndChannels*)ncclShmem->args.comm)->channels[ncclShmem->channelId]);',
        text
    )
    text = re.sub(
        r'copyToShmem16\(tid-WARP_SIZE, dst, src, bytes\);',
        'copyToShmem16(tid-WARP_SIZE, LDSPtr<uint8_t>(&ncclShmem->channel), src, bytes);',
        text, count=1
    )

    # WARP_SPEED copyToShmem16 calls
    text = re.sub(
        r'copyToShmem16\(tid-localWarpId\*WARP_SIZE, dst, src, bytes\);',
        'copyToShmem16(tid-localWarpId*WARP_SIZE, LDSPtr<uint8_t>(&ncclShmem->warpChannel[localWarpId]), src, bytes);',
        text
    )
    # WarpSpeed disabled path: LDS -> LDS copy
    text = re.sub(
        r'(void\* dst = &ncclShmem->warpChannel\[localWarpId\];\s*\n\s*void\* src = &ncclShmem->channel;\s*\n\s*int bytes = sizeof\(ncclDevChannel\);\s*\n\s*)copyToShmem16\(laneId, dst, src, bytes\);',
        r'LDSPtr<uint8_t> dst = LDSPtr<uint8_t>(&ncclShmem->warpChannel[localWarpId]);\n    LDSPtr<uint8_t> src = LDSPtr<uint8_t>(&ncclShmem->channel);\n    int bytes = sizeof(ncclDevChannel);\n    copyToShmem16(laneId, dst, src, bytes);',
        text
    )

    # Profiling: LDS -> Global
    text = re.sub(
        r'copyToShmem16\(tid, ncclShmem->comm\.devProf\+MAXCHANNELS\*ncclShmem->prof\.seq\+blockIdx\.x, &ncclShmem->prof, sizeof\(struct ncclProf\)\)',
        'copyToShmem16(tid, u8_gptr(ncclShmem->comm.devProf+MAXCHANNELS*ncclShmem->prof.seq+blockIdx.x), LDSPtr<uint8_t>(&ncclShmem->prof), sizeof(struct ncclProf))',
        text
    )

    return text


# ---------------------------------------------------------------------------
# I. common.h structural transforms (run BEFORE generic rules)
# ---------------------------------------------------------------------------
def apply_common_h_structural(text, path):
    """Structural transforms for common.h: guards and ncclScratchForWarp."""
    if not path.endswith('common.h'):
        return text

    # I1. Guard #include "device_table.h" — only the kernel TU needs the
    #     dispatch table; per-function TUs skip it.
    text = text.replace(
        '#include "device_table.h"\n',
        '#ifndef RCCL_SPLIT_DEVICE_TU\n#include "device_table.h"\n#endif\n'
    )

    # I2. Remove extern __shared__ declarations entirely.  In the split device
    #     path every function receives LDS pointers as explicit parameters; the
    #     kernel TU (common.cu) defines its own __device__ __shared__ storage.
    text = re.sub(
        r'extern __shared__ ncclShmemData ncclShmem;\n'
        r'#if[^\n]*\n'
        r'  extern __shared__ ulong2 ncclShmemPerWarp\[[^\n]*\];\n'
        r'#else\n'
        r'  extern __shared__ ulong2 ncclShmemPerWarp\[[^\n]*\];\n'
        r'#endif\n',
        '',
        text, count=1
    )

    # I3. Transform ncclScratchForWarp: non-template function using global
    #     ncclShmemPerWarp → template with explicit LDS pointer parameter.
    text = re.sub(
        r'__device__ inline void\* ncclScratchForWarp\(int warp\) \{\n'
        r'  return \(char\*\)ncclShmemPerWarp \+ warp\*ncclShmemScratchWarpSize\(\);\n'
        r'\}',
        'template<typename T>\n'
        '__device__ inline LDSPtr<T> ncclScratchForWarp(ncclShmemPerWarpPtr ncclShmemPerWarp, int warp) {\n'
        '  return (LDSPtr<T>)((LDSPtr<char>)(ncclShmemPerWarp) + warp*ncclShmemScratchWarpSize());\n'
        '}',
        text, count=1
    )

    # I4. Transform ncclKernelMain signature: add LDS pointer params before args.
    text = re.sub(
        r'void ncclKernelMain\(struct ncclDevKernelArgs const\* args\)',
        'void ncclKernelMain(LDSPtr<ncclShmemData> ncclShmem, '
        'ncclShmemPerWarpPtr ncclShmemPerWarp, '
        'struct ncclDevKernelArgs const* args)',
        text
    )

    # I5. Guard ncclKernelMain and kernel forward declarations with
    #     #ifndef RCCL_SPLIT_DEVICE_TU.  Only the kernel TU needs these.
    #     Negative lookahead prevents double-wrapping on re-runs.
    text = re.sub(
        r'\n(?!#ifndef RCCL_SPLIT_DEVICE_TU\n)(template<int SpecializedFnId, typename SpecializedRunWorkBatch, bool COLLTRACE, int COLL_UNROLL>\n'
        r'__device__ __forceinline__ void ncclKernelMain\()',
        r'\n#ifndef RCCL_SPLIT_DEVICE_TU\n\1',
        text, count=1
    )
    # Close the guard after the ENABLE_COLLTRACE #endif that follows the
    # Debug kernel forward declarations.
    text = re.sub(
        r'(#endif)\n\n(#define DEFINE_ncclDevKernel_nop)',
        r'\1\n#endif /* !RCCL_SPLIT_DEVICE_TU */\n\n\2',
        text, count=1
    )

    # I6. Transform DEFINE_ncclDevFunc macros: add LDS pointer params and
    #     forward them to RunWorkBatch::run().
    # IFC variant:
    text = re.sub(
        r'#define DEFINE_ncclDevFunc\(suffix, coll, redop, ty, algo, proto, acc, pipeline, unroll\) \\\n'
        r'  __device__ void ncclDevFunc_##suffix\(\) \{ \\\n'
        r'    RunWorkBatch<coll, ty, redop<ty>, algo, proto, acc, unroll, pipeline>\(\)\.run\(\); \\\n'
        r'  \}',
        '#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto, acc, pipeline, unroll) \\\n'
        '  __device__ void ncclDevFunc_##suffix(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp) { \\\n'
        '    RunWorkBatch<coll, ty, redop<ty>, algo, proto, acc, unroll, pipeline>().run(ncclShmem, ncclShmemPerWarp); \\\n'
        '  }',
        text, count=1
    )
    # Non-IFC variant:
    text = re.sub(
        r'#define DEFINE_ncclDevFunc\(suffix, coll, redop, ty, algo, proto, acc, pipeline, unroll\) \\\n'
        r'  __device__ __attribute__\(\(noinline\)\) void ncclDevFunc_##suffix\(\) \{ \\\n'
        r'    RunWorkBatch<coll, ty, redop<ty>, algo, proto, acc, unroll, pipeline>\(\)\.run\(\); \\\n'
        r'  \}',
        '#define DEFINE_ncclDevFunc(suffix, coll, redop, ty, algo, proto, acc, pipeline, unroll) \\\n'
        '  __device__ __attribute__((noinline)) void ncclDevFunc_##suffix(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp) { \\\n'
        '    RunWorkBatch<coll, ty, redop<ty>, algo, proto, acc, unroll, pipeline>().run(ncclShmem, ncclShmemPerWarp); \\\n'
        '  }',
        text, count=1
    )

    return text


# ---------------------------------------------------------------------------
# J. common.cu transforms (run AFTER generic rules)
# ---------------------------------------------------------------------------
def apply_common_cu_transforms(text, path):
    """Transforms specific to common.cu.cpp: shared mem storage, kernel entries."""
    if not is_common_cu(path):
        return text

    # J1. __shared__ → __device__ __shared__ so the variable has device scope
    #     and is visible across all linked TUs.
    text = text.replace(
        '__shared__ ncclShmemData ncclShmem;',
        '__device__ __shared__ ncclShmemData ncclShmem;'
    )
    text = re.sub(
        r'  __shared__ ulong2 ncclShmemPerWarp\[',
        '  __device__ __shared__ ulong2 ncclShmemPerWarp[',
        text
    )

    # J2. Transform kernel entry points: construct LDS pointers from the
    #     __device__ __shared__ globals and pass them to ncclKernelMain.
    text = re.sub(
        r'(ncclKernelMain<[^>]+>)\(&argsStorage\.args\)',
        r'\1(LDSPtr<ncclShmemData>(&ncclShmem), '
        r'ncclShmemPerWarpPtr((uint8_t*)ncclShmemPerWarp), '
        r'&argsStorage.args)',
        text
    )

    # J3. ncclDevFunc_Nop declarations: add LDS pointer params.
    #     (E12 handles ncclDevFunc_* but Nop may not match if it has no body.)
    text = re.sub(
        r'(__device__\s+(?:__attribute__\(\([^)]*\)\)\s+)?void\s+ncclDevFunc_Nop)\(\s*\);',
        r'\1(LDSPtr<ncclShmemData> ncclShmem, ncclShmemPerWarpPtr ncclShmemPerWarp);',
        text
    )

    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def transform_file(path):
    if not should_process(path):
        return

    with open(path, 'r') as f:
        text = f.read()

    original = text

    # Structural transforms for common.h (guards, ncclScratchForWarp defn)
    text = apply_common_h_structural(text, path)

    # Generic transforms (all device headers and common.cu.cpp)
    text = add_pragma_once(text, path)
    text = add_rccl_ptr_include(text, path)
    text = apply_lds_type_conversions(text)
    text = apply_macro_parameterization(text)
    text = apply_function_signature_additions(text)
    text = apply_primitives_class_changes(text)
    text = apply_call_site_updates(text)
    text = apply_copytoshmem16_overloads(text)
    text = apply_ifc_noinline_guards(text)

    # common.cu-specific transforms (device-scope shared mem, kernel entries)
    text = apply_common_cu_transforms(text, path)

    if text != original:
        with open(path, 'w') as f:
            f.write(text)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <file>", file=sys.stderr)
        sys.exit(1)

    path = sys.argv[1]
    transform_file(path)
