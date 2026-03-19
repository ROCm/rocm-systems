#!/usr/bin/env python3
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Post-hipify transformation: convert LDS-derived pointer types to explicit
# address_space(3) (LDSPtr<T>) so the compiler emits ds_ instructions instead
# of flat_ instructions for shared memory accesses.
#
# This build keeps ncclShmem and ncclShmemPerWarp as __shared__ globals —
# no dot→arrow conversion and no parameter threading needed.  Only pointer
# types derived from LDS (work structures, barriers, scratch) are converted.
#
# Applied to hipified device headers, common.cu.cpp, and generated headers.

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
# Ensure rccl_ptr.h is included (provides LDSPtr, u8_gptr, u64_gptr)
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
    return text


# ---------------------------------------------------------------------------
# A. LDS type conversions  (ncclShmem is still a global, NOT a pointer)
# ---------------------------------------------------------------------------
def apply_lds_type_conversions(text):
    # A4. Barrier members in Primitives classes
    text = text.replace('uint64_t* barriers;', 'LDSPtr<uint64_t> barriers;')
    text = text.replace('uint64_t* barriers_pat;', 'LDSPtr<uint64_t> barriers_pat;')
    # Barrier init in constructors: needs explicit cast from __shared__ address
    text = re.sub(
        r'barriers = &ncclShmem\.groups\[group\]\.barrier;',
        'barriers = LDSPtr<uint64_t>(&ncclShmem.groups[group].barrier);',
        text
    )
    text = re.sub(
        r'barriers_pat = &ncclShmem\.barrier_pat;',
        'barriers_pat = LDSPtr<uint64_t>(&ncclShmem.barrier_pat);',
        text
    )

    # A5. Work pointer types (order: longer names first to avoid partial matches)
    # Two patterns: `struct T*` in casts and `struct T *` in declarations (space before *)
    for typ in ('ncclDevWorkCollReg', 'ncclDevWorkColl', 'ncclDevWorkP2p'):
        # Declarations: `struct T *var` — space before * means we need space after replacement
        text = re.sub(r'struct ' + typ + r' \*', f'LDSPtr<{typ}> ', text)
        # Casts and tight pointers: `struct T*` or `(T*)`
        text = text.replace(f'struct {typ}*', f'LDSPtr<{typ}>')
        text = re.sub(r'\(' + typ + r'\*\)', f'(LDSPtr<{typ}>)', text)

    # A6. void** intermediates pointing into ncclShmem (uses . not ->)
    # Need explicit casts because __shared__ member addresses are generic in source.
    text = re.sub(
        r'void \*\*ptrs = isSendNotRecv \? \(ncclShmem\.groups\[group\]\.dsts \+ Dst\)',
        'LDSPtr<void*> ptrs = isSendNotRecv ? LDSPtr<void*>(ncclShmem.groups[group].dsts + Dst)',
        text
    )
    text = re.sub(
        r': \(ncclShmem\.groups\[group\]\.srcs \+ Src\);',
        ': LDSPtr<void*>(ncclShmem.groups[group].srcs + Src);',
        text
    )
    text = re.sub(
        r'void \*\*ptrs = isSendNotRecv \? ncclShmem\.groups\[group\]\.dsts',
        'LDSPtr<void*> ptrs = isSendNotRecv ? LDSPtr<void*>(ncclShmem.groups[group].dsts)',
        text
    )
    text = re.sub(
        r': ncclShmem\.groups\[group\]\.srcs;',
        ': LDSPtr<void*>(ncclShmem.groups[group].srcs);',
        text
    )
    text = re.sub(
        r'void\*\* srcs = (ncclShmem\.groups\[group\]\.srcs);',
        r'LDSPtr<void*> srcs = LDSPtr<void*>(\1);',
        text
    )
    text = re.sub(
        r'void\*\* dsts = (ncclShmem\.groups\[group\]\.dsts);',
        r'LDSPtr<void*> dsts = LDSPtr<void*>(\1);',
        text
    )

    # A7. ncclPatPeer pointer conversions (recvPeers/sendPeers are scratch LDS)
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
    # Fixups: blanket A5 may have mangled ncclPatPeer lines
    text = text.replace('LDSPtr<ncclDevWorkColl> peer = NULL', 'LDSPtr<ncclPatPeer> peer = nullptr')
    text = text.replace('LDSPtr<ncclDevWorkColl> peer = nullptr', 'LDSPtr<ncclPatPeer> peer = nullptr')

    # A8. parallelFactor polling pointer (uses . not ->)
    text = re.sub(
        r'volatile int\* pfPtr = &shmem->parallelFactor',
        'LDSPtr<volatile int> pfPtr = &shmem->parallelFactor',
        text
    )

    # A9. loadMeta pointer in unpack.h
    text = re.sub(r'loadMeta\*\s*s_meta;', 'LDSPtr<loadMeta> s_meta;', text)
    text = re.sub(r'loadMeta \*s_meta;', 'LDSPtr<loadMeta> s_meta;', text)
    text = re.sub(
        r's_meta = \(loadMeta\*\)\s*ncclScratchForWarp\(',
        's_meta = ncclScratchForWarp<loadMeta>(',
        text
    )

    # A10. shmemCvtPtr -> LDSPtr in unpack.h
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

    # A11a. op128.h: remove shmemCvtPtr function (replaced by #define in rccl_ptr.h)
    text = re.sub(
        r'inline __device__ uint64_t\* shmemCvtPtr\(volatile uint64_t\* shmemGenericPtr\) \{\n'
        r'  return \(uint64_t\*\)shmemGenericPtr;\n'
        r'\}\n',
        '',
        text
    )

    # A11. op128.h: loadShmem128 / storeShmem128 / loadShmemMisaligned128
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
    text = text.replace('*(shmemAsmPtr)', 'ptr[0]')
    text = text.replace('*(shmemAsmPtr+1)', 'ptr[1]')

    # ncclScratchForWarp call sites: convert casts to template calls
    # prims_ll128.h
    text = re.sub(
        r'uint64_t \*shm8 = shmemCvtPtr\(\(uint64_t\*\)ncclScratchForWarp\(([^)]*)\)\);',
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

    # A12. barrier_generic: replace __hip_atomic_* on LDS with plain operations.
    # The barriers pointer is in LDS; atomics are unnecessary — a plain
    # increment and load generate ds_write/ds_read instead of flat_atomic.
    text = text.replace(
        '__hip_atomic_fetch_add((BARRIERS_PTR), 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP)',
        '(*(BARRIERS_PTR) += 1)'
    )
    text = text.replace(
        '__hip_atomic_load((BARRIERS_PTR), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_WORKGROUP)',
        '(*(BARRIERS_PTR))'
    )

    # A13. collTrace / collTraceTail: declare as global pointers so atomics
    # and stores through them generate global_ instructions, not flat_.
    text = text.replace(
        'struct ncclCollTrace* collTrace;',
        'GPtr<struct ncclCollTrace> collTrace;'
    )
    text = text.replace(
        'union ncclCollTraceTail* collTraceTail;',
        'GPtr<union ncclCollTraceTail> collTraceTail;'
    )
    # The assignments from generic pointers need explicit casts
    text = re.sub(
        r'ncclShmem\.collTrace = args->comm->collTrace \+ COLLTRACE_NUM_ITEMS\*ncclShmem\.channelId;',
        'ncclShmem.collTrace = GPtr<struct ncclCollTrace>(args->comm->collTrace + COLLTRACE_NUM_ITEMS*ncclShmem.channelId);',
        text
    )
    text = re.sub(
        r'ncclShmem\.collTraceTail = args->comm->collTraceTail \+ ncclShmem\.channelId;',
        'ncclShmem.collTraceTail = GPtr<union ncclCollTraceTail>(args->comm->collTraceTail + ncclShmem.channelId);',
        text
    )
    # The local collTrace variable in INC_COLL_TRACE must also be GPtr
    text = text.replace(
        'struct ncclCollTrace* collTrace = ncclShmem.collTrace+pos;',
        'GPtr<struct ncclCollTrace> collTrace = ncclShmem.collTrace+pos;'
    )
    # Replace __hip_atomic_fetch_add on collTraceTail with __atomic_fetch_add
    text = text.replace(
        '__hip_atomic_fetch_add(&ncclShmem.collTraceTail->tail, 1, __ATOMIC_SEQ_CST, __HIP_MEMORY_SCOPE_WORKGROUP)',
        '__atomic_fetch_add(&ncclShmem.collTraceTail->tail, 1, __ATOMIC_SEQ_CST)'
    )
    # Replace __hip_atomic_store on collTrace->type with __atomic_store_n
    text = re.sub(
        r'__hip_atomic_store\(&collTrace->type, ([^,]+), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_WORKGROUP\)',
        r'__atomic_store_n(&collTrace->type, \1, __ATOMIC_RELEASE)',
        text
    )

    # loadWorkBatchToShmem: char* dst pointing to ncclShmem.workStorage (LDS)
    text = re.sub(
        r'char\* dst = ncclShmem\.workStorage;',
        'LDSPtr<char> dst = LDSPtr<char>(ncclShmem.workStorage);',
        text
    )
    text = re.sub(
        r'\*\(ulong2\*\)dst = tmp;',
        '*(LDSPtr<ulong2>)dst = tmp;',
        text
    )

    return text


# ---------------------------------------------------------------------------
# ncclScratchForWarp: make it a template returning LDSPtr<T>
# ---------------------------------------------------------------------------
def apply_scratch_for_warp_template(text, path):
    if not path.endswith('common.h'):
        return text
    text = re.sub(
        r'__device__ inline void\* ncclScratchForWarp\(int warp\) \{\n'
        r'  return \(char\*\)ncclShmemPerWarp \+ warp\*ncclShmemScratchWarpSize\(\);\n'
        r'\}',
        'template<typename T>\n'
        '__device__ inline LDSPtr<T> ncclScratchForWarp(int warp) {\n'
        '  return (LDSPtr<T>)((LDSPtr<char>)(ncclShmemPerWarp) + warp*ncclShmemScratchWarpSize());\n'
        '}',
        text, count=1
    )
    return text


# ---------------------------------------------------------------------------
# H. copyToShmem16: replace single void* function with typed overloads
# ---------------------------------------------------------------------------
def apply_copytoshmem16_overloads(text):
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
        '// Copy 16-byte aligned data from global to LDS.\n'
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
        '// Copy 16-byte aligned data from LDS to global (profiling).\n'
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

    # Update call sites in ncclKernelMain (common.h):
    # case 0: global → LDS (comm)
    text = text.replace(
        '{ void* dst = &ncclShmem.comm;\n'
        '      void* src = ncclShmem.args.comm;\n'
        '      int bytes = sizeof(ncclKernelComm);\n'
        '      static_assert(sizeof(ncclKernelComm) <= 16*WARP_SIZE, "ncclKernelComm cannot be loaded by a single warp in one insn.");\n'
        '      copyToShmem16(tid, dst, src, bytes);',
        '{ u8_gptr src = u8_gptr(ncclShmem.args.comm);\n'
        '      int bytes = sizeof(ncclKernelComm);\n'
        '      static_assert(sizeof(ncclKernelComm) <= 16*WARP_SIZE, "ncclKernelComm cannot be loaded by a single warp in one insn.");\n'
        '      copyToShmem16(tid, LDSPtr<uint8_t>(&ncclShmem.comm), src, bytes);'
    )
    # case 1: global → LDS (channel)
    text = text.replace(
        '{ // Get address of channel without incurring indirect load from ncclKernelComm::channels\n'
        '      void* dst = &ncclShmem.channel;\n'
        '      void* src = &((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId];\n'
        '      int bytes = sizeof(ncclDevChannel);\n'
        '      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");\n'
        '      copyToShmem16(tid-WARP_SIZE, dst, src, bytes);',
        '{ // Get address of channel without incurring indirect load from ncclKernelComm::channels\n'
        '      u8_gptr src = u8_gptr(&((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.channelId]);\n'
        '      int bytes = sizeof(ncclDevChannel);\n'
        '      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");\n'
        '      copyToShmem16(tid-WARP_SIZE, LDSPtr<uint8_t>(&ncclShmem.channel), src, bytes);'
    )
    # WARP_SPEED: global → LDS (warpChannel)
    text = re.sub(
        r'(void\* dst = &ncclShmem\.warpChannel\[localWarpId\];\n'
        r'\s*void\* src = &\(\(ncclKernelCommAndChannels\*\)ncclShmem\.args\.comm\)->channels\[ncclShmem\.warpChannelId\[localWarpId\]\];\n'
        r'\s*int bytes = sizeof\(ncclDevChannel\);\n'
        r'\s*static_assert[^;]*;\n)'
        r'(\s*)(//.*)?\n'
        r'(\s*)copyToShmem16\(tid-localWarpId\*WARP_SIZE, dst, src, bytes\);',
        r'u8_gptr src = u8_gptr(&((ncclKernelCommAndChannels*)ncclShmem.args.comm)->channels[ncclShmem.warpChannelId[localWarpId]]);\n'
        r'      int bytes = sizeof(ncclDevChannel);\n'
        r'      static_assert(sizeof(ncclDevChannel) <= 16*WARP_SIZE, "ncclDevChannel cannot be loaded by a single warp in one insn.");\n'
        r'\2\3\n'
        r'\4copyToShmem16(tid-localWarpId*WARP_SIZE, LDSPtr<uint8_t>(&ncclShmem.warpChannel[localWarpId]), src, bytes);',
        text
    )
    # WARP_SPEED disabled path: LDS → LDS copy
    text = re.sub(
        r'void\* dst = &ncclShmem\.warpChannel\[localWarpId\];\n'
        r'(\s*)void\* src = &ncclShmem\.channel;\n'
        r'(\s*)int bytes = sizeof\(ncclDevChannel\);\n'
        r'(\s*)copyToShmem16\(laneId, dst, src, bytes\);',
        r'LDSPtr<uint8_t> dst = LDSPtr<uint8_t>(&ncclShmem.warpChannel[localWarpId]);\n'
        r'\1LDSPtr<uint8_t> src_lds = LDSPtr<uint8_t>(&ncclShmem.channel);\n'
        r'\2int bytes = sizeof(ncclDevChannel);\n'
        r'\3copyToShmem16(laneId, dst, src_lds, bytes);',
        text
    )
    # Profiling: LDS → Global
    text = re.sub(
        r'copyToShmem16\(tid, ncclShmem\.comm\.devProf\+MAXCHANNELS\*ncclShmem\.prof\.seq\+blockIdx\.x, &ncclShmem\.prof, sizeof\(struct ncclProf\)\)',
        'copyToShmem16(tid, u8_gptr(ncclShmem.comm.devProf+MAXCHANNELS*ncclShmem.prof.seq+blockIdx.x), LDSPtr<uint8_t>(&ncclShmem.prof), sizeof(struct ncclProf))',
        text
    )

    return text


# ---------------------------------------------------------------------------
# Add shmemCvtPtr macro to rccl_ptr.h if not already present
# ---------------------------------------------------------------------------
def add_shmemcvtptr_macro(text, path):
    if not path.endswith('rccl_ptr.h'):
        return text
    if 'shmemCvtPtr' not in text:
        text += '\n#define shmemCvtPtr(p) ((LDSPtr<uint64_t>)(p))\n'
    return text


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
TRANSFORM_MARKER = '// RCCL_LDS_TRANSFORM_APPLIED'

def transform_file(path):
    if not should_process(path):
        return False

    with open(path, 'r') as f:
        text = f.read()

    if TRANSFORM_MARKER in text:
        return False  # already transformed

    original = text

    text = add_rccl_ptr_include(text, path)
    text = apply_scratch_for_warp_template(text, path)
    text = apply_lds_type_conversions(text)
    text = apply_copytoshmem16_overloads(text)
    text = add_shmemcvtptr_macro(text, path)

    if text != original:
        text = TRANSFORM_MARKER + '\n' + text
        with open(path, 'w') as f:
            f.write(text)
        return True
    return False


def collect_files(args):
    """Collect files from --dir, --gendir, or bare path arguments."""
    files = []
    i = 0
    while i < len(args):
        if args[i] == '--dir':
            i += 1
            d = args[i]
            for fn in sorted(os.listdir(d)):
                fp = os.path.join(d, fn)
                if os.path.isfile(fp) and should_process(fp):
                    files.append(fp)
        elif args[i] == '--gendir':
            i += 1
            d = args[i]
            for fn in sorted(os.listdir(d)):
                if fn.startswith('device_table') and fn.endswith('.h'):
                    files.append(os.path.join(d, fn))
            cu = os.path.join(d, 'common.cu.cpp')
            if os.path.isfile(cu):
                files.append(cu)
        else:
            files.append(args[i])
        i += 1
    return files


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} [--dir DIR] [--gendir DIR] [file ...]", file=sys.stderr)
        sys.exit(1)

    files = collect_files(sys.argv[1:])
    changed = 0
    for path in files:
        if transform_file(path):
            changed += 1
    print(f"lds_pointer_transform: {changed}/{len(files)} files modified")
