# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Post-hipify transformation: convert ncclShmem references to LDS pointers.
#
# This enables the compiler to emit ds_ (Local Data Share) instructions instead
# of flat_ instructions for shared memory accesses, avoiding the ~40us latency
# penalty from flat address resolution in the split-compiled device functions.
#
# Applied to device header files only (matching add_unroll.sh convention).

HIP_FILE=$1

if [[ "$HIP_FILE" =~ .*/src/device/.*\.h ]]; then

  # === Core parameter/member conversions ===

  # ncclShmemData reference → LDSPtr<ncclShmemData>
  sed -i 's/struct ncclShmemData& ncclShmem/LDSPtr<ncclShmemData> ncclShmem/g' "$HIP_FILE"
  sed -i 's/ncclShmemData& ncclShmem/LDSPtr<ncclShmemData> ncclShmem/g' "$HIP_FILE"
  sed -i 's/struct ncclShmemData& shmem/LDSPtr<ncclShmemData> shmem/g' "$HIP_FILE"
  sed -i 's/ncclShmemData& shmem/LDSPtr<ncclShmemData> shmem/g' "$HIP_FILE"

  # void* ncclShmemPerWarp → ncclShmemPerWarpPtr ncclShmemPerWarp
  # (also matches ncclShmemPerWarp_ due to prefix match)
  sed -i 's/void\* ncclShmemPerWarp/ncclShmemPerWarpPtr ncclShmemPerWarp/g' "$HIP_FILE"

  # === Member access: dot → arrow ===

  sed -i 's/ncclShmem\./ncclShmem->/g' "$HIP_FILE"
  sed -i 's/ncclShmem)\./ncclShmem)->/g' "$HIP_FILE"
  # Constructor parameter named 'shmem' (prims_simple.h, prims_ll.h, prims_ll128.h)
  sed -i 's/shmem\./shmem->/g' "$HIP_FILE"

  # === Barrier members (LDS pointers to shared memory barrier counters) ===

  sed -i 's/uint64_t\* barriers;/LDSPtr<uint64_t> barriers;/g' "$HIP_FILE"
  sed -i 's/uint64_t\* barriers_pat;/LDSPtr<uint64_t> barriers_pat;/g' "$HIP_FILE"

  # === ncclScratchForWarp conversions ===

  # prims_ll128.h: shmemCvtPtr wrapping ncclScratchForWarp
  sed -i 's/uint64_t \*shm8 = shmemCvtPtr((uint64_t\*)ncclScratchForWarp(\([^;]*\)));/LDSPtr<uint64_t> shm8 = ncclScratchForWarp<uint64_t>(\1);/g' "$HIP_FILE"

  # prims_ll128.h: T pointer from ncclScratchForWarp
  sed -i 's/T \*shm = (T\*)ncclScratchForWarp(/LDSPtr<T> shm = ncclScratchForWarp<T>(/g' "$HIP_FILE"

  # prims_ll128.h: T pointer derived from shm8 (LDSPtr<uint64_t>)
  sed -i 's/T \*shm = (T\*)shm8/LDSPtr<T> shm = LDSPtr<T>(shm8)/g' "$HIP_FILE"

  # sendrecv.h: Shared pointer from ncclScratchForWarp
  sed -i 's/Shared\* shared = (Shared\*)ncclScratchForWarp(/LDSPtr<Shared> shared = ncclScratchForWarp<Shared>(/g' "$HIP_FILE"

  # unpack.h: loadMeta pointer declaration and ncclScratchForWarp
  sed -i 's/loadMeta\* s_meta;/LDSPtr<loadMeta> s_meta;/g' "$HIP_FILE"
  sed -i 's/loadMeta \*s_meta;/LDSPtr<loadMeta> s_meta;/g' "$HIP_FILE"
  sed -i 's/s_meta = (loadMeta\*) *ncclScratchForWarp(/s_meta = ncclScratchForWarp<loadMeta>(/g' "$HIP_FILE"

  # all_gather.h, reduce_scatter.h: ncclPatShmem from ncclScratchForWarp
  sed -i 's/struct ncclPatShmem\* shmem = (struct ncclPatShmem\*)ncclScratchForWarp(/LDSPtr<ncclPatShmem> shmem = ncclScratchForWarp<ncclPatShmem>(/g' "$HIP_FILE"

  # prims_simple.h: patReduce/patCopy parameter type
  sed -i 's/struct ncclPatShmem\* shmem)/LDSPtr<ncclPatShmem> shmem)/g' "$HIP_FILE"

  # === Force noinline for gfx942/gfx950 (matching develop branch) ===

  # Collective headers: #ifdef USE_INDIRECT_FUNCTION_CALL + __forceinline__
  # → #if defined(...) && !defined(__gfx942__) && !defined(__gfx950__) + remove __forceinline__
  sed -i '/#ifdef USE_INDIRECT_FUNCTION_CALL/{N;s/#ifdef USE_INDIRECT_FUNCTION_CALL\n\([[:space:]]*__device__\) __forceinline__/#if defined(USE_INDIRECT_FUNCTION_CALL) \&\& !defined(__gfx942__) \&\& !defined(__gfx950__)\n\1/;}' "$HIP_FILE"

  # sendrecv.h: add !defined(__gfx942__) to existing partial guard
  sed -i 's/#if defined(USE_INDIRECT_FUNCTION_CALL) && !defined(__gfx950__)/#if defined(USE_INDIRECT_FUNCTION_CALL) \&\& !defined(__gfx942__) \&\& !defined(__gfx950__)/g' "$HIP_FILE"

  # sendrecv.h: remove __forceinline__ from IFC branch after the guard
  sed -i '/#if defined(USE_INDIRECT_FUNCTION_CALL).*!defined(__gfx950__)/{N;s/\n\([[:space:]]*__device__\) __forceinline__/\n\1/;}' "$HIP_FILE"

  # === shmemCvtPtr → LDSPtr<uint64_t> (unpack.h) ===
  # Must match full expression to handle paren balancing

  sed -i 's/shmemCvtPtr((uint64_t \*)(s_meta + (w \* PPW + t)))/LDSPtr<uint64_t>(s_meta + (w * PPW + t))/g' "$HIP_FILE"
  sed -i 's/shmemCvtPtr((uint64_t\*) (s_meta + meta_idx))/LDSPtr<uint64_t>(s_meta + meta_idx)/g' "$HIP_FILE"

fi
