/*************************************************************************
 * Copyright (c), Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Stub definitions of rocshmem::constmem and rocshmem::logd_constants.
//
// The rocshmem QP device bitcode (queue_pair.bc) references these as
// extern __constant__ symbols.  In a full rocshmem build they are defined
// in rocshmem_gpu.cpp, but the GIN build only links minimal QP bitcode
// into the symmetric fat-object kernels.
//
// These zero-initialized stubs satisfy the device linker.  They are never
// read at runtime — only GIN SDMA runs on symmetric kernels today, and
// SDMA does not touch constmem. GIN GDA for internal rccl routines still
// need enablement.

#include <hip/hip_runtime.h>
#include "gda/gda_enums.hpp"
#include "rocshmem/rocshmem_common.hpp"

namespace rocshmem {

struct constmem_t {
  BackendType backend_type;
  GDAProvider gda_provider;
  uint64_t alltoall_wg_algo;
  int my_pe;
  int num_pes;
  int ipc_first_pe;
  int ipc_stride;
  int ipc_shm_size;
} __attribute__((aligned(16)));

struct logd_constants {
  int pe_number;
  int flags;
} __attribute__((aligned(16)));

__constant__ constmem_t constmem;
__constant__ struct logd_constants logd_constants;

}  // namespace rocshmem
