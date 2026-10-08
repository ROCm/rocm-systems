/*************************************************************************
 * Copyright (c), Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Force-included ahead of each GIN symmetric kernel TU (see DeviceLinker.cmake).
//
// QueuePair device code dispatches on rocshmem::constmem.gda_provider. Every
// symmetric TU is its own code object, so each gets a TU-local copy that the
// rocshmem-gda plugin fills in after QP creation. Defining the include guard
// of rocSHMEM's constmem.hpp keeps its extern declaration out of these TUs.

#ifndef GIN_ROCSHMEM_CONSTMEM_H_
#define GIN_ROCSHMEM_CONSTMEM_H_

// Same includes as the constmem.hpp this replaces, for the headers after it.
#include <array>
#include <cstddef>
#include <cstdint>

#include <hip/hip_runtime.h>

#include "gda/gda_enums.hpp"
#include "rocshmem/rocshmem_common.hpp"
#include "util.hpp"
#include "gin/gin_rocshmem_gda_factory.h"

#define LIBRARY_SRC_CONSTMEM_HPP_

namespace rocshmem {

struct gin_constmem_t {
  GDAProvider gda_provider;
};

static __constant__ gin_constmem_t constmem;

}  // namespace rocshmem

// Host code must reference constmem for hipMemcpyToSymbol to reach this TU's copy.
static int gin_init_tu_constmem(int provider) {
  rocshmem::gin_constmem_t value{static_cast<rocshmem::GDAProvider>(provider)};
  return hipMemcpyToSymbol(HIP_SYMBOL(rocshmem::constmem), &value, sizeof(value)) == hipSuccess ? 0 : -1;
}

[[maybe_unused]] static const bool gin_tu_constmem_registered =
  (rocshmem_gin_register_constmem_init(gin_init_tu_constmem), true);

#endif  // GIN_ROCSHMEM_CONSTMEM_H_
