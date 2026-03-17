/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ******************************************************************************
 * Device-only definitions for Backend (create_ctx, destroy_ctx).
 * Used for device bitcode build (--cuda-device-only); host-only backend_bc.cpp
 * is excluded from that build.
 *****************************************************************************/

#include "backend_bc.hpp"
#include "backend_type.hpp"

#if defined(USE_GDA)
#include "gda/backend_gda.hpp"
#endif
#if defined(USE_RO)
#include "reverse_offload/backend_ro.hpp"
#endif
#if defined(USE_IPC)
#include "ipc/backend_ipc.hpp"
#endif

namespace rocshmem {

__device__ bool Backend::create_ctx(int64_t option, rocshmem_ctx_t* ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch (this->type) {
    case BackendType::GDA_BACKEND:
      return static_cast<GDABackend*>(this)->create_ctx(option, ctx);
    case BackendType::RO_BACKEND:
      return static_cast<ROBackend*>(this)->create_ctx(option, ctx);
    case BackendType::IPC_BACKEND:
    default:
      return static_cast<IPCBackend*>(this)->create_ctx(option, ctx);
  }
#elif defined(USE_GDA)
  return static_cast<GDABackend*>(this)->create_ctx(option, ctx);
#elif defined(USE_RO)
  return static_cast<ROBackend*>(this)->create_ctx(option, ctx);
#elif defined(USE_IPC)
  return static_cast<IPCBackend*>(this)->create_ctx(option, ctx);
#endif
}

__device__ void Backend::destroy_ctx(rocshmem_ctx_t* ctx) {
#if defined(USE_GDA) && defined(USE_RO) && defined(USE_IPC)
  switch (this->type) {
    case BackendType::GDA_BACKEND:
      static_cast<GDABackend*>(this)->destroy_ctx(ctx);
      break;
    case BackendType::RO_BACKEND:
      static_cast<ROBackend*>(this)->destroy_ctx(ctx);
      break;
    case BackendType::IPC_BACKEND:
    default:
      static_cast<IPCBackend*>(this)->destroy_ctx(ctx);
      break;
  }
#elif defined(USE_GDA)
  static_cast<GDABackend*>(this)->destroy_ctx(ctx);
#elif defined(USE_RO)
  static_cast<ROBackend*>(this)->destroy_ctx(ctx);
#elif defined(USE_IPC)
  static_cast<IPCBackend*>(this)->destroy_ctx(ctx);
#endif
}

}  // namespace rocshmem
