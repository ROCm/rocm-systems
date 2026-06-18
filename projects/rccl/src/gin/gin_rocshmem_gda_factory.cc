/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

/**
 * @file gin_rocshmem_gda_factory.cc
 *
 * Standalone GIN QP factory — creates rocshmem QueuePairs for RCCL's GIN
 * backend without requiring rocshmem_init() or a GDABackend instance.
 *
 * Handles the full QP lifecycle:
 *   NIC discovery → DV library loading → IB device open → PD creation →
 *   CQ/QP creation → dest_info exchange → state transitions → GPU QP init
 *
 * Provider-specific GPU QP initialization (doorbell mapping, CQ/SQ buffer
 * setup) is adapted from GDABackend's *_initialize_gpu_qp() methods.
 * The QueuePair class itself and its device-side methods are reused as-is
 * from their original location.
 */

#include "gin/gin_rocshmem_gda_factory.h"
#include <gda/queue_pair.hpp>
#include <gda/ibv_wrapper.hpp>
#include <gda/backend_gda.hpp>
#include <gda/debug_gda.hpp>
#include <envvar.hpp>
#include <gda/topology.hpp>
#include <util.hpp>
#include <constants.hpp>
#include <rocshmem/rocshmem_common.hpp>

#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#include <cstring>
#include <cstdlib>
#include <map>
#include <unistd.h>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <cassert>

// Provider headers for types and structures
#if defined(GDA_IONIC)
#include <gda/ionic/provider_gda_ionic.hpp>
#include <gda/ionic/ionic_dv.h>
#endif
#if defined(GDA_BNXT)
#include <gda/bnxt/provider_gda_bnxt.hpp>
#include <gda/bnxt/bnxt_re_dv.h>
#endif
#if defined(GDA_MLX5)
#include <gda/mlx5/provider_gda_mlx5.hpp>
#endif

using namespace rocshmem;

// Forward declarations for RCCL's IB verbs wrappers (from ibvwrap.h).
// We declare them here instead of including ibvwrap.h to avoid IB type
// clashes between RCCL's ibvcore.h and rocshmem's ibv_core.hpp.
// ncclResult_t is int.
int wrap_ibv_reg_mr_iova2(struct ibv_mr **ret, struct ibv_pd *pd, void *addr,
                          size_t length, uint64_t iova, int access);
int wrap_ibv_reg_dmabuf_mr(struct ibv_mr **ret, struct ibv_pd *pd,
                           uint64_t offset, size_t length, uint64_t iova,
                           int fd, int access);
int wrap_ibv_dereg_mr(struct ibv_mr *mr);

///////////////////////////////////////////////////////////////////////////////
// Internal types
///////////////////////////////////////////////////////////////////////////////

struct GinNicDevice {
  std::string nic_name;
  struct ibv_device *device = nullptr;
  struct ibv_context *context = nullptr;
  struct ibv_device_attr device_attr {};
  struct ibv_pd *pd_orig = nullptr;
  struct ibv_port_attr portinfo {};
  union ibv_gid gid {};
  int port = 1;
  int gid_index = 0;
  uint32_t gid_type = 0;

  // Parent domains (IONIC, MLX5)
  struct ibv_pd *pd_parent = nullptr;
  // IONIC UXDMA domains
  struct ibv_pd *pd_uxdma[2] = {nullptr, nullptr};
};

struct gin_dest_info {
  int lid;
  int qpn;
  int psn;
  union ibv_gid gid;
};

struct rocshmem_gin_qp_set {
  // GPU QP initialization — must be a member to access QueuePair private fields via friend
  int initialize_gpu_qp(QueuePair *gpu_qp, int idx);

  int nRanks;
  int myRank;
  int provider;  // GDAProvider enum

  // NIC state (owned by this set)
  GinNicDevice nic;

  // DV library handle
  void *dv_handle = nullptr;
#if defined(GDA_IONIC)
  ionicdv_funcs_t ionic_dv {};
#endif
#if defined(GDA_BNXT)
  bnxtdv_funcs_t bnxt_re_dv {};
  std::vector<struct bnxt_host_qp> bnxt_qps;
  std::vector<struct bnxt_host_cq> bnxt_scqs;
  std::vector<struct bnxt_host_cq> bnxt_rcqs;
  HIPAllocator *qp_allocator = nullptr;
#endif
#if defined(GDA_MLX5)
  mlx5dv_funcs_t mlx5dv {};
  std::vector<mlx5_devx_qp> mlx5_qps;
#endif

  // IBV QP/CQ arrays
  std::vector<struct ibv_qp*> ibv_qps;
  std::vector<struct ibv_cq*> ibv_cqs;

  // QueuePair objects
  QueuePair *host_qps = nullptr;
  QueuePair *gpu_qps = nullptr;

  uint32_t inline_threshold = 8;
  uint32_t sq_size = envvar::gda::sq_size;
};

///////////////////////////////////////////////////////////////////////////////
// Static helpers: PD allocators for parent domains
///////////////////////////////////////////////////////////////////////////////

static void* gin_pd_alloc_device_uncached(struct ibv_pd*, void*, size_t size,
                                           size_t, uint64_t) {
  void* ptr = nullptr;
  if (hipExtMallocWithFlags(&ptr, size, hipDeviceMallocUncached) != hipSuccess)
    return nullptr;
  (void)hipMemset(ptr, 0, size);
  (void)hipStreamSynchronize(0);
  return ptr;
}

static void* gin_pd_alloc_host(struct ibv_pd*, void*, size_t size,
                                size_t, uint64_t) {
  void* ptr = nullptr;
  if (hipHostMalloc(&ptr, size, hipHostMallocDefault) != hipSuccess)
    return nullptr;
  memset(ptr, 0, size);
  return ptr;
}

static void gin_pd_release(struct ibv_pd*, void*, void* ptr, uint64_t) {
  (void)hipFree(ptr);
}

///////////////////////////////////////////////////////////////////////////////
// Standalone rocm_memory_lock_to_fine_grain — does not depend on rocshmem_init()
///////////////////////////////////////////////////////////////////////////////

static int gin_memory_lock_to_fine_grain(void* ptr, size_t size, void** gpu_ptr, int gpu_id) {
  // Get the GPU agent for gpu_id
  hsa_agent_t gpu_agent{};
  int gpu_count = 0;
  auto gpu_cb = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_GPU) {
      auto* p = static_cast<std::pair<int, std::pair<int*, hsa_agent_t*>>*>(data);
      if (*(p->second.first) == p->first) {
        *(p->second.second) = agent;
      }
      (*(p->second.first))++;
    }
    return HSA_STATUS_SUCCESS;
  };
  auto gpu_data = std::make_pair(gpu_id, std::make_pair(&gpu_count, &gpu_agent));
  hsa_iterate_agents(gpu_cb, &gpu_data);

  // Get a CPU fine-grained+kernarg pool
  hsa_amd_memory_pool_t cpu_pool{};
  auto cpu_cb = [](hsa_agent_t agent, void* data) -> hsa_status_t {
    hsa_device_type_t type;
    hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &type);
    if (type == HSA_DEVICE_TYPE_CPU) {
      auto pool_cb = [](hsa_amd_memory_pool_t pool, void* pdata) -> hsa_status_t {
        hsa_amd_memory_pool_global_flag_t flags;
        hsa_amd_memory_pool_get_info(pool, HSA_AMD_MEMORY_POOL_INFO_GLOBAL_FLAGS, &flags);
        if (flags == (HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_KERNARG_INIT |
                      HSA_AMD_MEMORY_POOL_GLOBAL_FLAG_FINE_GRAINED)) {
          *static_cast<hsa_amd_memory_pool_t*>(pdata) = pool;
        }
        return HSA_STATUS_SUCCESS;
      };
      hsa_amd_agent_iterate_memory_pools(agent, pool_cb, data);
    }
    return HSA_STATUS_SUCCESS;
  };
  hsa_iterate_agents(cpu_cb, &cpu_pool);

  hsa_status_t status = hsa_amd_memory_lock_to_pool(
      ptr, size, &gpu_agent, 1, cpu_pool, 0, gpu_ptr);
  if (status != HSA_STATUS_SUCCESS) {
    LOG_ERROR("gin_memory_lock_to_fine_grain failed: 0x%x", status);
    return -1;
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// NIC discovery and IB device open
///////////////////////////////////////////////////////////////////////////////

static int gin_detect_provider(struct ibv_device_attr *attr) {
#if defined(GDA_BNXT)
  if (attr->vendor_id == GDA_BNXT_VENDOR_ID) return GDAProvider::BNXT;
#endif
#if defined(GDA_IONIC)
  if (attr->vendor_id == GDA_IONIC_VENDOR_ID) return GDAProvider::IONIC;
#endif
#if defined(GDA_MLX5)
  if (attr->vendor_id == GDA_MLX5_VENDOR_ID) return GDAProvider::MLX5;
#endif
  return -1;
}

static int gin_open_ib_device(rocshmem_gin_qp_set *set) {
  struct ibv_device **dev_list = nullptr;
  int ndev = 0;

  dev_list = ibv.get_device_list(&ndev);
  if (!dev_list || ndev == 0) {
    LOG_ERROR("GIN QP factory: no IB devices found (ibverbs get_device_list returned %d)", ndev);
    return -1;
  }

  // Use topology-aware NIC selection (same as GDABackend::select_nics)
  int gpu_dev = 0;
  (void)hipGetDevice(&gpu_dev);
  std::string closest_nic;
  if (GetClosestNicToGpu(gpu_dev, nullptr, &closest_nic) >= 0) {
    LOG_TRACE("GIN QP factory: GPU %d closest NIC: %s", gpu_dev, closest_nic.c_str());
  }

  // Find the selected device (or first active if topology unavailable)
  for (int d = 0; d < ndev; d++) {
    // If topology selected a NIC, skip non-matching devices
    if (!closest_nic.empty() &&
        closest_nic != ibv.get_device_name(dev_list[d])) {
      continue;
    }
    struct ibv_context *ctx = nullptr;

#if defined(GDA_MLX5)
    // Try DevX open first for MLX5
    if (set->mlx5dv.open_device) {
      struct mlx5dv_context_attr mlx5_attr = {};
      mlx5_attr.flags = MLX5DV_CONTEXT_FLAGS_DEVX;
      ctx = set->mlx5dv.open_device(dev_list[d], &mlx5_attr);
    }
#endif
    if (!ctx) {
      ctx = ibv.open_device(dev_list[d]);
    }
    if (!ctx) continue;

    struct ibv_device_attr dev_attr;
    if (ibv.query_device(ctx, &dev_attr) != 0) {
      ibv.close_device(ctx);
      continue;
    }

    int provider = gin_detect_provider(&dev_attr);
    if (provider < 0) {
      LOG_WARN("GIN QP factory: device %s vendor_id=0x%x not recognized as IONIC/BNXT/MLX5, skipping",
               ibv.get_device_name(dev_list[d]), dev_attr.vendor_id);
      ibv.close_device(ctx);
      continue;
    }

    // Check for active port
    struct ibv_port_attr port_attr;
    int port = 1;
    if (ibv.query_port(ctx, port, &port_attr) != 0 ||
        port_attr.state != IBV_PORT_ACTIVE) {
      LOG_WARN("GIN QP factory: device %s port %d not active (state=%d), skipping",
               ibv.get_device_name(dev_list[d]), port, (int)port_attr.state);
      ibv.close_device(ctx);
      continue;
    }

    // Found a usable device
    set->nic.device = dev_list[d];
    set->nic.context = ctx;
    set->nic.device_attr = dev_attr;
    set->nic.portinfo = port_attr;
    set->nic.port = port;
    set->nic.nic_name = ibv.get_device_name(dev_list[d]);
    set->provider = provider;

    // Select GID — matches GDABackend::select_gid_index() logic:
    //   1. IB type → use immediately (non-Ethernet)
    //   2. Skip link-local (FE80 prefix) and all-zero
    //   3. Prefer RoCEv2 over RoCEv1
    {
      const uint8_t local_gid_prefix[2] = {0xFE, 0x80};
      int gid_tbl_len = port_attr.gid_tbl_len;
      struct ibv_gid_entry *gid_entries = (struct ibv_gid_entry*)calloc(gid_tbl_len, sizeof(struct ibv_gid_entry));
      ssize_t n = ibv.query_gid_table(ctx, gid_entries, gid_tbl_len, 0);
      int selected_idx = -1;
      uint32_t selected_type = IBV_GID_TYPE_ROCE_V1;

      for (int i = 0; i < n; i++) {
        union ibv_gid *g = &gid_entries[i].gid;
        uint32_t gtype = gid_entries[i].gid_type;

        // IB mode: use immediately
        if (gtype == IBV_GID_TYPE_IB) {
          selected_idx = i;
          selected_type = gtype;
          set->nic.gid = *g;
          set->nic.gid_index = i;
          break;
        }

        // Skip link-local
        if (memcmp(g->raw, local_gid_prefix, 2) == 0) continue;
        // Skip all-zero
        bool all_zero = true;
        for (int b = 0; b < 16; b++) { if (g->raw[b] != 0) { all_zero = false; break; } }
        if (all_zero) continue;

        // First valid, or prefer higher type (RoCEv2 > RoCEv1)
        if (selected_idx == -1 || gtype > selected_type) {
          selected_idx = i;
          selected_type = gtype;
          set->nic.gid = *g;
          set->nic.gid_index = i;
        }
      }
      set->nic.gid_type = selected_type;
      free(gid_entries);
    }

    // Allocate PD
    set->nic.pd_orig = ibv.alloc_pd(ctx);
    if (!set->nic.pd_orig) {
      ibv.close_device(ctx);
      continue;
    }

    LOG_INFO("GIN QP factory: GPU %d using %s port %d (provider=%d) "
             "gid_index=%d gid=%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x:%02x%02x",
             gpu_dev, set->nic.nic_name.c_str(), set->nic.port, set->provider,
             set->nic.gid_index,
             set->nic.gid.raw[0],  set->nic.gid.raw[1],  set->nic.gid.raw[2],  set->nic.gid.raw[3],
             set->nic.gid.raw[4],  set->nic.gid.raw[5],  set->nic.gid.raw[6],  set->nic.gid.raw[7],
             set->nic.gid.raw[8],  set->nic.gid.raw[9],  set->nic.gid.raw[10], set->nic.gid.raw[11],
             set->nic.gid.raw[12], set->nic.gid.raw[13], set->nic.gid.raw[14], set->nic.gid.raw[15]);
    ibv.free_device_list(dev_list);
    return 0;
  }

  ibv.free_device_list(dev_list);
  LOG_ERROR("GIN QP factory: no usable IB device found with an active port");
  return -1;
}

///////////////////////////////////////////////////////////////////////////////
// DV library loading (extracted from GDABackend::*_dv_dl_init)
///////////////////////////////////////////////////////////////////////////////

static int gin_load_dv_library(rocshmem_gin_qp_set *set) {
  switch (set->provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC: {
    set->dv_handle = dlopen("libionic.so", RTLD_LAZY);
    if (!set->dv_handle)
      set->dv_handle = dlopen("/usr/local/lib/libionic.so", RTLD_LAZY);
    if (!set->dv_handle) {
      LOG_ERROR("GIN QP factory: failed to load libionic.so: %s", dlerror());
      return -1;
    }

    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, get_ctx);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, qp_get_udma_idx);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, get_cq);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, get_qp);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, pd_set_sqcmb);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, pd_set_rqcmb);
    DLSYM_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, pd_set_udma_mask);
    DLSYM_OPT_HELPER(set->ionic_dv, ionic_dv_, set->dv_handle, create_cq_ex);
    return 0;
  }
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT: {
    set->dv_handle = dlopen("libbnxt_re.so", RTLD_LAZY);
    if (!set->dv_handle)
      set->dv_handle = dlopen("/usr/local/lib/libbnxt_re.so", RTLD_LAZY);
    if (!set->dv_handle) {
      LOG_ERROR("GIN QP factory: failed to load libbnxt_re.so: %s", dlerror());
      return -1;
    }

    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, init_obj);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, create_qp);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, destroy_qp);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, modify_qp);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, qp_mem_alloc);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, create_cq);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, destroy_cq);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, cq_mem_alloc);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, umem_reg);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, umem_dereg);
    DLSYM_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, alloc_db_region);
    DLSYM_OPT_HELPER(set->bnxt_re_dv, bnxt_re_dv_, set->dv_handle, free_db_region);
    return 0;
  }
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5: {
    set->dv_handle = dlopen("libmlx5.so", RTLD_LAZY);
    if (!set->dv_handle) {
      LOG_ERROR("GIN QP factory: failed to load libmlx5.so: %s", dlerror());
      return -1;
    }

    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, init_obj);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, open_device);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_obj_create);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_obj_modify);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_obj_destroy);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_umem_reg_ex);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_umem_dereg);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_alloc_uar);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_free_uar);
    DLSYM_HELPER(set->mlx5dv, mlx5dv_, set->dv_handle, devx_query_eqn);
    return 0;
  }
#endif
  default:
    return -1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Parent domain creation (for IONIC and MLX5)
///////////////////////////////////////////////////////////////////////////////

static int gin_create_parent_domain(rocshmem_gin_qp_set *set) {
  struct ibv_parent_domain_init_attr pattr;

  memset(&pattr, 0, sizeof(pattr));
  pattr.pd = set->nic.pd_orig;
  pattr.td = nullptr;
  pattr.comp_mask = IBV_PARENT_DOMAIN_INIT_ATTR_ALLOCATORS;
  pattr.free = gin_pd_release;
  pattr.pd_context = nullptr;

#if defined(GDA_IONIC)
  if (set->provider == GDAProvider::IONIC) {
    pattr.alloc = gin_pd_alloc_device_uncached;
    set->nic.pd_parent = ibv.alloc_parent_domain(set->nic.context, &pattr);
    if (!set->nic.pd_parent) return -1;

    set->ionic_dv.pd_set_sqcmb(set->nic.pd_parent, false, false, false);
    set->ionic_dv.pd_set_rqcmb(set->nic.pd_parent, false, false, false);

    for (int u = 0; u < 2; u++) {
      set->nic.pd_uxdma[u] = ibv.alloc_parent_domain(set->nic.context, &pattr);
      if (!set->nic.pd_uxdma[u]) return -1;
      set->ionic_dv.pd_set_sqcmb(set->nic.pd_uxdma[u], false, false, false);
      set->ionic_dv.pd_set_rqcmb(set->nic.pd_uxdma[u], false, false, false);
      set->ionic_dv.pd_set_udma_mask(set->nic.pd_uxdma[u], 1u << u);
    }
    return 0;
  }
#endif

#if defined(GDA_MLX5)
  if (set->provider == GDAProvider::MLX5) {
    pattr.alloc = gin_pd_alloc_host;
    set->nic.pd_parent = ibv.alloc_parent_domain(set->nic.context, &pattr);
    if (!set->nic.pd_parent) return -1;
    return 0;
  }
#endif

  return 0;  // BNXT doesn't use parent domains
}

///////////////////////////////////////////////////////////////////////////////
// CQ creation (per provider)
///////////////////////////////////////////////////////////////////////////////

static int gin_create_cqs(rocshmem_gin_qp_set *set) {
  int n = set->nRanks;
  set->ibv_cqs.resize(n, nullptr);

  switch (set->provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC: {
    struct ibv_cq_init_attr_ex cq_attr;
    struct ionic_cq_init_attr_ex ionic_cq_attr;
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.cqe = set->sq_size << 1;
    cq_attr.comp_mask = IBV_CQ_INIT_ATTR_MASK_PD;

    memset(&ionic_cq_attr, 0, sizeof(ionic_cq_attr));
    if (set->ionic_dv.create_cq_ex) {
      ionic_cq_attr.comp_mask = IONIC_CQ_INIT_ATTR_MASK_FLAGS;
      ionic_cq_attr.flags = IONIC_CQ_INIT_ATTR_CCQE;
    }

    for (int i = 0; i < n; i++) {
      cq_attr.parent_domain = set->nic.pd_uxdma[i & 1];
      struct ibv_cq_ex *cq_ex = nullptr;
      if (set->ionic_dv.create_cq_ex)
        cq_ex = set->ionic_dv.create_cq_ex(set->nic.context, &cq_attr, &ionic_cq_attr);
      if (!cq_ex)
        cq_ex = ibv_create_cq_ex(set->nic.context, &cq_attr);
      if (!cq_ex) return -1;
      set->ibv_cqs[i] = ibv.cq_ex_to_cq(cq_ex);
      if (!set->ibv_cqs[i]) return -1;
    }
    return 0;
  }
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT: {
    set->bnxt_scqs.resize(n);
    set->bnxt_rcqs.resize(n);
    if (!set->qp_allocator)
      set->qp_allocator = new HIPAllocatorFinegrained();

    int cqe = 1;  // CQE compression: only need length 1
    int dmabuf_enabled = ibv.is_dmabuf_supported();

    for (int i = 0; i < n; i++) {
      auto *ctx = set->nic.context;

      // SCQ
      struct bnxt_re_dv_cq_attr cq_attr;
      memset(&cq_attr, 0, sizeof(cq_attr));
      set->bnxt_scqs[i].handle = set->bnxt_re_dv.cq_mem_alloc(ctx, cqe, &cq_attr);
      if (!set->bnxt_scqs[i].handle) return -1;
      cq_attr.ncqe = cqe;

      set->bnxt_scqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
      set->bnxt_scqs[i].depth = cq_attr.ncqe;
      set->qp_allocator->allocate(reinterpret_cast<void**>(&set->bnxt_scqs[i].buf),
                                   set->bnxt_scqs[i].length);
      (void)hipMemset(set->bnxt_scqs[i].buf, 0, set->bnxt_scqs[i].length);

      if (dmabuf_enabled) {
        if (set->qp_allocator->GetDmabufHandle(set->bnxt_scqs[i].buf,
                                               set->bnxt_scqs[i].length,
                                               &set->bnxt_scqs[i].dmabuf_fd,
                                               &set->bnxt_scqs[i].dmabuf_offset) != hipSuccess)
          return -1;
      }

      struct bnxt_re_dv_umem_reg_attr umem_attr;
      memset(&umem_attr, 0, sizeof(umem_attr));
      umem_attr.addr = set->bnxt_scqs[i].buf;
      umem_attr.size = set->bnxt_scqs[i].length;
      umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
      umem_attr.dmabuf_fd = dmabuf_enabled ? set->bnxt_scqs[i].dmabuf_fd : 0;
      set->bnxt_scqs[i].umem_handle = set->bnxt_re_dv.umem_reg(ctx, &umem_attr);
      if (!set->bnxt_scqs[i].umem_handle) return -1;

      struct bnxt_re_dv_cq_init_attr cq_init;
      memset(&cq_init, 0, sizeof(cq_init));
      cq_init.cq_handle = (uint64_t)set->bnxt_scqs[i].handle;
      cq_init.umem_handle = set->bnxt_scqs[i].umem_handle;
      cq_init.ncqe = cq_attr.ncqe;
      set->bnxt_scqs[i].cq = set->bnxt_re_dv.create_cq(ctx, &cq_init);
      if (!set->bnxt_scqs[i].cq) return -1;

      // RCQ
      memset(&cq_attr, 0, sizeof(cq_attr));
      set->bnxt_rcqs[i].handle = set->bnxt_re_dv.cq_mem_alloc(ctx, cqe, &cq_attr);
      if (!set->bnxt_rcqs[i].handle) return -1;

      set->bnxt_rcqs[i].length = cq_attr.ncqe * cq_attr.cqe_size;
      set->bnxt_rcqs[i].depth = cq_attr.ncqe;
      set->qp_allocator->allocate(reinterpret_cast<void**>(&set->bnxt_rcqs[i].buf),
                                   set->bnxt_rcqs[i].length);
      (void)hipMemset(set->bnxt_rcqs[i].buf, 0, set->bnxt_rcqs[i].length);

      if (dmabuf_enabled) {
        if (set->qp_allocator->GetDmabufHandle(set->bnxt_rcqs[i].buf,
                                               set->bnxt_rcqs[i].length,
                                               &set->bnxt_rcqs[i].dmabuf_fd,
                                               &set->bnxt_rcqs[i].dmabuf_offset) != hipSuccess)
          return -1;
      }

      memset(&umem_attr, 0, sizeof(umem_attr));
      umem_attr.addr = set->bnxt_rcqs[i].buf;
      umem_attr.size = set->bnxt_rcqs[i].length;
      umem_attr.access_flags = IBV_ACCESS_LOCAL_WRITE;
      umem_attr.dmabuf_fd = dmabuf_enabled ? set->bnxt_rcqs[i].dmabuf_fd : 0;
      set->bnxt_rcqs[i].umem_handle = set->bnxt_re_dv.umem_reg(ctx, &umem_attr);
      if (!set->bnxt_rcqs[i].umem_handle) return -1;

      memset(&cq_init, 0, sizeof(cq_init));
      cq_init.cq_handle = (uint64_t)set->bnxt_rcqs[i].handle;
      cq_init.umem_handle = set->bnxt_rcqs[i].umem_handle;
      cq_init.ncqe = cq_attr.ncqe;
      set->bnxt_rcqs[i].cq = set->bnxt_re_dv.create_cq(ctx, &cq_init);
      if (!set->bnxt_rcqs[i].cq) return -1;

      set->ibv_cqs[i] = set->bnxt_scqs[i].cq;
    }
    return 0;
  }
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5: {
    // MLX5 creates CQs inside mlx5dv.create_qp(), handled in gin_create_qps
    return 0;
  }
#endif
  default:
    return -1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// QP creation (per provider)
///////////////////////////////////////////////////////////////////////////////

static int gin_create_qps(rocshmem_gin_qp_set *set) {
  int n = set->nRanks;
  set->ibv_qps.resize(n, nullptr);

  switch (set->provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC: {
    struct ibv_qp_init_attr_ex attr;
    memset(&attr, 0, sizeof(attr));
    attr.cap.max_send_wr = set->sq_size;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    attr.cap.max_inline_data = set->inline_threshold;
    attr.sq_sig_all = 0;
    attr.qp_type = IBV_QPT_RC;
    attr.comp_mask = IBV_QP_INIT_ATTR_PD;

    for (int i = 0; i < n; i++) {
      attr.pd = set->nic.pd_uxdma[i & 1];
      attr.send_cq = set->ibv_cqs[i];
      attr.recv_cq = set->ibv_cqs[i];
      set->ibv_qps[i] = ibv.create_qp_ex(set->nic.context, &attr);
      if (!set->ibv_qps[i]) return -1;
    }
    return 0;
  }
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT: {
    set->bnxt_qps.resize(n);
    if (!set->qp_allocator)
      set->qp_allocator = new HIPAllocatorFinegrained();

    for (int i = 0; i < n; i++) {
      auto *ctx = set->nic.context;
      auto *pd = set->nic.pd_orig;

      struct ibv_qp_init_attr ib_qp_attr;
      memset(&ib_qp_attr, 0, sizeof(ib_qp_attr));
      ib_qp_attr.send_cq = set->bnxt_scqs[i].cq;
      ib_qp_attr.recv_cq = set->bnxt_rcqs[i].cq;
      ib_qp_attr.cap.max_send_wr = set->sq_size;
      ib_qp_attr.cap.max_recv_wr = 0;
      ib_qp_attr.cap.max_send_sge = 1;
      ib_qp_attr.cap.max_recv_sge = 0;
      ib_qp_attr.cap.max_inline_data = set->inline_threshold;
      ib_qp_attr.qp_type = IBV_QPT_RC;
      ib_qp_attr.sq_sig_all = 0;

      memset(&set->bnxt_qps[i].mem_info, 0, sizeof(struct bnxt_re_dv_qp_mem_info));
      if (set->bnxt_re_dv.qp_mem_alloc(pd, &ib_qp_attr, &set->bnxt_qps[i].mem_info) != 0)
        return -1;

      int dmabuf_enabled = ibv.is_dmabuf_supported();

      void *sq_ptr = nullptr;
      set->qp_allocator->allocate(&sq_ptr, set->bnxt_qps[i].mem_info.sq_len);
      (void)hipMemset(sq_ptr, 0, set->bnxt_qps[i].mem_info.sq_len);
      set->bnxt_qps[i].mem_info.sq_va = (uint64_t)sq_ptr;
      set->bnxt_qps[i].sq_buf = sq_ptr;

      if (dmabuf_enabled) {
        if (set->qp_allocator->GetDmabufHandle(sq_ptr,
                                               set->bnxt_qps[i].mem_info.sq_len,
                                               &set->bnxt_qps[i].sq_dmabuf_fd,
                                               &set->bnxt_qps[i].sq_dmabuf_offset) != hipSuccess)
          return -1;
      }

      uint64_t msntbl_len = set->bnxt_qps[i].mem_info.sq_psn_sz * set->bnxt_qps[i].mem_info.sq_npsn;
      uint64_t msntbl_offset = set->bnxt_qps[i].mem_info.sq_len - msntbl_len;
      set->bnxt_qps[i].msntbl = (void*)((char*)sq_ptr + msntbl_offset);
      set->bnxt_qps[i].msn_tbl_sz = set->bnxt_qps[i].mem_info.sq_npsn;

      struct bnxt_re_dv_umem_reg_attr sq_umem;
      memset(&sq_umem, 0, sizeof(sq_umem));
      sq_umem.addr = sq_ptr;
      sq_umem.size = set->bnxt_qps[i].mem_info.sq_len;
      sq_umem.access_flags = IBV_ACCESS_LOCAL_WRITE;
      sq_umem.dmabuf_fd = dmabuf_enabled ? set->bnxt_qps[i].sq_dmabuf_fd : 0;
      void *sq_umem_handle = set->bnxt_re_dv.umem_reg(ctx, &sq_umem);
      if (!sq_umem_handle) return -1;

      void *rq_ptr = nullptr;
      set->qp_allocator->allocate(&rq_ptr, set->bnxt_qps[i].mem_info.rq_len);
      (void)hipMemset(rq_ptr, 0, set->bnxt_qps[i].mem_info.rq_len);
      set->bnxt_qps[i].mem_info.rq_va = (uint64_t)rq_ptr;
      set->bnxt_qps[i].rq_buf = rq_ptr;

      if (dmabuf_enabled) {
        if (set->qp_allocator->GetDmabufHandle(rq_ptr,
                                               set->bnxt_qps[i].mem_info.rq_len,
                                               &set->bnxt_qps[i].rq_dmabuf_fd,
                                               &set->bnxt_qps[i].rq_dmabuf_offset) != hipSuccess)
          return -1;
      }

      struct bnxt_re_dv_umem_reg_attr rq_umem;
      memset(&rq_umem, 0, sizeof(rq_umem));
      rq_umem.addr = rq_ptr;
      rq_umem.size = set->bnxt_qps[i].mem_info.rq_len;
      rq_umem.access_flags = IBV_ACCESS_LOCAL_WRITE;
      rq_umem.dmabuf_fd = dmabuf_enabled ? set->bnxt_qps[i].rq_dmabuf_fd : 0;
      void *rq_umem_handle = set->bnxt_re_dv.umem_reg(ctx, &rq_umem);
      if (!rq_umem_handle) return -1;

      set->bnxt_qps[i].db_region_attr = set->bnxt_re_dv.alloc_db_region(ctx);
      if (!set->bnxt_qps[i].db_region_attr) return -1;

      memset(&set->bnxt_qps[i].attr, 0, sizeof(struct bnxt_re_dv_qp_init_attr));
      set->bnxt_qps[i].attr.send_cq = ib_qp_attr.send_cq;
      set->bnxt_qps[i].attr.recv_cq = ib_qp_attr.recv_cq;
      set->bnxt_qps[i].attr.max_send_wr = ib_qp_attr.cap.max_send_wr;
      set->bnxt_qps[i].attr.max_recv_wr = ib_qp_attr.cap.max_recv_wr;
      set->bnxt_qps[i].attr.max_send_sge = ib_qp_attr.cap.max_send_sge;
      set->bnxt_qps[i].attr.max_recv_sge = ib_qp_attr.cap.max_recv_sge;
      set->bnxt_qps[i].attr.max_inline_data = ib_qp_attr.cap.max_inline_data;
      set->bnxt_qps[i].attr.qp_type = ib_qp_attr.qp_type;
      set->bnxt_qps[i].attr.qp_handle = set->bnxt_qps[i].mem_info.qp_handle;
      set->bnxt_qps[i].attr.dbr_handle = set->bnxt_qps[i].db_region_attr;
      set->bnxt_qps[i].attr.sq_umem_handle = sq_umem_handle;
      set->bnxt_qps[i].attr.sq_len = set->bnxt_qps[i].mem_info.sq_len;
      set->bnxt_qps[i].attr.sq_slots = set->bnxt_qps[i].mem_info.sq_slots;
      set->bnxt_qps[i].attr.sq_wqe_sz = set->bnxt_qps[i].mem_info.sq_wqe_sz;
      set->bnxt_qps[i].attr.sq_psn_sz = set->bnxt_qps[i].mem_info.sq_psn_sz;
      set->bnxt_qps[i].attr.sq_npsn = set->bnxt_qps[i].mem_info.sq_npsn;
      set->bnxt_qps[i].attr.rq_umem_handle = rq_umem_handle;
      set->bnxt_qps[i].attr.rq_len = set->bnxt_qps[i].mem_info.rq_len;
      set->bnxt_qps[i].attr.rq_slots = set->bnxt_qps[i].mem_info.rq_slots;
      set->bnxt_qps[i].attr.rq_wqe_sz = set->bnxt_qps[i].mem_info.rq_wqe_sz;
      set->bnxt_qps[i].attr.comp_mask = set->bnxt_qps[i].mem_info.comp_mask;

      set->ibv_qps[i] = set->bnxt_re_dv.create_qp(pd, &set->bnxt_qps[i].attr);
      if (!set->ibv_qps[i]) return -1;
    }
    return 0;
  }
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5: {
    set->mlx5_qps.resize(n);
    set->inline_threshold = sizeof(gda_mlx5_wqe_inline_data::data);

    for (int i = 0; i < n; i++) {
      int err = set->mlx5dv.create_qp(set->mlx5_qps[i], set->nic.context,
                                        set->nic.pd_orig, set->sq_size);
      if (err) return -1;
    }
    return 0;
  }
#endif
  default:
    return -1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// QP state transitions (adapted from GDABackend::modify_qps_*)
///////////////////////////////////////////////////////////////////////////////

static int gin_modify_qps_rst_to_init(rocshmem_gin_qp_set *set) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_INIT;
  attr.pkey_index = 0;
  attr.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_LOCAL_WRITE
                        | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC;
  attr.port_num = set->nic.port;

  int mask = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;

  for (int i = 0; i < set->nRanks; i++) {
    int err;
#if defined(GDA_BNXT)
    if (set->provider == GDAProvider::BNXT)
      err = set->bnxt_re_dv.modify_qp(set->ibv_qps[i], &attr, mask, 0, 0);
    else
#endif
#if defined(GDA_MLX5)
    if (set->provider == GDAProvider::MLX5)
      err = set->mlx5dv.modify_qp(set->mlx5_qps[i], &attr, mask, set->nic.gid_type);
    else
#endif
      err = ibv.modify_qp(set->ibv_qps[i], &attr, mask);
    if (err) return -1;
  }
  return 0;
}

static int gin_modify_qps_init_to_rtr(rocshmem_gin_qp_set *set,
                                       struct gin_dest_info *remote_info) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTR;
  attr.min_rnr_timer = 12;
  attr.path_mtu = set->nic.portinfo.active_mtu;
  attr.ah_attr.port_num = set->nic.port;

#if defined(GDA_IONIC)
  if (set->provider == GDAProvider::IONIC)
    attr.max_dest_rd_atomic = 15;
  else
#endif
    attr.max_dest_rd_atomic = 1;

  int mask = IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_RQ_PSN | IBV_QP_DEST_QPN
           | IBV_QP_AV | IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;

  for (int i = 0; i < set->nRanks; i++) {
    if (set->nic.portinfo.link_layer == IBV_LINK_LAYER_ETHERNET) {
      attr.ah_attr.grh.sgid_index = set->nic.gid_index;
      attr.ah_attr.is_global = 1;
      attr.ah_attr.grh.hop_limit = 255;
      attr.ah_attr.sl = 1;
      attr.ah_attr.grh.traffic_class = envvar::gda::traffic_class;
      memcpy(&attr.ah_attr.grh.dgid, &remote_info[i].gid, 16);
    } else {
      attr.ah_attr.is_global = 0;
      attr.ah_attr.dlid = remote_info[i].lid;
    }
    attr.rq_psn = remote_info[i].psn;
    attr.dest_qp_num = remote_info[i].qpn;

    int err;
#if defined(GDA_BNXT)
    if (set->provider == GDAProvider::BNXT)
      err = set->bnxt_re_dv.modify_qp(set->ibv_qps[i], &attr, mask, 0, 0);
    else
#endif
#if defined(GDA_MLX5)
    if (set->provider == GDAProvider::MLX5)
      err = set->mlx5dv.modify_qp(set->mlx5_qps[i], &attr, mask, set->nic.gid_type);
    else
#endif
      err = ibv.modify_qp(set->ibv_qps[i], &attr, mask);
    if (err) return -1;
  }
  return 0;
}

static int gin_modify_qps_rtr_to_rts(rocshmem_gin_qp_set *set,
                                      struct gin_dest_info *remote_info) {
  struct ibv_qp_attr attr;
  memset(&attr, 0, sizeof(attr));
  attr.qp_state = IBV_QPS_RTS;
  attr.timeout = 14;
  attr.retry_cnt = 7;
  attr.rnr_retry = 7;

#if defined(GDA_IONIC)
  if (set->provider == GDAProvider::IONIC)
    attr.max_rd_atomic = 15;
  else
#endif
    attr.max_rd_atomic = 1;

  int mask = IBV_QP_STATE | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC
           | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY;

  for (int i = 0; i < set->nRanks; i++) {
    attr.sq_psn = remote_info[i].psn;

    int err;
#if defined(GDA_BNXT)
    if (set->provider == GDAProvider::BNXT)
      err = set->bnxt_re_dv.modify_qp(set->ibv_qps[i], &attr, mask, 0, 0);
    else
#endif
#if defined(GDA_MLX5)
    if (set->provider == GDAProvider::MLX5)
      err = set->mlx5dv.modify_qp(set->mlx5_qps[i], &attr, mask, set->nic.gid_type);
    else
#endif
      err = ibv.modify_qp(set->ibv_qps[i], &attr, mask);
    if (err) return -1;
  }
  return 0;
}

///////////////////////////////////////////////////////////////////////////////
// GPU QP initialization (adapted from GDABackend::*_initialize_gpu_qp)
// Sets up doorbell pointers, CQ/SQ buffers on the GPU-side QueuePair.
// Does NOT set lkey/rkey (GIN uses put_nbi for per-buffer keys).
///////////////////////////////////////////////////////////////////////////////

int rocshmem_gin_qp_set::initialize_gpu_qp(QueuePair *gpu_qp, int idx) {
  switch (this->provider) {
#if defined(GDA_IONIC)
  case GDAProvider::IONIC: {
    ionic_dv_ctx dvctx;
    ionic_dv.get_ctx(&dvctx, nic.context);

    int hip_dev_id = -1;
    if (hipGetDevice(&hip_dev_id) != hipSuccess) return -1;

    void *gpu_db_page = nullptr;
    if (gin_memory_lock_to_fine_grain(dvctx.db_page, 0x1000, &gpu_db_page, hip_dev_id) != 0)
      return -1;

    uint64_t *db_page_u64 = reinterpret_cast<uint64_t*>(dvctx.db_page);
    uint64_t *gpu_db_page_u64 = reinterpret_cast<uint64_t*>(gpu_db_page);
    uint64_t *gpu_db_ptr = &gpu_db_page_u64[dvctx.db_ptr - db_page_u64];

    uint8_t udma_idx = ionic_dv.qp_get_udma_idx(ibv_qps[idx]);

    ionic_dv_cq dvcq;
    ionic_dv.get_cq(&dvcq, ibv_cqs[idx], udma_idx);

    gpu_qp->cq_dbreg = &gpu_db_ptr[dvctx.cq_qtype];
    gpu_qp->cq_dbval = dvcq.q.db_val;
    gpu_qp->cq_mask = dvcq.q.mask;
    gpu_qp->ionic_cq_buf = reinterpret_cast<ionic_v1_cqe*>(dvcq.q.ptr);

    ionic_dv_qp dvqp;
    ionic_dv.get_qp(&dvqp, ibv_qps[idx]);

    gpu_qp->sq_dbreg = &gpu_db_ptr[dvctx.sq_qtype];
    gpu_qp->sq_dbval = dvqp.sq.db_val;
    gpu_qp->sq_mask = dvqp.sq.mask;
    gpu_qp->ionic_sq_buf = reinterpret_cast<ionic_v1_wqe*>(dvqp.sq.ptr);

    gpu_qp->qp_num = ibv_qps[idx]->qp_num;
    gpu_qp->inline_threshold = 32;
    // lkey/rkey left at 0 — GIN uses put_nbi
    return 0;
  }
#endif
#if defined(GDA_BNXT)
  case GDAProvider::BNXT: {
    struct bnxt_re_dv_obj dv_obj;
    struct bnxt_re_dv_cq dv_cq;
    int err;

    // Export QP (required for bnxt DV GPU-direct access)
    struct bnxt_re_dv_qp dv_qp;
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.qp.in = this->ibv_qps[idx];
    dv_obj.qp.out = &dv_qp;
    err = this->bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_QP);
    if (err) return -1;

    // Export SCQ
    memset(&dv_obj, 0, sizeof(dv_obj));
    dv_obj.cq.in = this->bnxt_scqs[idx].cq;
    dv_obj.cq.out = &dv_cq;
    err = this->bnxt_re_dv.init_obj(&dv_obj, BNXT_RE_DV_OBJ_CQ);
    if (err) return -1;

    memset(&gpu_qp->bnxt_cq, 0, sizeof(bnxt_device_cq));
    gpu_qp->bnxt_cq.buf = this->bnxt_scqs[idx].buf;
    gpu_qp->bnxt_cq.depth = this->bnxt_scqs[idx].depth;
    gpu_qp->bnxt_cq.id = dv_cq.cqn;

    // Export SQ
    memset(&gpu_qp->bnxt_sq, 0, sizeof(bnxt_device_sq));
    gpu_qp->bnxt_sq.buf = this->bnxt_qps[idx].sq_buf;
    gpu_qp->bnxt_sq.depth = this->bnxt_qps[idx].mem_info.sq_slots;
    gpu_qp->bnxt_sq.id = this->ibv_qps[idx]->qp_num;
    gpu_qp->bnxt_sq.msntbl = this->bnxt_qps[idx].msntbl;
    gpu_qp->bnxt_sq.msn_tbl_sz = this->bnxt_qps[idx].msn_tbl_sz;
    gpu_qp->bnxt_sq.psn_sz_log2 = std::log2(this->bnxt_qps[idx].mem_info.sq_psn_sz);
    gpu_qp->bnxt_sq.mtu = 128 << this->nic.portinfo.active_mtu; // ibv_mtu enum: 1=256, 2=512, 3=1024, 4=2048, 5=4096

    // Export doorbell
    if (hipHostRegister(this->bnxt_qps[idx].db_region_attr->dbr, getpagesize(),
                        hipHostRegisterDefault) != hipSuccess) return -1;
    if (hipHostGetDevicePointer((void**)&gpu_qp->bnxt_dbr,
                                bnxt_qps[idx].db_region_attr->dbr, 0) != hipSuccess) return -1;

    gpu_qp->qp_num = this->ibv_qps[idx]->qp_num;
    gpu_qp->inline_threshold = inline_threshold;

    LOG_TRACE("gin_qp init[%d]: qp_num=%u sq.buf=%p sq.depth=%u sq.id=%u "
              "cq.buf=%p cq.depth=%u cq.id=%u dbr=%p (host=%p) "
              "msntbl=%p msn_tbl_sz=%u psn_sz_log2=%u mtu=%lu inline=%u",
              idx, gpu_qp->qp_num,
              gpu_qp->bnxt_sq.buf, gpu_qp->bnxt_sq.depth, gpu_qp->bnxt_sq.id,
              gpu_qp->bnxt_cq.buf, gpu_qp->bnxt_cq.depth, gpu_qp->bnxt_cq.id,
              (void*)gpu_qp->bnxt_dbr, (void*)this->bnxt_qps[idx].db_region_attr->dbr,
              gpu_qp->bnxt_sq.msntbl, gpu_qp->bnxt_sq.msn_tbl_sz,
              gpu_qp->bnxt_sq.psn_sz_log2, (unsigned long)gpu_qp->bnxt_sq.mtu,
              gpu_qp->inline_threshold);
    return 0;
  }
#endif
#if defined(GDA_MLX5)
  case GDAProvider::MLX5: {
    mlx5_devx_qp &qp = mlx5_qps[idx];

    int hip_dev_id = -1;
    if (hipGetDevice(&hip_dev_id) != hipSuccess) return -1;

    gpu_qp->mlx5_cq = gda_mlx5_device_cq(
      (mlx5_cqe64*)qp.cq,
      qp.cq_dbrec
    );

    void *gpu_db_ptr = nullptr;
    if (gin_memory_lock_to_fine_grain(qp.uar->reg_addr,
                                      MLX5_DB_BLUEFLAME_BUFFER_SIZE,
                                      &gpu_db_ptr, hip_dev_id) != 0)
      return -1;

    gpu_qp->mlx5_sq = gda_mlx5_device_sq{
      (gda_mlx5_wqe*)qp.sq,
      &qp.qp_dbrec[MLX5_SND_DBR],
      (gda_mlx5_doorbell*)gpu_db_ptr,
      qp.sq_depth
    };

    gpu_qp->qp_num = qp.qpn;
    gpu_qp->inline_threshold = inline_threshold;
    // lkey/rkey: for MLX5, keys are big-endian. Left at 0 for GIN.
    return 0;
  }
#endif
  default:
    return -1;
  }
}

///////////////////////////////////////////////////////////////////////////////
// Public API implementation
///////////////////////////////////////////////////////////////////////////////

int rocshmem_gin_create_qps(int nRanks, int myRank,
                             int (*allgather)(void* ctx, void* buf, size_t size),
                             void* allgather_ctx,
                             rocshmem_gin_qp_set_t *out_qp_set,
                             void ***out_gpu_qps) {
  // Note: log_pe_number and device-side logd_constants must be initialized
  // by the consumer binary (e.g. rccl-tests) which links rocshmem's device
  // bitcode and can resolve HIP_SYMBOL(logd_constants).

  auto *set = new rocshmem_gin_qp_set();
  set->nRanks = nRanks;
  set->myRank = myRank;

  // 1. Open IB device and detect provider
  if (gin_open_ib_device(set) != 0) {
    delete set;
    return -1;
  }

  // 2. Load provider-specific DV library
  if (gin_load_dv_library(set) != 0) {
    LOG_ERROR("GIN QP factory: failed to load DV library for provider %d", set->provider);
    ibv.dealloc_pd(set->nic.pd_orig);
    ibv.close_device(set->nic.context);
    delete set;
    return -1;
  }

  // 3. Create parent domains (IONIC, MLX5)
  if (gin_create_parent_domain(set) != 0) {
    LOG_ERROR("GIN QP factory: failed to create parent domain");
    goto fail;
  }

  // 4. Create CQs
  if (gin_create_cqs(set) != 0) {
    LOG_ERROR("GIN QP factory: failed to create CQs");
    goto fail;
  }

  // 5. Create QPs
  if (gin_create_qps(set) != 0) {
    LOG_ERROR("GIN QP factory: failed to create QPs");
    goto fail;
  }

  {
    // 6. Fill local dest_info and exchange via allgather
    std::vector<gin_dest_info> all_dest_info(nRanks);
    gin_dest_info my_info;
    my_info.lid = set->nic.portinfo.lid;
    my_info.psn = 0;
    my_info.gid = set->nic.gid;

    // For allgather: each rank contributes one dest_info per QP it owns.
    // Since we have 1 QP per peer, rank i's QP connects to peer i.
    // The allgather exchanges each rank's single dest_info.
    // After allgather, all_dest_info[r] = rank r's QP info.

    // Fill my entry — all my QPs have different qpn but same LID/GID
    // For the allgather, we exchange one info per rank (QP number + addressing)
    // We pick a single QP to represent this rank to each peer.
    // Since each peer i talks to us via our QP i, we need to send per-peer info.
    // Use a flat array: nRanks * nRanks entries, allgathered.
    std::vector<gin_dest_info> local_infos(nRanks);
    std::vector<gin_dest_info> all_infos(nRanks * nRanks);

    for (int i = 0; i < nRanks; i++) {
      local_infos[i].lid = set->nic.portinfo.lid;
      local_infos[i].psn = 0;
      local_infos[i].gid = set->nic.gid;
#if defined(GDA_MLX5)
      if (set->provider == GDAProvider::MLX5)
        local_infos[i].qpn = set->mlx5_qps[i].qpn;
      else
#endif
        local_infos[i].qpn = set->ibv_qps[i]->qp_num;
    }

    // Place my info in the allgather buffer
    memcpy(&all_infos[myRank * nRanks], local_infos.data(),
           nRanks * sizeof(gin_dest_info));

    // Allgather: each rank contributes nRanks entries
    if (allgather(allgather_ctx, all_infos.data(),
                  nRanks * sizeof(gin_dest_info)) != 0) goto fail;

    // Extract remote info: for my QP[i], the remote end is rank i's QP[myRank]
    std::vector<gin_dest_info> remote_info(nRanks);
    for (int i = 0; i < nRanks; i++) {
      remote_info[i] = all_infos[i * nRanks + myRank];
    }

    // 7. State transitions
    if (gin_modify_qps_rst_to_init(set) != 0) {
      LOG_ERROR("GIN QP factory: QP state RST->INIT failed (check IB permissions/MTU)");
      goto fail;
    }
    if (gin_modify_qps_init_to_rtr(set, remote_info.data()) != 0) {
      LOG_ERROR("GIN QP factory: QP state INIT->RTR failed (check remote dest_info / GID)");
      goto fail;
    }
    if (gin_modify_qps_rtr_to_rts(set, remote_info.data()) != 0) {
      LOG_ERROR("GIN QP factory: QP state RTR->RTS failed");
      goto fail;
    }

    // Dump QP state for comparison with GDABackend
    dump_ibv_context(set->nic.context);
    dump_ibv_device(set->nic.context->device);
    dump_ibv_pd(set->nic.pd_orig);
    dump_ibv_port_attr(&set->nic.portinfo);
    for (int i = 0; i < nRanks; i++) {
      if (set->ibv_qps[i]) dump_ibv_qp(set->ibv_qps[i], i);
#if defined(GDA_MLX5)
      if (set->provider == GDAProvider::MLX5) set->mlx5_qps[i].dump(i);
#endif
    }
  }

  {
    // 8. Construct QueuePair objects and copy to GPU
    size_t qp_size = sizeof(QueuePair) * nRanks;

    if (hipMalloc(&set->gpu_qps, qp_size) != hipSuccess) goto fail;
    set->host_qps = (QueuePair*)malloc(qp_size);
    if (!set->host_qps) goto fail;

    for (int i = 0; i < nRanks; i++) {
      new (&set->host_qps[i]) QueuePair(set->nic.pd_orig, set->provider);
      if (hipMemcpy(&set->gpu_qps[i], &set->host_qps[i], sizeof(QueuePair),
                    hipMemcpyHostToDevice) != hipSuccess) goto fail;

      // 9. Initialize GPU-specific state (doorbells, CQ/SQ buffers)
      if (set->initialize_gpu_qp(&set->gpu_qps[i], i) != 0) goto fail;
    }

    // Build array of QueuePair pointers for the GPU context
    QueuePair **host_ptrs = (QueuePair**)malloc(nRanks * sizeof(QueuePair*));
    for (int i = 0; i < nRanks; i++)
      host_ptrs[i] = &set->gpu_qps[i];

    void **gpu_ptr_array = nullptr;
    if (hipMalloc(&gpu_ptr_array, nRanks * sizeof(void*)) != hipSuccess) {
      free(host_ptrs);
      goto fail;
    }
    if (hipMemcpy(gpu_ptr_array, host_ptrs, nRanks * sizeof(void*),
                  hipMemcpyHostToDevice) != hipSuccess) {
      free(host_ptrs);
      (void)hipFree(gpu_ptr_array);
      goto fail;
    }
    free(host_ptrs);

    *out_gpu_qps = gpu_ptr_array;
  }

  LOG_TRACE("GIN QP factory: %d QPs ready on %s (rank %d/%d)",
            nRanks, set->nic.nic_name.c_str(), myRank, nRanks);
  *out_qp_set = set;
  return 0;

fail:
  LOG_ERROR("GIN QP factory: rocshmem_gin_create_qps failed (rank %d/%d)", myRank, nRanks);
  rocshmem_gin_destroy_qps(set);
  return -1;
}

void rocshmem_gin_destroy_qps(rocshmem_gin_qp_set_t qp_set) {
  if (!qp_set) return;

  // Destroy QueuePair objects
  if (qp_set->host_qps) {
    for (int i = 0; i < qp_set->nRanks; i++)
      qp_set->host_qps[i].~QueuePair();
    free(qp_set->host_qps);
  }
  if (qp_set->gpu_qps) (void)hipFree(qp_set->gpu_qps);

#if defined(GDA_BNXT)
  if (qp_set->provider == GDAProvider::BNXT) {
    for (size_t i = 0; i < qp_set->bnxt_qps.size(); i++) {
      auto &bqp = qp_set->bnxt_qps[i];

      if (qp_set->ibv_qps[i])
        (void)qp_set->bnxt_re_dv.destroy_qp(qp_set->ibv_qps[i]);

      if (bqp.db_region_attr) {
        (void)hipHostUnregister(bqp.db_region_attr->dbr);
        if (qp_set->bnxt_re_dv.free_db_region)
          (void)qp_set->bnxt_re_dv.free_db_region(qp_set->nic.context, bqp.db_region_attr);
      }

      if (bqp.attr.rq_umem_handle)
        (void)qp_set->bnxt_re_dv.umem_dereg(bqp.attr.rq_umem_handle);
      if (bqp.attr.sq_umem_handle)
        (void)qp_set->bnxt_re_dv.umem_dereg(bqp.attr.sq_umem_handle);

      if (bqp.sq_buf) qp_set->qp_allocator->deallocate(bqp.sq_buf);
      if (bqp.rq_buf) qp_set->qp_allocator->deallocate(bqp.rq_buf);

      if (bqp.sq_dmabuf_fd > 0) close(bqp.sq_dmabuf_fd);
      if (bqp.rq_dmabuf_fd > 0) close(bqp.rq_dmabuf_fd);
    }

    for (size_t i = 0; i < qp_set->bnxt_scqs.size(); i++) {
      auto &scq = qp_set->bnxt_scqs[i];
      auto &rcq = qp_set->bnxt_rcqs[i];

      if (scq.cq) (void)qp_set->bnxt_re_dv.destroy_cq(scq.cq);
      if (rcq.cq) (void)qp_set->bnxt_re_dv.destroy_cq(rcq.cq);

      if (scq.umem_handle) (void)qp_set->bnxt_re_dv.umem_dereg(scq.umem_handle);
      if (rcq.umem_handle) (void)qp_set->bnxt_re_dv.umem_dereg(rcq.umem_handle);

      if (scq.buf) qp_set->qp_allocator->deallocate(scq.buf);
      if (rcq.buf) qp_set->qp_allocator->deallocate(rcq.buf);

      if (scq.dmabuf_fd > 0) close(scq.dmabuf_fd);
      if (rcq.dmabuf_fd > 0) close(rcq.dmabuf_fd);
    }

    delete qp_set->qp_allocator;
  } else
#endif
#if defined(GDA_MLX5)
  if (qp_set->provider == GDAProvider::MLX5) {
    for (auto &mq : qp_set->mlx5_qps)
      qp_set->mlx5dv.destroy_qp(mq);
  } else
#endif
  {
    // IONIC and fallback: standard verbs cleanup
    for (auto *qp : qp_set->ibv_qps)
      if (qp) ibv.destroy_qp(qp);
    for (auto *cq : qp_set->ibv_cqs)
      if (cq) ibv.destroy_cq(cq);
  }

  // PDs and device
  if (qp_set->nic.pd_uxdma[0]) ibv.dealloc_pd(qp_set->nic.pd_uxdma[0]);
  if (qp_set->nic.pd_uxdma[1]) ibv.dealloc_pd(qp_set->nic.pd_uxdma[1]);
  if (qp_set->nic.pd_parent) ibv.dealloc_pd(qp_set->nic.pd_parent);
  if (qp_set->nic.pd_orig) ibv.dealloc_pd(qp_set->nic.pd_orig);
  if (qp_set->nic.context) ibv.close_device(qp_set->nic.context);

  if (qp_set->dv_handle) dlclose(qp_set->dv_handle);

  delete qp_set;
}

int rocshmem_gin_reg_mr(rocshmem_gin_qp_set_t qp_set,
                         void *addr, size_t size, int atomic,
                         void **out_mr, uint32_t *out_lkey, uint32_t *out_rkey) {
  if (!qp_set || !qp_set->nic.pd_orig) return -1;

  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ;
  if (atomic) access |= IBV_ACCESS_REMOTE_ATOMIC;

  struct ibv_mr *mr = ibv.reg_mr(qp_set->nic.pd_orig, addr, size, access);
  if (!mr) return -1;

  *out_mr = mr;
  *out_lkey = mr->lkey;
  *out_rkey = mr->rkey;
  return 0;
}

static std::map<uintptr_t, int> gin_dmabuf_fd_map;

int rocshmem_gin_reg_mr_vmm(rocshmem_gin_qp_set_t qp_set,
                             void *addr, size_t size, int atomic,
                             void **out_mr, uint32_t *out_lkey, uint32_t *out_rkey) {
  if (!qp_set || !qp_set->nic.pd_orig) return -1;

  int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE
             | IBV_ACCESS_REMOTE_READ;
  if (atomic) access |= IBV_ACCESS_REMOTE_ATOMIC;

  struct ibv_pd *pd = qp_set->nic.pd_orig;
  struct ibv_mr *mr = nullptr;

#if HIP_VERSION >= 70000000
  {
    int fd = -1;
    static size_t page_size = sysconf(_SC_PAGESIZE);
    size_t aligned_size = (size + page_size - 1) & ~(page_size - 1);

    hipError_t err = hipMemGetHandleForAddressRange(
        (void *)&fd, (hipDeviceptr_t)addr, aligned_size,
        hipMemRangeHandleTypeDmaBufFd, 0);
    if (err == hipSuccess && fd >= 0) {
#if 0
      // iova=0: offset-based addressing, no baseAddr needed on device side.
      // Requires NIC support (mlx5 yes, bnxt no as of 2026-06).
      wrap_ibv_reg_dmabuf_mr(&mr, pd, 0, aligned_size, 0, fd, access);
#else
      wrap_ibv_reg_dmabuf_mr(&mr, pd, 0, aligned_size, (uint64_t)addr, fd, access);
#endif
      if (mr) {
        gin_dmabuf_fd_map[(uintptr_t)mr] = fd;
        LOG_TRACE("gin_reg_mr_vmm: dmabuf for %p size %zd lkey=0x%x rkey=0x%x",
                  addr, size, mr->lkey, mr->rkey);
      } else {
        close(fd);
      }
    }
    if (!mr) {
      LOG_TRACE("gin_reg_mr_vmm: dmabuf failed, trying iova2 fallback");
    }
  }
#endif

  if (!mr) {
#if 0
    // iova=0 fallback for NICs that support offset-based addressing
    wrap_ibv_reg_mr_iova2(&mr, pd, addr, size, 0, access);
#else
    wrap_ibv_reg_mr_iova2(&mr, pd, addr, size, (uintptr_t)addr, access);
#endif
  }
  if (!mr) return -1;

  *out_mr = mr;
  *out_lkey = mr->lkey;
  *out_rkey = mr->rkey;
  return 0;
}

void rocshmem_gin_dereg_mr(void *mr) {
  if (!mr) return;
  auto it = gin_dmabuf_fd_map.find((uintptr_t)mr);
  if (it != gin_dmabuf_fd_map.end()) {
    close(it->second);
    gin_dmabuf_fd_map.erase(it);
  }
  wrap_ibv_dereg_mr((struct ibv_mr*)mr);
}

int rocshmem_gin_get_provider(rocshmem_gin_qp_set_t qp_set) {
  if (!qp_set) return -1;
  return qp_set->provider;
}

static bool gin_validate_device(int provider, struct ibv_device_attr *dev_attr) {
#if defined(GDA_BNXT)
  if (provider == GDAProvider::BNXT) {
    const uint32_t supported_bnxt_part_ids[] = { 0x1760 /* BCM57608 */ };
    const char min_fw_ver[] = "233.2.104.0";

    bool part_ok = false;
    for (auto pid : supported_bnxt_part_ids) {
      if (dev_attr->vendor_part_id == pid) { part_ok = true; break; }
    }
    if (!part_ok) {
      LOG_WARN("GIN probe: unsupported BNXT part_id=0x%x", dev_attr->vendor_part_id);
      return false;
    }
    if (strverscmp(min_fw_ver, dev_attr->fw_ver) > 0) {
      LOG_WARN("GIN probe: BNXT firmware %s below minimum %s", dev_attr->fw_ver, min_fw_ver);
      return false;
    }
  }
#endif
  return true;
}

int rocshmem_gin_probe_devices(void) {
  if (!ibv.is_initialized) return 0;

  int ndev = 0;
  struct ibv_device **dev_list = ibv.get_device_list(&ndev);
  if (!dev_list || ndev == 0) return 0;

  int count = 0;
  for (int d = 0; d < ndev; d++) {
    struct ibv_context *ctx = ibv.open_device(dev_list[d]);
    if (!ctx) continue;

    struct ibv_device_attr dev_attr;
    if (ibv.query_device(ctx, &dev_attr) != 0) {
      ibv.close_device(ctx);
      continue;
    }

    int provider = gin_detect_provider(&dev_attr);
    if (provider < 0 || !gin_validate_device(provider, &dev_attr)) {
      ibv.close_device(ctx);
      continue;
    }

    for (int port = 1; port <= dev_attr.phys_port_cnt; port++) {
      struct ibv_port_attr port_attr;
      if (ibv.query_port(ctx, port, &port_attr) == 0 &&
          port_attr.state == IBV_PORT_ACTIVE) {
        count++;
        break;
      }
    }
    ibv.close_device(ctx);
  }

  ibv.free_device_list(dev_list);
  return count;
}

