/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * Per-peer GDA queue-pair introspection.
 * See include/rocshmem/qp_introspect.hpp for the public contract.
 *
 * Lives in a library TU where the GDA internals are in scope; the public
 * surface is only the flat POD plus the free function.
 */
#include "rocshmem/qp_introspect.hpp"

#include <hip/hip_runtime.h>

#include "backend_bc.hpp"
#include "backend_type.hpp"
#include "gda/backend_gda.hpp"
#include "gda/gda_enums.hpp"
#include "gda/queue_pair.hpp"
#include "log.hpp"

namespace rocshmem {

extern Backend *backend;

// The public QpInfoVendor duplicates the internal GDAProvider so that
// qp_introspect.hpp needs no internal includes. Pin them together so the two
// cannot drift silently.
static_assert(static_cast<uint32_t>(QpInfoVendor::IONIC) ==
                  static_cast<uint32_t>(GDAProvider::IONIC),
              "QpInfoVendor::IONIC must match GDAProvider::IONIC");
static_assert(static_cast<uint32_t>(QpInfoVendor::BNXT) ==
                  static_cast<uint32_t>(GDAProvider::BNXT),
              "QpInfoVendor::BNXT must match GDAProvider::BNXT");
static_assert(static_cast<uint32_t>(QpInfoVendor::MLX5) ==
                  static_cast<uint32_t>(GDAProvider::MLX5),
              "QpInfoVendor::MLX5 must match GDAProvider::MLX5");

bool GDABackend::fill_qp_info(int peer, int ctx_id, QpInfo *out) {
  if (out == nullptr) return false;
  if (peer < 0 || peer >= static_cast<int>(num_pes)) return false;
  if (ctx_id < 0) return false;
  if (gpu_qps == nullptr) return false;

  // Index math mirrors GDAContext's constructor (context_gda_device.cpp): a
  // context's QPs form a contiguous run in the backend's array, with the
  // default context first and each user context after it.
  const size_t qps_per_pe =
      ctx_id ? qps_per_pe_usr_ctx_ : qps_per_pe_default_ctx_;
  size_t offset = (ctx_id > 0) ? (qps_per_pe_default_ctx_ +
                                  qps_per_pe_usr_ctx_ * (ctx_id - 1))
                               : 0;
  offset *= num_pes;

  const size_t conn = offset + static_cast<size_t>(peer) * qps_per_pe;
  if (conn >= static_cast<size_t>(num_qps)) return false;

  // initialize_gpu_qp writes the field VALUES straight into gpu_qps[conn] in
  // device memory; host_qps[conn] holds only the initial construct. So read the
  // device copy. QueuePair has no default constructor -- copy the bytes
  // into raw aligned storage and read POD members from it; no constructor
  // or destructor runs on this buffer.
  alignas(QueuePair) unsigned char storage[sizeof(QueuePair)];
  if (hipMemcpy(storage, &gpu_qps[conn], sizeof(QueuePair),
                hipMemcpyDeviceToHost) != hipSuccess) {
    return false;
  }
  const QueuePair *qp = reinterpret_cast<const QueuePair *>(storage);

  out->base_heap = static_cast<uint64_t>(qp->base_heap);
  out->lkey = qp->lkey;
  out->rkey = qp->rkey;
  out->qpn = qp->qp_num;
  out->vendor = static_cast<QpInfoVendor>(gda_provider);

  switch (gda_provider) {
    case GDAProvider::IONIC: {
      out->sq_buf = reinterpret_cast<uint64_t>(qp->ionic_sq_buf);
      out->cq_buf = reinterpret_cast<uint64_t>(qp->ionic_cq_buf);
      // Live producer index, taken from the device array itself rather than the
      // host copy: an external builder has to continue this sequence.
      out->sq_prod = reinterpret_cast<uint64_t>(&gpu_qps[conn].sq_prod);
      out->sq_depth = static_cast<uint32_t>(qp->sq_mask + 1);
      out->cq_depth = static_cast<uint32_t>(qp->cq_mask + 1);
      out->ionic.db = reinterpret_cast<uint64_t>(qp->sq_dbreg);
      out->ionic.dbval = qp->sq_dbval;
      out->ionic.sq_mask = qp->sq_mask;
      out->ionic.cq_mask = qp->cq_mask;
      // Which UDMA engine this QP is bound to. Not derivable from the device
      // struct; query the verbs QP the same way setup_gpu_qp does.
      if (qps.size() <= conn || qps[conn] == nullptr) return false;
      if (ionic_dv.qp_get_udma_idx == nullptr) return false;
      out->ionic.udma_idx = ionic_dv.qp_get_udma_idx(qps[conn]);
      return true;
    }

    case GDAProvider::MLX5:
    case GDAProvider::BNXT:
      // Declared in QpInfo's union but not implemented (TODO). Publishing
      // untested ring or doorbell addresses that a consumer would post work
      // requests into is worse than reporting failure: a wrong address surfaces
      // as silent data corruption or a NIC error far from its cause.
      //
      // The caller-facing warning lives in rocshmem_query_qp_info(), which now
      // rejects unsupported providers before reaching here.
      return false;

    case GDAProvider::UNSET:
    default:
      return false;
  }
}

QpInfoVendor rocshmem_qp_introspect_provider() {
  if (backend == nullptr) return QpInfoVendor::UNKNOWN;
  if (backend->get_type() != BackendType::GDA_BACKEND) {
    return QpInfoVendor::UNKNOWN;
  }
  // Reports what was detected, including providers this API cannot describe --
  // that is the point: it lets a caller distinguish "no GDA here" from
  // "GDA on a NIC we have not implemented yet", which otherwise look the
  // same.
  return static_cast<QpInfoVendor>(
      static_cast<GDABackend *>(backend)->get_gda_provider());
}

bool rocshmem_qp_introspect_available() {
  // Must check the provider, not just the backend. Only ionic is
  // implemented, so reporting "available" on an mlx5 or bnxt GDA backend
  // would promise a capability that rocshmem_query_qp_info then refuses --
  // the caller is told to go ahead and gets nothing back, with no way to
  // tell why.
  return rocshmem_qp_introspect_provider() == QpInfoVendor::IONIC;
}

bool rocshmem_query_qp_info(int peer, int ctx_id, QpInfo *out) {
  const QpInfoVendor provider = rocshmem_qp_introspect_provider();
  if (provider != QpInfoVendor::IONIC) {
    // A provider was detected, we just cannot describe it. Say so, once: to a
    // caller this failure is otherwise indistinguishable from "no GDA backend",
    // and the fix for each is completely different.
    if (provider != QpInfoVendor::UNKNOWN) {
      static bool warned = false;  // benign race: at worst the warning repeats
      if (!warned) {
        warned = true;
        LOG_WARN("QP introspection is not implemented for the detected GDA "
                 "provider (%s), so no queue pair will be reported. Only ionic "
                 "is supported today.",
                 provider == QpInfoVendor::MLX5 ? "mlx5" : "bnxt");
      }
    }
    return false;
  }
  // Refuse the default context. ctx_id 0 is the QP rocSHMEM drives for its own
  // collectives, and it keeps private send-queue bookkeeping for it. A caller
  // that posts its own WQEs there corrupts that state and deadlocks operations
  // like rocshmem_barrier_all. Handing the descriptor out is the whole risk, so
  // the refusal belongs here rather than in a doc comment. Callers must
  // create a user context (ctx_id > 0) and introspect that.
  //
  // fill_qp_info() itself still accepts ctx_id 0: it is the backend's own
  // accessor and the default context is legitimate internally. Only this
  // published entry point is restricted.
  if (ctx_id <= 0) return false;
  return static_cast<GDABackend *>(backend)->fill_qp_info(peer, ctx_id, out);
}

}  // namespace rocshmem
