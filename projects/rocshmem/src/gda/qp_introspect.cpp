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

#include <atomic>

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
  // device memory; host_qps[conn] holds only the initial construct, so the
  // device copy is the one to read.
  //
  // Copy the individual members rather than the whole object. QueuePair is
  // polymorphic (virtual destructor), so it is not trivially copyable: memcpy
  // of the object bytes followed by reinterpret_cast to QueuePair* would be
  // undefined behaviour, and the copied vtable pointer would be meaningless on
  // the host anyway. Each member read below is a POD, which is well defined.
  const QueuePair *dev_qp = &gpu_qps[conn];  // device address, never
                                             // dereferenced on the host
  auto fetch = [](auto *dst, const auto *device_src) {
    return hipMemcpy(dst, device_src, sizeof(*dst),
                     hipMemcpyDeviceToHost) == hipSuccess;
  };

  decltype(QueuePair::base_heap) base_heap{};
  decltype(QueuePair::lkey) lkey{};
  decltype(QueuePair::rkey) rkey{};
  decltype(QueuePair::qp_num) qp_num{};
  if (!fetch(&base_heap, &dev_qp->base_heap)) return false;
  if (!fetch(&lkey, &dev_qp->lkey)) return false;
  if (!fetch(&rkey, &dev_qp->rkey)) return false;
  if (!fetch(&qp_num, &dev_qp->qp_num)) return false;

  out->base_heap = static_cast<uint64_t>(base_heap);
  out->lkey = lkey;
  out->rkey = rkey;
  out->qpn = qp_num;
  out->vendor = static_cast<QpInfoVendor>(gda_provider);

  switch (gda_provider) {
    case GDAProvider::IONIC: {
      decltype(QueuePair::ionic_sq_buf) ionic_sq_buf{};
      decltype(QueuePair::ionic_cq_buf) ionic_cq_buf{};
      decltype(QueuePair::sq_mask) sq_mask{};
      decltype(QueuePair::cq_mask) cq_mask{};
      decltype(QueuePair::sq_dbreg) sq_dbreg{};
      decltype(QueuePair::sq_dbval) sq_dbval{};
      if (!fetch(&ionic_sq_buf, &dev_qp->ionic_sq_buf)) return false;
      if (!fetch(&ionic_cq_buf, &dev_qp->ionic_cq_buf)) return false;
      if (!fetch(&sq_mask, &dev_qp->sq_mask)) return false;
      if (!fetch(&cq_mask, &dev_qp->cq_mask)) return false;
      if (!fetch(&sq_dbreg, &dev_qp->sq_dbreg)) return false;
      if (!fetch(&sq_dbval, &dev_qp->sq_dbval)) return false;

      out->sq_buf = reinterpret_cast<uint64_t>(ionic_sq_buf);
      out->cq_buf = reinterpret_cast<uint64_t>(ionic_cq_buf);
      // Live producer index, taken from the device array itself rather than the
      // host copy: an external builder has to continue this sequence.
      out->sq_prod = reinterpret_cast<uint64_t>(&gpu_qps[conn].sq_prod);
      out->sq_depth = static_cast<uint32_t>(sq_mask + 1);
      out->cq_depth = static_cast<uint32_t>(cq_mask + 1);
      out->ionic.db = reinterpret_cast<uint64_t>(sq_dbreg);
      out->ionic.dbval = sq_dbval;
      out->ionic.sq_mask = sq_mask;
      out->ionic.cq_mask = cq_mask;
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
      // Atomic, not a plain bool: this entry point can be called from several
      // threads, and a non-atomic read/write pair would be a data race, which
      // is undefined behaviour however harmless the intent.
      static std::atomic<bool> warned{false};
      if (!warned.exchange(true, std::memory_order_relaxed)) {
        LOG_WARN("QP introspection is not implemented for the detected GDA "
                 "provider (%s), so no queue pair will be reported. Only ionic "
                 "is supported today.",
                 provider == QpInfoVendor::MLX5 ? "mlx5" : "bnxt");
      }
    }
    return false;
  }
  // Refuse the default context. ctx_id 0 is the QP rocSHMEM drives for its own
  // operations, keeping a private producer index, send-queue lock and cached
  // doorbell position that an external builder updates none of. Measured on
  // ionic: external posts alone do not break collectives (barrier_all still
  // completes), but a workload mixing external posts with its own completion
  // polling on that queue fails to finish, where the same workload on a user
  // context does. Handing the descriptor out is the whole risk, so the refusal
  // belongs here rather than in a doc comment. Callers must create a user
  // context (ctx_id > 0) and introspect that.
  //
  // fill_qp_info() itself still accepts ctx_id 0: it is the backend's own
  // accessor and the default context is legitimate internally. Only this
  // published entry point is restricted.
  if (ctx_id <= 0) return false;
  return static_cast<GDABackend *>(backend)->fill_qp_info(peer, ctx_id, out);
}

}  // namespace rocshmem
