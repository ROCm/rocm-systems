/*************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 ************************************************************************/

// gin_ibv_shim.cc — GIN QP factory's IB verbs backend, built on rccl's wrap_ibv_*.
//
// WHY THIS FILE EXISTS
// --------------------
// Two reasons, both about letting the GIN QP factory use rccl's native IB verbs
// layer so librccl.so needs no rocSHMEM host code:
//
//  1. rccl's native wrapper (ibvwrap.h) does not expose every verb the factory
//     needs. The extended verbs for provider QP/CQ setup — create_cq_ex,
//     cq_ex_to_cq, alloc_parent_domain, create_qp_ex, query_gid_table — plus a
//     DMA-BUF capability probe are added alongside this shim (see ibvwrap.*,
//     ibvsymbols.*, ibvcore.h). This file gathers the full set the factory calls.
//
//  2. ibvwrap.h (rccl's ibvcore.h) and the factory's rocSHMEM gda/ibv_core.hpp
//     define the same struct ibv_* tags and cannot coexist in one translation
//     unit. This shim is the single TU that owns rccl's ibv types; the factory
//     never includes ibvwrap.h. The two sides exchange only opaque struct ibv_*
//     pointers, so a C++ call binds by tag name alone — a rocSHMEM-typed ibv_qp*
//     on the factory side resolves to the rccl-typed ibv_qp* here. Both headers
//     model the same libibverbs ABI, so the pointed-to layouts are identical.
//
// It also restores rocSHMEM's direct-return convention (pointer-or-NULL / int)
// over rccl's out-param + ncclResult_t one, keeping the factory call sites a
// mechanical `ibv.` -> `gin_ibv_` rename. See gin_ibv_shim.h.

#include "gin/gin_ibv_shim.h"
#include "ibvwrap.h"

int gin_ibv_available(void) {
  return wrap_ibv_symbols() == ncclSuccess ? 1 : 0;
}

struct ibv_device** gin_ibv_get_device_list(int* num_devices) {
  struct ibv_device** list = nullptr;
  if (wrap_ibv_get_device_list(&list, num_devices) != ncclSuccess) return nullptr;
  return list;
}

void gin_ibv_free_device_list(struct ibv_device** list) {
  (void)wrap_ibv_free_device_list(list);
}

const char* gin_ibv_get_device_name(struct ibv_device* device) {
  return wrap_ibv_get_device_name(device);
}

struct ibv_context* gin_ibv_open_device(struct ibv_device* device) {
  struct ibv_context* ctx = nullptr;
  if (wrap_ibv_open_device(&ctx, device) != ncclSuccess) return nullptr;
  return ctx;
}

int gin_ibv_close_device(struct ibv_context* context) {
  return wrap_ibv_close_device(context) == ncclSuccess ? 0 : -1;
}

int gin_ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) {
  return wrap_ibv_query_device(context, device_attr) == ncclSuccess ? 0 : -1;
}

int gin_ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr) {
  return wrap_ibv_query_port(context, port_num, port_attr) == ncclSuccess ? 0 : -1;
}

struct ibv_pd* gin_ibv_alloc_pd(struct ibv_context* context) {
  struct ibv_pd* pd = nullptr;
  if (wrap_ibv_alloc_pd(&pd, context) != ncclSuccess) return nullptr;
  return pd;
}

int gin_ibv_dealloc_pd(struct ibv_pd* pd) {
  return wrap_ibv_dealloc_pd(pd) == ncclSuccess ? 0 : -1;
}

struct ibv_mr* gin_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  struct ibv_mr* mr = nullptr;
  if (wrap_ibv_reg_mr(&mr, pd, addr, length, access) != ncclSuccess) return nullptr;
  return mr;
}

struct ibv_mr* gin_ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova, int access) {
  struct ibv_mr* mr = nullptr;
  if (wrap_ibv_reg_mr_iova2(&mr, pd, addr, length, iova, access) != ncclSuccess) return nullptr;
  return mr;
}

struct ibv_mr* gin_ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova, int fd,
                                     int access) {
  struct ibv_mr* mr = nullptr;
  if (wrap_ibv_reg_dmabuf_mr(&mr, pd, offset, length, iova, fd, access) != ncclSuccess) return nullptr;
  return mr;
}

int gin_ibv_dereg_mr(struct ibv_mr* mr) {
  return wrap_ibv_dereg_mr(mr) == ncclSuccess ? 0 : -1;
}

int gin_ibv_destroy_cq(struct ibv_cq* cq) {
  return wrap_ibv_destroy_cq(cq) == ncclSuccess ? 0 : -1;
}

int gin_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  return wrap_ibv_modify_qp(qp, attr, attr_mask) == ncclSuccess ? 0 : -1;
}

int gin_ibv_destroy_qp(struct ibv_qp* qp) {
  return wrap_ibv_destroy_qp(qp) == ncclSuccess ? 0 : -1;
}

struct ibv_cq_ex* gin_ibv_create_cq_ex(struct ibv_context* context, struct ibv_cq_init_attr_ex* cq_attr) {
  struct ibv_cq_ex* cq = nullptr;
  if (wrap_ibv_create_cq_ex(&cq, context, cq_attr) != ncclSuccess) return nullptr;
  return cq;
}

struct ibv_cq* gin_ibv_cq_ex_to_cq(struct ibv_cq_ex* cq) {
  return wrap_ibv_cq_ex_to_cq(cq);
}

struct ibv_pd* gin_ibv_alloc_parent_domain(struct ibv_context* context, struct ibv_parent_domain_init_attr* attr) {
  struct ibv_pd* pd = nullptr;
  if (wrap_ibv_alloc_parent_domain(&pd, context, attr) != ncclSuccess) return nullptr;
  return pd;
}

struct ibv_qp* gin_ibv_create_qp_ex(struct ibv_context* context, struct ibv_qp_init_attr_ex* qp_init_attr_ex) {
  struct ibv_qp* qp = nullptr;
  if (wrap_ibv_create_qp_ex(&qp, context, qp_init_attr_ex) != ncclSuccess) return nullptr;
  return qp;
}

ssize_t gin_ibv_query_gid_table(struct ibv_context* context, struct ibv_gid_entry* entries, size_t max_entries,
                                uint32_t flags) {
  return wrap_ibv_query_gid_table(context, entries, max_entries, flags);
}

int gin_ibv_is_dmabuf_supported(struct ibv_pd* pd) {
  return wrap_ibv_is_dmabuf_supported(pd);
}
