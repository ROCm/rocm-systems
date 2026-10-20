/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/ibvwrap.cc fakes. See ibvwrap_fakes.h.

#include "ibvwrap_fakes.h"

#include "ibvwrap.h"
#include "signature-drift.h"

static ncclResult_t DefaultIbvAllocPd(struct ibv_pd** ret, struct ibv_context*) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_pd**, struct ibv_context*)> g_ibvAllocPd = DefaultIbvAllocPd;

static ncclResult_t DefaultIbvDeallocPd(struct ibv_pd*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_pd*)> g_ibvDeallocPd = DefaultIbvDeallocPd;

static struct ibv_mr* DefaultIbvDirectRegMr(struct ibv_pd*, void*, size_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, int)> g_ibvDirectRegMr = DefaultIbvDirectRegMr;

static struct ibv_mr* DefaultIbvDirectRegMrIova2(struct ibv_pd*, void*, size_t, uint64_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, uint64_t, int)> g_ibvDirectRegMrIova2 =
    DefaultIbvDirectRegMrIova2;

static ncclResult_t DefaultIbvDeregMr(struct ibv_mr*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_mr*)> g_ibvDeregMr = DefaultIbvDeregMr;

static ncclResult_t DefaultIbvRegMr(struct ibv_mr** ret, struct ibv_pd*, void*, size_t, int) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, int)> g_ibvRegMr = DefaultIbvRegMr;

static ncclResult_t DefaultIbvRegDmabufMr(struct ibv_mr** ret, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> g_ibvRegDmabufMr =
    DefaultIbvRegDmabufMr;

static ncclResult_t DefaultIbvCreateCq(struct ibv_cq** ret, struct ibv_context*, int, void*, struct ibv_comp_channel*,
                                       int) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_cq**, struct ibv_context*, int, void*, struct ibv_comp_channel*, int)>
    g_ibvCreateCq = DefaultIbvCreateCq;

static ncclResult_t DefaultIbvDestroyCq(struct ibv_cq*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_cq*)> g_ibvDestroyCq = DefaultIbvDestroyCq;

static ncclResult_t DefaultIbvCreateQp(struct ibv_qp** ret, struct ibv_pd*, struct ibv_qp_init_attr*) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp**, struct ibv_pd*, struct ibv_qp_init_attr*)> g_ibvCreateQp =
    DefaultIbvCreateQp;

static ncclResult_t DefaultIbvModifyQp(struct ibv_qp*, struct ibv_qp_attr*, int) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int)> g_ibvModifyQp = DefaultIbvModifyQp;

static ncclResult_t DefaultIbvDestroyQp(struct ibv_qp*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_qp*)> g_ibvDestroyQp = DefaultIbvDestroyQp;

static ncclResult_t DefaultIbvQueryGid(struct ibv_context*, uint8_t, int, union ibv_gid*) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_context*, uint8_t, int, union ibv_gid*)> g_ibvQueryGid = DefaultIbvQueryGid;

static ncclResult_t DefaultIbvQueryEce(struct ibv_qp*, struct ibv_ece*, int*) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_qp*, struct ibv_ece*, int*)> g_ibvQueryEce = DefaultIbvQueryEce;

static ncclResult_t DefaultIbvSetEce(struct ibv_qp*, struct ibv_ece*, int*) { return ncclSystemError; }
std::function<ncclResult_t(struct ibv_qp*, struct ibv_ece*, int*)> g_ibvSetEce = DefaultIbvSetEce;

static const char* DefaultIbvGetDeviceName(struct ibv_device*) { return nullptr; }
std::function<const char*(struct ibv_device*)> g_ibvGetDeviceName = DefaultIbvGetDeviceName;

ASSERT_HOOK_MATCHES_PROD(g_ibvAllocPd, wrap_ibv_alloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeallocPd, wrap_ibv_dealloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMr, wrap_direct_ibv_reg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMrIova2, wrap_direct_ibv_reg_mr_iova2);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeregMr, wrap_ibv_dereg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvRegMr, wrap_ibv_reg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvRegDmabufMr, wrap_ibv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvCreateCq, wrap_ibv_create_cq);
ASSERT_HOOK_MATCHES_PROD(g_ibvDestroyCq, wrap_ibv_destroy_cq);
ASSERT_HOOK_MATCHES_PROD(g_ibvCreateQp, wrap_ibv_create_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvModifyQp, wrap_ibv_modify_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvDestroyQp, wrap_ibv_destroy_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryGid, wrap_ibv_query_gid);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryEce, wrap_ibv_query_ece);
ASSERT_HOOK_MATCHES_PROD(g_ibvSetEce, wrap_ibv_set_ece);
ASSERT_HOOK_MATCHES_PROD(g_ibvGetDeviceName, wrap_ibv_get_device_name);

ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context* context) {
  return g_ibvAllocPd(ret, context);
}
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd* pd) { return g_ibvDeallocPd(pd); }
struct ibv_mr* wrap_direct_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return g_ibvDirectRegMr(pd, addr, length, access);
}
struct ibv_mr* wrap_direct_ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                            int access) {
  return g_ibvDirectRegMrIova2(pd, addr, length, iova, access);
}
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr* mr) { return g_ibvDeregMr(mr); }
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr** ret, struct ibv_pd* pd, void* addr, size_t length, int access) {
  return g_ibvRegMr(ret, pd, addr, length, access);
}
ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr** ret, struct ibv_pd* pd, uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access) {
  return g_ibvRegDmabufMr(ret, pd, offset, length, iova, fd, access);
}
ncclResult_t wrap_ibv_create_cq(struct ibv_cq** ret, struct ibv_context* context, int cqe, void* cq_context,
                                struct ibv_comp_channel* channel, int comp_vector) {
  return g_ibvCreateCq(ret, context, cqe, cq_context, channel, comp_vector);
}
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq* cq) { return g_ibvDestroyCq(cq); }
ncclResult_t wrap_ibv_create_qp(struct ibv_qp** ret, struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr) {
  return g_ibvCreateQp(ret, pd, qp_init_attr);
}
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask) {
  return g_ibvModifyQp(qp, attr, attr_mask);
}
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp* qp) { return g_ibvDestroyQp(qp); }
ncclResult_t wrap_ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid) {
  return g_ibvQueryGid(context, port_num, index, gid);
}
ncclResult_t wrap_ibv_query_ece(struct ibv_qp* qp, struct ibv_ece* ece, int* supported) {
  return g_ibvQueryEce(qp, ece, supported);
}
ncclResult_t wrap_ibv_set_ece(struct ibv_qp* qp, struct ibv_ece* ece, int* supported) {
  return g_ibvSetEce(qp, ece, supported);
}
const char* wrap_ibv_get_device_name(struct ibv_device* device) { return g_ibvGetDeviceName(device); }

// Plain string helpers: pure functions of their argument, nothing to control.
const char* ibvWcStatusStr(enum ibv_wc_status status) {
  return status == IBV_WC_SUCCESS ? "IBV_WC_SUCCESS" : "IBV_WC_ERROR(fake)";
}
const char* ibvWcOpcodeStr(enum ibv_wc_opcode) { return "IBV_WC_OPCODE(fake)"; }

void ResetIbvwrapFakes() {
  g_ibvAllocPd = DefaultIbvAllocPd;
  g_ibvDeallocPd = DefaultIbvDeallocPd;
  g_ibvDirectRegMr = DefaultIbvDirectRegMr;
  g_ibvDirectRegMrIova2 = DefaultIbvDirectRegMrIova2;
  g_ibvDeregMr = DefaultIbvDeregMr;
  g_ibvRegMr = DefaultIbvRegMr;
  g_ibvRegDmabufMr = DefaultIbvRegDmabufMr;
  g_ibvCreateCq = DefaultIbvCreateCq;
  g_ibvDestroyCq = DefaultIbvDestroyCq;
  g_ibvCreateQp = DefaultIbvCreateQp;
  g_ibvModifyQp = DefaultIbvModifyQp;
  g_ibvDestroyQp = DefaultIbvDestroyQp;
  g_ibvQueryGid = DefaultIbvQueryGid;
  g_ibvQueryEce = DefaultIbvQueryEce;
  g_ibvSetEce = DefaultIbvSetEce;
  g_ibvGetDeviceName = DefaultIbvGetDeviceName;
}
