/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Implementation of the src/misc/ibvwrap.cc fakes. See ibvwrap_fakes.h.

#include "ibvwrap_fakes.h"

#include <cerrno>

#include "ibvwrap.h"
#include "signature-drift.h"

static ncclResult_t DefaultIbvSymbols() { return ncclSystemError; }
std::function<ncclResult_t()> g_ibvSymbols = DefaultIbvSymbols;

static ncclResult_t DefaultIbvForkInit() { return ncclSystemError; }
std::function<ncclResult_t()> g_ibvForkInit = DefaultIbvForkInit;

static ncclResult_t DefaultIbvGetDeviceList(struct ibv_device*** devices, int* count) {
  if (devices) *devices = nullptr;
  if (count) *count = 0;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_device***, int*)> g_ibvGetDeviceList = DefaultIbvGetDeviceList;

static ncclResult_t DefaultIbvFreeDeviceList(struct ibv_device**) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_device**)> g_ibvFreeDeviceList = DefaultIbvFreeDeviceList;

static const char* DefaultIbvGetDeviceName(struct ibv_device*) { return "micro-ib"; }
std::function<const char*(struct ibv_device*)> g_ibvGetDeviceName = DefaultIbvGetDeviceName;

static ncclResult_t DefaultIbvOpenDevice(struct ibv_context** context, struct ibv_device*) {
  if (context) *context = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context**, struct ibv_device*)> g_ibvOpenDevice = DefaultIbvOpenDevice;

static ncclResult_t DefaultIbvCloseDevice(struct ibv_context*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_context*)> g_ibvCloseDevice = DefaultIbvCloseDevice;

static ncclResult_t DefaultIbvGetAsyncEvent(struct ibv_context*, struct ibv_async_event*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, struct ibv_async_event*)> g_ibvGetAsyncEvent =
    DefaultIbvGetAsyncEvent;

static ncclResult_t DefaultIbvAckAsyncEvent(struct ibv_async_event*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_async_event*)> g_ibvAckAsyncEvent = DefaultIbvAckAsyncEvent;

static ncclResult_t DefaultIbvQueryDevice(struct ibv_context*, struct ibv_device_attr*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, struct ibv_device_attr*)> g_ibvQueryDevice =
    DefaultIbvQueryDevice;

static ncclResult_t DefaultIbvQueryPort(struct ibv_context*, uint8_t, struct ibv_port_attr*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, struct ibv_port_attr*)> g_ibvQueryPort =
    DefaultIbvQueryPort;

static ncclResult_t DefaultIbvQueryGid(struct ibv_context*, uint8_t, int, union ibv_gid*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, int, union ibv_gid*)> g_ibvQueryGid =
    DefaultIbvQueryGid;

static ncclResult_t DefaultIbvQueryPkey(struct ibv_context*, uint8_t, int, uint16_t*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, int, uint16_t*)> g_ibvQueryPkey =
    DefaultIbvQueryPkey;

static ncclResult_t DefaultIbvQueryQp(struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int, struct ibv_qp_init_attr*)> g_ibvQueryQp =
    DefaultIbvQueryQp;

static ncclResult_t DefaultIbvAllocPd(struct ibv_pd** ret, struct ibv_context*) {
  if (ret) *ret = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_pd**, struct ibv_context*)> g_ibvAllocPd = DefaultIbvAllocPd;

static ncclResult_t DefaultIbvDeallocPd(struct ibv_pd*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_pd*)> g_ibvDeallocPd = DefaultIbvDeallocPd;

static ncclResult_t DefaultIbvRegMr(struct ibv_mr** mr, struct ibv_pd*, void*, size_t, int) {
  if (mr) *mr = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, int)> g_ibvRegMr = DefaultIbvRegMr;

static ncclResult_t DefaultIbvRegMrIova2(struct ibv_mr** mr, struct ibv_pd*, void*, size_t, uint64_t, int) {
  if (mr) *mr = nullptr;
  return ncclInternalError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, void*, size_t, uint64_t, int)> g_ibvRegMrIova2 =
    DefaultIbvRegMrIova2;

static ncclResult_t DefaultIbvRegDmaBufMr(struct ibv_mr** mr, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
  if (mr) *mr = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_mr**, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> g_ibvRegDmaBufMr =
    DefaultIbvRegDmaBufMr;

static struct ibv_mr* DefaultIbvDirectRegMr(struct ibv_pd*, void*, size_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, int)> g_ibvDirectRegMr = DefaultIbvDirectRegMr;

static struct ibv_mr* DefaultIbvDirectRegMrIova2(struct ibv_pd*, void*, size_t, uint64_t, int) { return nullptr; }
std::function<struct ibv_mr*(struct ibv_pd*, void*, size_t, uint64_t, int)> g_ibvDirectRegMrIova2 =
    DefaultIbvDirectRegMrIova2;

static struct ibv_mr* DefaultIbvDirectRegDmaBufMr(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
  errno = EOPNOTSUPP;
  return nullptr;
}
std::function<struct ibv_mr*(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int)> g_ibvDirectRegDmaBufMr =
    DefaultIbvDirectRegDmaBufMr;

static ncclResult_t DefaultIbvDeregMr(struct ibv_mr*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_mr*)> g_ibvDeregMr = DefaultIbvDeregMr;

static ncclResult_t DefaultIbvCreateCq(struct ibv_cq** cq, struct ibv_context*, int, void*,
                                       struct ibv_comp_channel*, int) {
  if (cq) *cq = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_cq**, struct ibv_context*, int, void*, struct ibv_comp_channel*, int)>
    g_ibvCreateCq = DefaultIbvCreateCq;

static ncclResult_t DefaultIbvDestroyCq(struct ibv_cq*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_cq*)> g_ibvDestroyCq = DefaultIbvDestroyCq;

static ncclResult_t DefaultIbvCreateQp(struct ibv_qp** qp, struct ibv_pd*, struct ibv_qp_init_attr*) {
  if (qp) *qp = nullptr;
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp**, struct ibv_pd*, struct ibv_qp_init_attr*)> g_ibvCreateQp =
    DefaultIbvCreateQp;

static ncclResult_t DefaultIbvModifyQp(struct ibv_qp*, struct ibv_qp_attr*, int) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp*, struct ibv_qp_attr*, int)> g_ibvModifyQp = DefaultIbvModifyQp;

static ncclResult_t DefaultIbvDestroyQp(struct ibv_qp*) { return ncclSuccess; }
std::function<ncclResult_t(struct ibv_qp*)> g_ibvDestroyQp = DefaultIbvDestroyQp;

static ncclResult_t DefaultIbvQueryEce(struct ibv_qp*, struct ibv_ece*, int*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp*, struct ibv_ece*, int*)> g_ibvQueryEce = DefaultIbvQueryEce;

static ncclResult_t DefaultIbvSetEce(struct ibv_qp*, struct ibv_ece*, int*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_qp*, struct ibv_ece*, int*)> g_ibvSetEce = DefaultIbvSetEce;

static ncclResult_t DefaultIbvQueryPortSpeed(struct ibv_context*, uint8_t, uint64_t*) {
  return ncclSystemError;
}
std::function<ncclResult_t(struct ibv_context*, uint8_t, uint64_t*)> g_ibvQueryPortSpeed =
    DefaultIbvQueryPortSpeed;

static ncclResult_t DefaultIbvEventTypeStr(char**, enum ibv_event_type) { return ncclSystemError; }
std::function<ncclResult_t(char**, enum ibv_event_type)> g_ibvEventTypeStr = DefaultIbvEventTypeStr;

static const char* DefaultIbvWcStatusStr(enum ibv_wc_status) { return "micro-wc-status"; }
static const char* DefaultIbvWcOpcodeStr(enum ibv_wc_opcode) { return "micro-wc-opcode"; }
static const char* DefaultIbvWrOpcodeStr(enum ibv_wr_opcode) { return "micro-wr-opcode"; }
std::function<const char*(enum ibv_wc_status)> g_ibvWcStatusStr = DefaultIbvWcStatusStr;
std::function<const char*(enum ibv_wc_opcode)> g_ibvWcOpcodeStr = DefaultIbvWcOpcodeStr;
std::function<const char*(enum ibv_wr_opcode)> g_ibvWrOpcodeStr = DefaultIbvWrOpcodeStr;

ASSERT_HOOK_MATCHES_PROD(g_ibvSymbols, wrap_ibv_symbols);
ASSERT_HOOK_MATCHES_PROD(g_ibvForkInit, wrap_ibv_fork_init);
ASSERT_HOOK_MATCHES_PROD(g_ibvGetDeviceList, wrap_ibv_get_device_list);
ASSERT_HOOK_MATCHES_PROD(g_ibvFreeDeviceList, wrap_ibv_free_device_list);
ASSERT_HOOK_MATCHES_PROD(g_ibvGetDeviceName, wrap_ibv_get_device_name);
ASSERT_HOOK_MATCHES_PROD(g_ibvOpenDevice, wrap_ibv_open_device);
ASSERT_HOOK_MATCHES_PROD(g_ibvCloseDevice, wrap_ibv_close_device);
ASSERT_HOOK_MATCHES_PROD(g_ibvGetAsyncEvent, wrap_ibv_get_async_event);
ASSERT_HOOK_MATCHES_PROD(g_ibvAckAsyncEvent, wrap_ibv_ack_async_event);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryDevice, wrap_ibv_query_device);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryPort, wrap_ibv_query_port);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryGid, wrap_ibv_query_gid);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryPkey, wrap_ibv_query_pkey);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryQp, wrap_ibv_query_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvAllocPd, wrap_ibv_alloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeallocPd, wrap_ibv_dealloc_pd);
ASSERT_HOOK_MATCHES_PROD(g_ibvRegMr, wrap_ibv_reg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvRegMrIova2, wrap_ibv_reg_mr_iova2);
ASSERT_HOOK_MATCHES_PROD(g_ibvRegDmaBufMr, wrap_ibv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMr, wrap_direct_ibv_reg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegMrIova2, wrap_direct_ibv_reg_mr_iova2);
ASSERT_HOOK_MATCHES_PROD(g_ibvDirectRegDmaBufMr, wrap_direct_ibv_reg_dmabuf_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvDeregMr, wrap_ibv_dereg_mr);
ASSERT_HOOK_MATCHES_PROD(g_ibvCreateCq, wrap_ibv_create_cq);
ASSERT_HOOK_MATCHES_PROD(g_ibvDestroyCq, wrap_ibv_destroy_cq);
ASSERT_HOOK_MATCHES_PROD(g_ibvCreateQp, wrap_ibv_create_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvModifyQp, wrap_ibv_modify_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvDestroyQp, wrap_ibv_destroy_qp);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryEce, wrap_ibv_query_ece);
ASSERT_HOOK_MATCHES_PROD(g_ibvSetEce, wrap_ibv_set_ece);
ASSERT_HOOK_MATCHES_PROD(g_ibvQueryPortSpeed, wrap_ibv_query_port_speed);
ASSERT_HOOK_MATCHES_PROD(g_ibvEventTypeStr, wrap_ibv_event_type_str);
ASSERT_HOOK_MATCHES_PROD(g_ibvWcStatusStr, ibvWcStatusStr);
ASSERT_HOOK_MATCHES_PROD(g_ibvWcOpcodeStr, ibvWcOpcodeStr);
ASSERT_HOOK_MATCHES_PROD(g_ibvWrOpcodeStr, ibvWrOpcodeStr);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t wrap_ibv_symbols() { return g_ibvSymbols(); }
ncclResult_t wrap_ibv_fork_init() { return g_ibvForkInit(); }
ncclResult_t wrap_ibv_get_device_list(struct ibv_device*** devices, int* count) {
  return g_ibvGetDeviceList(devices, count);
}
ncclResult_t wrap_ibv_free_device_list(struct ibv_device** devices) {
  return g_ibvFreeDeviceList(devices);
}
const char* wrap_ibv_get_device_name(struct ibv_device* device) { return g_ibvGetDeviceName(device); }
ncclResult_t wrap_ibv_open_device(struct ibv_context** context, struct ibv_device* device) {
  return g_ibvOpenDevice(context, device);
}
ncclResult_t wrap_ibv_close_device(struct ibv_context* context) { return g_ibvCloseDevice(context); }
ncclResult_t wrap_ibv_get_async_event(struct ibv_context* context, struct ibv_async_event* event) {
  return g_ibvGetAsyncEvent(context, event);
}
ncclResult_t wrap_ibv_ack_async_event(struct ibv_async_event* event) {
  return g_ibvAckAsyncEvent(event);
}
ncclResult_t wrap_ibv_query_device(struct ibv_context* context, struct ibv_device_attr* attr) {
  return g_ibvQueryDevice(context, attr);
}
ncclResult_t wrap_ibv_query_port(struct ibv_context* context, uint8_t port, struct ibv_port_attr* attr) {
  return g_ibvQueryPort(context, port, attr);
}
ncclResult_t wrap_ibv_query_gid(struct ibv_context* context, uint8_t port, int index, union ibv_gid* gid) {
  return g_ibvQueryGid(context, port, index, gid);
}
ncclResult_t wrap_ibv_query_pkey(struct ibv_context* context, uint8_t port, int index, uint16_t* pkey) {
  return g_ibvQueryPkey(context, port, index, pkey);
}
ncclResult_t wrap_ibv_query_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int mask,
                               struct ibv_qp_init_attr* initAttr) {
  return g_ibvQueryQp(qp, attr, mask, initAttr);
}
ncclResult_t wrap_ibv_alloc_pd(struct ibv_pd** ret, struct ibv_context* context) {
  return g_ibvAllocPd(ret, context);
}
ncclResult_t wrap_ibv_dealloc_pd(struct ibv_pd* pd) { return g_ibvDeallocPd(pd); }
ncclResult_t wrap_ibv_reg_mr(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, int access) {
  return g_ibvRegMr(mr, pd, addr, length, access);
}
ncclResult_t wrap_ibv_reg_mr_iova2(struct ibv_mr** mr, struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                   int access) {
  return g_ibvRegMrIova2(mr, pd, addr, length, iova, access);
}
ncclResult_t wrap_ibv_reg_dmabuf_mr(struct ibv_mr** mr, struct ibv_pd* pd, uint64_t offset, size_t length,
                                    uint64_t iova, int fd, int access) {
  return g_ibvRegDmaBufMr(mr, pd, offset, length, iova, fd, access);
}
struct ibv_mr* wrap_direct_ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access) {
  return g_ibvDirectRegMr(pd, addr, length, access);
}
struct ibv_mr* wrap_direct_ibv_reg_mr_iova2(struct ibv_pd* pd, void* addr, size_t length, uint64_t iova,
                                            int access) {
  return g_ibvDirectRegMrIova2(pd, addr, length, iova, access);
}
struct ibv_mr* wrap_direct_ibv_reg_dmabuf_mr(struct ibv_pd* pd, uint64_t offset, size_t length, uint64_t iova,
                                             int fd, int access) {
  return g_ibvDirectRegDmaBufMr(pd, offset, length, iova, fd, access);
}
ncclResult_t wrap_ibv_dereg_mr(struct ibv_mr* mr) { return g_ibvDeregMr(mr); }
ncclResult_t wrap_ibv_create_cq(struct ibv_cq** cq, struct ibv_context* context, int cqe, void* cqContext,
                                struct ibv_comp_channel* channel, int vector) {
  return g_ibvCreateCq(cq, context, cqe, cqContext, channel, vector);
}
ncclResult_t wrap_ibv_destroy_cq(struct ibv_cq* cq) { return g_ibvDestroyCq(cq); }
ncclResult_t wrap_ibv_create_qp(struct ibv_qp** qp, struct ibv_pd* pd, struct ibv_qp_init_attr* attr) {
  return g_ibvCreateQp(qp, pd, attr);
}
ncclResult_t wrap_ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int mask) {
  return g_ibvModifyQp(qp, attr, mask);
}
ncclResult_t wrap_ibv_destroy_qp(struct ibv_qp* qp) { return g_ibvDestroyQp(qp); }
ncclResult_t wrap_ibv_query_ece(struct ibv_qp* qp, struct ibv_ece* ece, int* supported) {
  return g_ibvQueryEce(qp, ece, supported);
}
ncclResult_t wrap_ibv_set_ece(struct ibv_qp* qp, struct ibv_ece* ece, int* supported) {
  return g_ibvSetEce(qp, ece, supported);
}
ncclResult_t wrap_ibv_query_port_speed(struct ibv_context* context, uint8_t port, uint64_t* speed) {
  return g_ibvQueryPortSpeed(context, port, speed);
}
ncclResult_t wrap_ibv_event_type_str(char** text, enum ibv_event_type event) {
  return g_ibvEventTypeStr(text, event);
}
const char* ibvWcStatusStr(enum ibv_wc_status status) { return g_ibvWcStatusStr(status); }
const char* ibvWcOpcodeStr(enum ibv_wc_opcode opcode) { return g_ibvWcOpcodeStr(opcode); }
const char* ibvWrOpcodeStr(enum ibv_wr_opcode opcode) { return g_ibvWrOpcodeStr(opcode); }

void ResetIbvwrapFakes() {
  g_ibvSymbols = DefaultIbvSymbols;
  g_ibvForkInit = DefaultIbvForkInit;
  g_ibvGetDeviceList = DefaultIbvGetDeviceList;
  g_ibvFreeDeviceList = DefaultIbvFreeDeviceList;
  g_ibvGetDeviceName = DefaultIbvGetDeviceName;
  g_ibvOpenDevice = DefaultIbvOpenDevice;
  g_ibvCloseDevice = DefaultIbvCloseDevice;
  g_ibvGetAsyncEvent = DefaultIbvGetAsyncEvent;
  g_ibvAckAsyncEvent = DefaultIbvAckAsyncEvent;
  g_ibvQueryDevice = DefaultIbvQueryDevice;
  g_ibvQueryPort = DefaultIbvQueryPort;
  g_ibvQueryGid = DefaultIbvQueryGid;
  g_ibvQueryPkey = DefaultIbvQueryPkey;
  g_ibvQueryQp = DefaultIbvQueryQp;
  g_ibvAllocPd = DefaultIbvAllocPd;
  g_ibvDeallocPd = DefaultIbvDeallocPd;
  g_ibvRegMr = DefaultIbvRegMr;
  g_ibvRegMrIova2 = DefaultIbvRegMrIova2;
  g_ibvRegDmaBufMr = DefaultIbvRegDmaBufMr;
  g_ibvDirectRegMr = DefaultIbvDirectRegMr;
  g_ibvDirectRegMrIova2 = DefaultIbvDirectRegMrIova2;
  g_ibvDirectRegDmaBufMr = DefaultIbvDirectRegDmaBufMr;
  g_ibvDeregMr = DefaultIbvDeregMr;
  g_ibvCreateCq = DefaultIbvCreateCq;
  g_ibvDestroyCq = DefaultIbvDestroyCq;
  g_ibvCreateQp = DefaultIbvCreateQp;
  g_ibvModifyQp = DefaultIbvModifyQp;
  g_ibvDestroyQp = DefaultIbvDestroyQp;
  g_ibvQueryEce = DefaultIbvQueryEce;
  g_ibvSetEce = DefaultIbvSetEce;
  g_ibvQueryPortSpeed = DefaultIbvQueryPortSpeed;
  g_ibvEventTypeStr = DefaultIbvEventTypeStr;
  g_ibvWcStatusStr = DefaultIbvWcStatusStr;
  g_ibvWcOpcodeStr = DefaultIbvWcOpcodeStr;
  g_ibvWrOpcodeStr = DefaultIbvWrOpcodeStr;
}
