/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/ibvwrap.cc and the inline wrappers in
 * src/include/ibvwrap.h.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cerrno>
#include <chrono>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "ScopedHook.h"
#include "fakes/dl_fakes.h"
#include "fakes/env_fakes.h"
#include "fakes/nccl_fakes.h"
#include "fakes/net_ib_ops_fault_fakes.h"

#include "fakes/param_redirect.h"
#include IBVWRAP_CC_PATH

namespace {

// The wrappers only forward these, so they are never dereferenced.
ibv_context* const kCtx = reinterpret_cast<ibv_context*>(0xC0);
ibv_pd* const kPd = reinterpret_cast<ibv_pd*>(0xD0);
ibv_qp* const kQp = reinterpret_cast<ibv_qp*>(0xE0);
ibv_cq* const kCq = reinterpret_cast<ibv_cq*>(0xF0);
ibv_mr* const kMr = reinterpret_cast<ibv_mr*>(0x100);
ibv_device* const kDevice = reinterpret_cast<ibv_device*>(0x110);
ibv_comp_channel* const kChannel = reinterpret_cast<ibv_comp_channel*>(0x120);
void* const kAddr = reinterpret_cast<void*>(0x130);

int g_ret;          // what an int-returning fake verb returns
void* g_returned;   // what an object-returning fake verb returns
void* g_out;        // what the wrapper wrote to its *ret
int g_calls;

ibv_device_attr g_deviceAttr;
ibv_qp_attr g_qpAttr;
ibv_qp_init_attr g_qpInitAttr;
ibv_async_event g_event;
ibv_ece g_ece;
union ibv_gid g_gid;
uint16_t g_pkey;
ibv_device* g_deviceList[2] = {kDevice, nullptr};

class IbvwrapMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_ = ibvSymbols;
    std::memset(&ibvSymbols, 0, sizeof(ibvSymbols));
    g_ret = 0;
    g_returned = kAddr;
    g_out = nullptr;
    g_calls = 0;
  }
  void TearDown() override {
    ibvSymbols = saved_;
    ResetNcclFakes();
    ResetNetIbOpsFaultFakes();
  }

 private:
  ncclIbvSymbols saved_;
};

// One wrapper and the fake verb behind it; the fake checks the arguments it was forwarded.
struct WrapCase {
  const char* name;
  void (*install)();
  ncclResult_t (*call)();
};

template <typename Case>
std::string CaseName(const ::testing::TestParamInfo<Case>& info) {
  return info.param.name;
}

// Runs an object-returning wrapper with a stale *ret and records what it wrote there.
template <typename T, typename Call>
ncclResult_t CallWithOut(Call call) {
  T* out = reinterpret_cast<T*>(0xBAD);
  ncclResult_t res = call(&out);
  g_out = out;
  return res;
}

void InstallRegMr() {
  ibvSymbols.ibv_internal_reg_mr = [](ibv_pd* pd, void* addr, size_t len, int access) {
    EXPECT_EQ(pd, kPd);
    EXPECT_EQ(addr, kAddr);
    EXPECT_EQ(len, 4096u);
    EXPECT_EQ(access, IBV_ACCESS_LOCAL_WRITE);
    ++g_calls;
    return static_cast<ibv_mr*>(g_returned);
  };
}

void InstallRegMrIova2() {
  ibvSymbols.ibv_internal_reg_mr_iova2 = [](ibv_pd* pd, void* addr, size_t len, uint64_t iova, unsigned int access) {
    EXPECT_EQ(pd, kPd);
    EXPECT_EQ(addr, kAddr);
    EXPECT_EQ(len, 4096u);
    EXPECT_EQ(iova, 0x5000u);
    EXPECT_EQ(access, unsigned(IBV_ACCESS_REMOTE_READ));
    ++g_calls;
    return static_cast<ibv_mr*>(g_returned);
  };
}

void InstallRegDmabufMr() {
  ibvSymbols.ibv_internal_reg_dmabuf_mr = [](ibv_pd* pd, uint64_t offset, size_t len, uint64_t iova, int fd,
                                             int access) {
    EXPECT_EQ(pd, kPd);
    EXPECT_EQ(offset, 0x40u);
    EXPECT_EQ(len, 4096u);
    EXPECT_EQ(iova, 0x5000u);
    EXPECT_EQ(fd, 7);
    EXPECT_EQ(access, IBV_ACCESS_REMOTE_WRITE);
    ++g_calls;
    return static_cast<ibv_mr*>(g_returned);
  };
}

// --- Verbs that return 0 on success and an errno value on failure ---

const WrapCase kErrnoCases[] = {
    {"query_device",
     [] {
       ibvSymbols.ibv_internal_query_device = [](ibv_context* c, ibv_device_attr* a) {
         EXPECT_EQ(c, kCtx);
         EXPECT_EQ(a, &g_deviceAttr);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_query_device(kCtx, &g_deviceAttr); }},
    {"query_gid",
     [] {
       ibvSymbols.ibv_internal_query_gid = [](ibv_context* c, uint8_t port, int index, union ibv_gid* gid) {
         EXPECT_EQ(c, kCtx);
         EXPECT_EQ(port, 2);
         EXPECT_EQ(index, 3);
         EXPECT_EQ(gid, &g_gid);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_query_gid(kCtx, 2, 3, &g_gid); }},
    {"query_pkey",
     [] {
       ibvSymbols.ibv_internal_query_pkey = [](ibv_context* c, uint8_t port, int index, uint16_t* pkey) {
         EXPECT_EQ(c, kCtx);
         EXPECT_EQ(port, 2);
         EXPECT_EQ(index, 3);
         EXPECT_EQ(pkey, &g_pkey);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_query_pkey(kCtx, 2, 3, &g_pkey); }},
    {"query_qp",
     [] {
       ibvSymbols.ibv_internal_query_qp = [](ibv_qp* qp, ibv_qp_attr* attr, int mask, ibv_qp_init_attr* init) {
         EXPECT_EQ(qp, kQp);
         EXPECT_EQ(attr, &g_qpAttr);
         EXPECT_EQ(mask, IBV_QP_STATE);
         EXPECT_EQ(init, &g_qpInitAttr);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_query_qp(kQp, &g_qpAttr, IBV_QP_STATE, &g_qpInitAttr); }},
    {"dealloc_pd",
     [] {
       ibvSymbols.ibv_internal_dealloc_pd = [](ibv_pd* pd) {
         EXPECT_EQ(pd, kPd);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_dealloc_pd(kPd); }},
    {"dereg_mr",
     [] {
       ibvSymbols.ibv_internal_dereg_mr = [](ibv_mr* mr) {
         EXPECT_EQ(mr, kMr);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_dereg_mr(kMr); }},
    {"destroy_cq",
     [] {
       ibvSymbols.ibv_internal_destroy_cq = [](ibv_cq* cq) {
         EXPECT_EQ(cq, kCq);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_destroy_cq(kCq); }},
    {"destroy_qp",
     [] {
       ibvSymbols.ibv_internal_destroy_qp = [](ibv_qp* qp) {
         EXPECT_EQ(qp, kQp);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_destroy_qp(kQp); }},
};

class IbvwrapErrnoReturnMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<WrapCase> {};

TEST_P(IbvwrapErrnoReturnMicrotest, VerbReturnsZero_Succeeds) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapErrnoReturnMicrotest, VerbReturnsErrno_ReturnsSystemError) {
  GetParam().install();
  g_ret = EINVAL;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
}

TEST_P(IbvwrapErrnoReturnMicrotest, VerbReturnsMinusOne_ReturnsSystemError) {
  GetParam().install();
  g_ret = -1;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
}

TEST_P(IbvwrapErrnoReturnMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(GetParam().call(), ncclInternalError);
}

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapErrnoReturnMicrotest, ::testing::ValuesIn(kErrnoCases), CaseName<WrapCase>);

// --- Verbs that return -1 on failure and 0 or a count otherwise ---

const WrapCase kMinusOneCases[] = {
    {"close_device",
     [] {
       ibvSymbols.ibv_internal_close_device = [](ibv_context* c) {
         EXPECT_EQ(c, kCtx);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_close_device(kCtx); }},
    {"get_async_event",
     [] {
       ibvSymbols.ibv_internal_get_async_event = [](ibv_context* c, ibv_async_event* e) {
         EXPECT_EQ(c, kCtx);
         EXPECT_EQ(e, &g_event);
         ++g_calls;
         return g_ret;
       };
     },
     [] { return wrap_ibv_get_async_event(kCtx, &g_event); }},
};

class IbvwrapMinusOneMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<WrapCase> {};

TEST_P(IbvwrapMinusOneMicrotest, VerbReturnsZero_Succeeds) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapMinusOneMicrotest, VerbReturnsMinusOne_ReturnsSystemError) {
  GetParam().install();
  g_ret = -1;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
}

TEST_P(IbvwrapMinusOneMicrotest, VerbReturnsPositive_Succeeds) {
  GetParam().install();
  g_ret = 1;
  EXPECT_EQ(GetParam().call(), ncclSuccess);
}

TEST_P(IbvwrapMinusOneMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(GetParam().call(), ncclInternalError);
}

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapMinusOneMicrotest, ::testing::ValuesIn(kMinusOneCases), CaseName<WrapCase>);

// ibv_fork_init returns an errno value on failure; only its 0 and -1 results are pinned here.
class IbvwrapForkInitMicrotest : public IbvwrapMicrotest {
 protected:
  void SetUp() override {
    IbvwrapMicrotest::SetUp();
    ibvSymbols.ibv_internal_fork_init = []() {
      ++g_calls;
      return g_ret;
    };
  }
};

TEST_F(IbvwrapForkInitMicrotest, VerbReturnsZero_Succeeds) {
  EXPECT_EQ(wrap_ibv_fork_init(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapForkInitMicrotest, VerbReturnsMinusOne_ReturnsSystemError) {
  g_ret = -1;
  EXPECT_EQ(wrap_ibv_fork_init(), ncclSystemError);
}

TEST_F(IbvwrapForkInitMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  ibvSymbols.ibv_internal_fork_init = nullptr;
  EXPECT_EQ(wrap_ibv_fork_init(), ncclInternalError);
}

// --- Verbs that return an object, or NULL on failure ---

ibv_qp_init_attr g_createQpAttr;

const WrapCase kObjectCases[] = {
    {"open_device",
     [] {
       ibvSymbols.ibv_internal_open_device = [](ibv_device* d) {
         EXPECT_EQ(d, kDevice);
         ++g_calls;
         return static_cast<ibv_context*>(g_returned);
       };
     },
     [] { return CallWithOut<ibv_context>([](ibv_context** out) { return wrap_ibv_open_device(out, kDevice); }); }},
    {"alloc_pd",
     [] {
       ibvSymbols.ibv_internal_alloc_pd = [](ibv_context* c) {
         EXPECT_EQ(c, kCtx);
         ++g_calls;
         return static_cast<ibv_pd*>(g_returned);
       };
     },
     [] { return CallWithOut<ibv_pd>([](ibv_pd** out) { return wrap_ibv_alloc_pd(out, kCtx); }); }},
    {"reg_mr", InstallRegMr,
     [] {
       return CallWithOut<ibv_mr>(
           [](ibv_mr** out) { return wrap_ibv_reg_mr(out, kPd, kAddr, 4096, IBV_ACCESS_LOCAL_WRITE); });
     }},
    {"reg_mr_iova2", InstallRegMrIova2,
     [] {
       return CallWithOut<ibv_mr>([](ibv_mr** out) {
         return wrap_ibv_reg_mr_iova2(out, kPd, kAddr, 4096, 0x5000, IBV_ACCESS_REMOTE_READ);
       });
     }},
    {"reg_dmabuf_mr", InstallRegDmabufMr,
     [] {
       return CallWithOut<ibv_mr>([](ibv_mr** out) {
         return wrap_ibv_reg_dmabuf_mr(out, kPd, 0x40, 4096, 0x5000, 7, IBV_ACCESS_REMOTE_WRITE);
       });
     }},
    {"create_cq",
     [] {
       ibvSymbols.ibv_internal_create_cq = [](ibv_context* c, int cqe, void* cqContext, ibv_comp_channel* channel,
                                              int vector) {
         EXPECT_EQ(c, kCtx);
         EXPECT_EQ(cqe, 64);
         EXPECT_EQ(cqContext, kAddr);
         EXPECT_EQ(channel, kChannel);
         EXPECT_EQ(vector, 3);
         ++g_calls;
         return static_cast<ibv_cq*>(g_returned);
       };
     },
     [] {
       return CallWithOut<ibv_cq>([](ibv_cq** out) { return wrap_ibv_create_cq(out, kCtx, 64, kAddr, kChannel, 3); });
     }},
    {"create_qp",
     [] {
       ibvSymbols.ibv_internal_create_qp = [](ibv_pd* pd, ibv_qp_init_attr* attr) {
         EXPECT_EQ(pd, kPd);
         EXPECT_EQ(attr, &g_createQpAttr);
         ++g_calls;
         return static_cast<ibv_qp*>(g_returned);
       };
     },
     [] { return CallWithOut<ibv_qp>([](ibv_qp** out) { return wrap_ibv_create_qp(out, kPd, &g_createQpAttr); }); }},
};

class IbvwrapObjectReturnMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<WrapCase> {};

TEST_P(IbvwrapObjectReturnMicrotest, VerbReturnsObject_SucceedsAndHandsItBack) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), ncclSuccess);
  EXPECT_EQ(g_out, kAddr);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapObjectReturnMicrotest, VerbReturnsNull_ReturnsSystemErrorAndNullObject) {
  GetParam().install();
  g_returned = nullptr;
  EXPECT_EQ(GetParam().call(), ncclSystemError);
  EXPECT_EQ(g_out, nullptr);
}

TEST_P(IbvwrapObjectReturnMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(GetParam().call(), ncclInternalError);
}

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapObjectReturnMicrotest, ::testing::ValuesIn(kObjectCases), CaseName<WrapCase>);

// --- wrap_direct_*: hand back the verb's result as is, NULL when not loaded ---

struct DirectCase {
  const char* name;
  void (*install)();
  ibv_mr* (*call)();
};

const DirectCase kDirectCases[] = {
    {"reg_mr", InstallRegMr, [] { return wrap_direct_ibv_reg_mr(kPd, kAddr, 4096, IBV_ACCESS_LOCAL_WRITE); }},
    {"reg_mr_iova2", InstallRegMrIova2,
     [] { return wrap_direct_ibv_reg_mr_iova2(kPd, kAddr, 4096, 0x5000, IBV_ACCESS_REMOTE_READ); }},
    {"reg_dmabuf_mr", InstallRegDmabufMr,
     [] { return wrap_direct_ibv_reg_dmabuf_mr(kPd, 0x40, 4096, 0x5000, 7, IBV_ACCESS_REMOTE_WRITE); }},
};

class IbvwrapDirectMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<DirectCase> {};

TEST_P(IbvwrapDirectMicrotest, SymbolLoaded_ReturnsVerbResult) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), kAddr);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapDirectMicrotest, SymbolNotLoaded_ReturnsNull) { EXPECT_EQ(GetParam().call(), nullptr); }

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapDirectMicrotest, ::testing::ValuesIn(kDirectCases),
                         CaseName<DirectCase>);

// ncclIbDmaBufSupport() reads errno to tell "unsupported" from a real failure.
TEST_F(IbvwrapMicrotest, DirectRegDmabufMr_SymbolNotLoaded_SetsEopnotsupp) {
  errno = 0;
  EXPECT_EQ(wrap_direct_ibv_reg_dmabuf_mr(kPd, 0, 64, 0, 7, 0), nullptr);
  EXPECT_EQ(errno, EOPNOTSUPP);
}

// --- Verbs whose result is not checked ---

const WrapCase kPassthroughCases[] = {
    {"free_device_list",
     [] {
       ibvSymbols.ibv_internal_free_device_list = [](ibv_device** list) {
         EXPECT_EQ(list, g_deviceList);
         ++g_calls;
       };
     },
     [] { return wrap_ibv_free_device_list(g_deviceList); }},
    {"ack_async_event",
     [] {
       ibvSymbols.ibv_internal_ack_async_event = [](ibv_async_event* e) {
         EXPECT_EQ(e, &g_event);
         ++g_calls;
       };
     },
     [] { return wrap_ibv_ack_async_event(&g_event); }},
};

class IbvwrapPassthroughMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<WrapCase> {};

TEST_P(IbvwrapPassthroughMicrotest, SymbolLoaded_CallsItAndSucceeds) {
  GetParam().install();
  EXPECT_EQ(GetParam().call(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapPassthroughMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(GetParam().call(), ncclInternalError);
}

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapPassthroughMicrotest, ::testing::ValuesIn(kPassthroughCases), CaseName<WrapCase>);

// --- ECE is optional: a missing or unsupported verb is not an error ---

struct EceCase {
  const char* name;
  void (*install)();
  ncclResult_t (*call)(int* supported);
};

int FakeEce(ibv_qp* qp, ibv_ece* ece) {
  EXPECT_EQ(qp, kQp);
  EXPECT_EQ(ece, &g_ece);
  ++g_calls;
  return g_ret;
}

const EceCase kEceCases[] = {
    {"query_ece", [] { ibvSymbols.ibv_internal_query_ece = FakeEce; },
     [](int* supported) { return wrap_ibv_query_ece(kQp, &g_ece, supported); }},
    {"set_ece", [] { ibvSymbols.ibv_internal_set_ece = FakeEce; },
     [](int* supported) { return wrap_ibv_set_ece(kQp, &g_ece, supported); }},
};

class IbvwrapEceMicrotest : public IbvwrapMicrotest, public ::testing::WithParamInterface<EceCase> {};

TEST_P(IbvwrapEceMicrotest, SymbolNotLoaded_SucceedsAsUnsupported) {
  int supported = -1;
  EXPECT_EQ(GetParam().call(&supported), ncclSuccess);
  EXPECT_EQ(supported, 0);
}

TEST_P(IbvwrapEceMicrotest, VerbReturnsEopnotsupp_SucceedsAsUnsupported) {
  GetParam().install();
  g_ret = EOPNOTSUPP;
  int supported = -1;
  EXPECT_EQ(GetParam().call(&supported), ncclSuccess);
  EXPECT_EQ(supported, 0);
}

TEST_P(IbvwrapEceMicrotest, VerbReturnsZero_SucceedsAsSupported) {
  GetParam().install();
  int supported = -1;
  EXPECT_EQ(GetParam().call(&supported), ncclSuccess);
  EXPECT_EQ(supported, 1);
  EXPECT_EQ(g_calls, 1);
}

TEST_P(IbvwrapEceMicrotest, VerbReturnsOtherErrno_ReturnsSystemError) {
  GetParam().install();
  g_ret = EINVAL;
  int supported = -1;
  EXPECT_EQ(GetParam().call(&supported), ncclSystemError);
}

INSTANTIATE_TEST_SUITE_P(Ibvwrap, IbvwrapEceMicrotest, ::testing::ValuesIn(kEceCases),
                         CaseName<EceCase>);

// --- Wrappers with their own contracts ---

TEST_F(IbvwrapMicrotest, GetDeviceList_ListReturned_PassesItAndCountThrough) {
  ibvSymbols.ibv_internal_get_device_list = [](int* n) {
    *n = 1;
    return g_deviceList;
  };
  ibv_device** list = nullptr;
  int n = -1;
  EXPECT_EQ(wrap_ibv_get_device_list(&list, &n), ncclSuccess);
  EXPECT_EQ(list, g_deviceList);
  EXPECT_EQ(n, 1);
}

TEST_F(IbvwrapMicrotest, GetDeviceList_NullList_ReportsZeroDevices) {
  ibvSymbols.ibv_internal_get_device_list = [](int* n) -> ibv_device** {
    *n = 5;
    return nullptr;
  };
  ibv_device** list = g_deviceList;
  int n = -1;
  EXPECT_EQ(wrap_ibv_get_device_list(&list, &n), ncclSuccess);
  EXPECT_EQ(list, nullptr);
  EXPECT_EQ(n, 0);
}

TEST_F(IbvwrapMicrotest, FreeDeviceList_NullList_SucceedsWithoutCallingVerb) {
  ibvSymbols.ibv_internal_free_device_list = [](ibv_device**) { ++g_calls; };
  EXPECT_EQ(wrap_ibv_free_device_list(nullptr), ncclSuccess);
  EXPECT_EQ(g_calls, 0);
}

TEST_F(IbvwrapMicrotest, GetDeviceName_SymbolLoaded_ReturnsVerbResult) {
  ibvSymbols.ibv_internal_get_device_name = [](ibv_device* d) {
    EXPECT_EQ(d, kDevice);
    return "mlx5_0";
  };
  EXPECT_STREQ(wrap_ibv_get_device_name(kDevice), "mlx5_0");
}

using IbvwrapDeathMicrotest = IbvwrapMicrotest;

TEST_F(IbvwrapDeathMicrotest, GetDeviceName_SymbolNotLoaded_ExitsProcess) {
  ::testing::GTEST_FLAG(death_test_style) = "threadsafe";
  EXPECT_EXIT(wrap_ibv_get_device_name(kDevice), ::testing::ExitedWithCode(255), "");
}

TEST_F(IbvwrapMicrotest, RegMrIova2_NullRet_SucceedsWithoutRegistering) {
  InstallRegMrIova2();
  EXPECT_EQ(wrap_ibv_reg_mr_iova2(nullptr, kPd, kAddr, 4096, 0x5000, IBV_ACCESS_REMOTE_READ), ncclSuccess);
  EXPECT_EQ(g_calls, 0);
}

TEST_F(IbvwrapMicrotest, RegMrIova2_NullRetAndSymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(wrap_ibv_reg_mr_iova2(nullptr, kPd, kAddr, 4096, 0, 0), ncclInternalError);
}

TEST_F(IbvwrapMicrotest, EventTypeStr_SymbolLoaded_ReturnsVerbString) {
  ibvSymbols.ibv_internal_event_type_str = [](ibv_event_type event) {
    EXPECT_EQ(event, IBV_EVENT_PORT_ERR);
    return "port error";
  };
  char* str = nullptr;
  EXPECT_EQ(wrap_ibv_event_type_str(&str, IBV_EVENT_PORT_ERR), ncclSuccess);
  EXPECT_STREQ(str, "port error");
}

// --- RCCL_IB_FAULT_INJECTION: shims go onto each opened context and come off before it closes ---

class IbvwrapFaultInjectionMicrotest : public IbvwrapMicrotest {
 protected:
  void SetUp() override {
    IbvwrapMicrotest::SetUp();
    g_loadParam = [](const char* env, int64_t deft) -> int64_t {
      return std::string(env) == "RCCL_IB_FAULT_INJECTION" ? 1 : deft;
    };
    events_.clear();
    g_returned = kCtx;
    ibvSymbols.ibv_internal_open_device = [](ibv_device*) { return static_cast<ibv_context*>(g_returned); };
    ibvSymbols.ibv_internal_close_device = [](ibv_context* c) {
      EXPECT_EQ(c, kCtx);
      events_.push_back("close");
      return 0;
    };
  }

  static std::vector<std::string> events_;
};
std::vector<std::string> IbvwrapFaultInjectionMicrotest::events_;

TEST_F(IbvwrapFaultInjectionMicrotest, OpenDevice_VerbSucceeds_InstallsShimsOnOpenedContext) {
  ScopedHook install(g_ncclIbOpsFaultInstall, [](ibv_context* c) {
    EXPECT_EQ(c, kCtx);
    return ncclSuccess;
  });
  ibv_context* ctx = nullptr;
  EXPECT_EQ(wrap_ibv_open_device(&ctx, kDevice), ncclSuccess);
  EXPECT_EQ(ctx, kCtx);
  EXPECT_EQ(install.calls, 1);
  EXPECT_TRUE(events_.empty());
}

TEST_F(IbvwrapFaultInjectionMicrotest, OpenDevice_InstallFails_ClosesDeviceAndReturnsInstallError) {
  ScopedHook install(g_ncclIbOpsFaultInstall, [](ibv_context*) { return ncclInvalidArgument; });
  ibv_context* ctx = nullptr;
  EXPECT_EQ(wrap_ibv_open_device(&ctx, kDevice), ncclInvalidArgument);
  EXPECT_EQ(ctx, nullptr);
  EXPECT_EQ(events_, (std::vector<std::string>{"close"}));
}

TEST_F(IbvwrapFaultInjectionMicrotest, OpenDevice_VerbFails_ReturnsSystemErrorWithoutInstalling) {
  g_returned = nullptr;
  ibv_context* ctx = kCtx;
  EXPECT_EQ(wrap_ibv_open_device(&ctx, kDevice), ncclSystemError);
  EXPECT_EQ(ctx, nullptr);
}

TEST_F(IbvwrapFaultInjectionMicrotest, OpenDevice_SymbolNotLoaded_ReturnsInternalError) {
  ibvSymbols.ibv_internal_open_device = nullptr;
  ibv_context* ctx = nullptr;
  EXPECT_EQ(wrap_ibv_open_device(&ctx, kDevice), ncclInternalError);
}

TEST_F(IbvwrapFaultInjectionMicrotest, CloseDevice_ContextGiven_RemovesShimsBeforeClosing) {
  ScopedHook remove(g_ncclIbOpsFaultRemove, [](ibv_context* c) {
    EXPECT_EQ(c, kCtx);
    events_.push_back("remove");
    return ncclSuccess;
  });
  EXPECT_EQ(wrap_ibv_close_device(kCtx), ncclSuccess);
  EXPECT_EQ(events_, (std::vector<std::string>{"remove", "close"}));
}

TEST_F(IbvwrapFaultInjectionMicrotest, CloseDevice_ShimRemovalFails_ReturnsErrorWithoutClosing) {
  ScopedHook remove(g_ncclIbOpsFaultRemove, [](ibv_context*) { return ncclInternalError; });
  EXPECT_EQ(wrap_ibv_close_device(kCtx), ncclInternalError);
  EXPECT_TRUE(events_.empty());
}

TEST_F(IbvwrapFaultInjectionMicrotest, CloseDevice_NullContext_SkipsShimRemoval) {
  ScopedHook remove(g_ncclIbOpsFaultRemove, [](ibv_context*) { return ncclSuccess; });
  ibvSymbols.ibv_internal_close_device = [](ibv_context*) { return 0; };
  EXPECT_EQ(wrap_ibv_close_device(nullptr), ncclSuccess);
  EXPECT_EQ(remove.calls, 0);
}

// --- wrap_ibv_query_port: extended query first, then the legacy verb on a zeroed attr ---

class IbvwrapQueryPortMicrotest : public IbvwrapMicrotest {
 protected:
  void SetUp() override {
    IbvwrapMicrotest::SetUp();
    std::memset(&vctx_, 0, sizeof(vctx_));
    vctx_.sz = sizeof(vctx_);
    vctx_.context.abi_compat = __VERBS_ABI_IS_EXTENDED;
    std::memset(&attr_, 0xff, sizeof(attr_));
    ibvSymbols.ibv_internal_query_port = [](ibv_context*, uint8_t port, ibv_port_attr* attr) {
      EXPECT_EQ(port, 1);
      ibv_port_attr zero;
      std::memset(&zero, 0, sizeof(zero));
      EXPECT_EQ(std::memcmp(attr, &zero, sizeof(zero)), 0);
      attr->lid = 7;
      ++g_calls;
      return g_ret;
    };
  }

  void UseExtendedQuery(int ret) {
    extRet_ = ret;
    vctx_.query_port = [](ibv_context*, uint8_t port, ibv_port_attr* attr, size_t len) {
      EXPECT_EQ(port, 1);
      EXPECT_EQ(len, sizeof(ibv_port_attr));
      attr->lid = 42;
      return extRet_;
    };
  }

  verbs_context vctx_;
  ibv_port_attr attr_;
  static int extRet_;
};
int IbvwrapQueryPortMicrotest::extRet_;

TEST_F(IbvwrapQueryPortMicrotest, ExtendedQuerySucceeds_UsesItWithoutLegacyVerb) {
  UseExtendedQuery(0);
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSuccess);
  EXPECT_EQ(attr_.lid, 42);
  EXPECT_EQ(g_calls, 0);
}

TEST_F(IbvwrapQueryPortMicrotest, ExtendedQueryFails_FallsBackToLegacyVerbOnZeroedAttr) {
  UseExtendedQuery(EOPNOTSUPP);
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSuccess);
  EXPECT_EQ(attr_.lid, 7);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapQueryPortMicrotest, ContextNotExtended_UsesLegacyVerb) {
  UseExtendedQuery(0);
  vctx_.context.abi_compat = nullptr;
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

// verbs_context grows downward from the ibv_context; this sz stops just short of query_port.
TEST_F(IbvwrapQueryPortMicrotest, ExtendedContextTooOldForQueryPort_UsesLegacyVerb) {
  UseExtendedQuery(0);
  vctx_.sz = sizeof(vctx_) - offsetof(verbs_context, query_port) - 1;
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapQueryPortMicrotest, ExtendedContextWithoutQueryPort_UsesLegacyVerb) {
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapQueryPortMicrotest, LegacyVerbFails_ReturnsSystemError) {
  g_ret = EINVAL;
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclSystemError);
}

TEST_F(IbvwrapQueryPortMicrotest, FallbackWithLegacySymbolNotLoaded_ReturnsInternalError) {
  ibvSymbols.ibv_internal_query_port = nullptr;
  EXPECT_EQ(wrap_ibv_query_port(&vctx_.context, 1, &attr_), ncclInternalError);
}

// --- wrap_ibv_query_port_speed ---

class IbvwrapQueryPortSpeedMicrotest : public IbvwrapMicrotest {
 protected:
  void SetUp() override {
    IbvwrapMicrotest::SetUp();
    ibvSymbols.ibv_internal_query_port_speed = [](ibv_context* c, uint8_t port, uint64_t* speed) {
      EXPECT_EQ(c, kCtx);
      EXPECT_EQ(port, 1);
      *speed = 400;
      ++g_calls;
      return g_ret;
    };
  }
  uint64_t speed_ = 0;
};

TEST_F(IbvwrapQueryPortSpeedMicrotest, VerbSucceeds_ReturnsSpeed) {
  EXPECT_EQ(wrap_ibv_query_port_speed(kCtx, 1, &speed_), ncclSuccess);
  EXPECT_EQ(speed_, 400u);
}

TEST_F(IbvwrapQueryPortSpeedMicrotest, VerbFails_ReturnsSystemError) {
  g_ret = EINVAL;
  EXPECT_EQ(wrap_ibv_query_port_speed(kCtx, 1, &speed_), ncclSystemError);
}

TEST_F(IbvwrapQueryPortSpeedMicrotest, SymbolNotLoaded_ReturnsSystemError) {
  ibvSymbols.ibv_internal_query_port_speed = nullptr;
  EXPECT_EQ(wrap_ibv_query_port_speed(kCtx, 1, &speed_), ncclSystemError);
}

TEST_F(IbvwrapQueryPortSpeedMicrotest, DisabledByParam_ReturnsSystemErrorWithoutQuerying) {
  ScopedHook param(g_loadParam, [](const char* env, int64_t deft) -> int64_t {
    return std::string(env) == "IB_QUERY_PORT_SPEED" ? 0 : deft;
  });
  EXPECT_EQ(wrap_ibv_query_port_speed(kCtx, 1, &speed_), ncclSystemError);
  EXPECT_EQ(g_calls, 0);
}

// --- wrap_ibv_modify_qp retries and its failure log line ---

class IbvwrapModifyQpMicrotest : public IbvwrapMicrotest {
 protected:
  void SetUp() override {
    IbvwrapMicrotest::SetUp();
    retryCnt_ = 3;
    sleepMs_ = 0;
    retryAll_ = 0;
    results_ = {0};
    std::memset(&context_, 0, sizeof(context_));
    std::memset(&pd_, 0, sizeof(pd_));
    std::memset(&qp_, 0, sizeof(qp_));
    pd_.context = &context_;
    qp_.pd = &pd_;
    qp_.state = IBV_QPS_INIT;
    std::memset(&attr_, 0, sizeof(attr_));
    attr_.qp_state = IBV_QPS_RTR;
    ibvSymbols.ibv_internal_modify_qp = [](ibv_qp*, ibv_qp_attr*, int mask) {
      EXPECT_EQ(mask, IBV_QP_STATE);
      int ret = g_calls < static_cast<int>(results_.size()) ? results_[g_calls] : results_.back();
      ++g_calls;
      return ret;
    };
    ibvSymbols.ibv_internal_get_device_name = [](ibv_device*) { return "mlx5_0"; };
    ibvSymbols.ibv_internal_query_qp = [](ibv_qp*, ibv_qp_attr*, int, ibv_qp_init_attr*) { return EINVAL; };
    g_loadParam = [](const char* env, int64_t deft) -> int64_t {
      std::string name(env);
      if (name == "IB_MQP_RETRY_CNT") return retryCnt_;
      if (name == "IB_MQP_RETRY_SLEEP_MSEC") return sleepMs_;
      if (name == "IB_MQP_RETRY_ALL") return retryAll_;
      return deft;
    };
  }

  ncclResult_t Modify() { return wrap_ibv_modify_qp(&qp_, &attr_, IBV_QP_STATE); }

  static std::vector<int> results_;  // what each attempt returns; the last one repeats
  static int64_t retryCnt_, sleepMs_, retryAll_;
  ibv_context context_;
  ibv_pd pd_;
  ibv_qp qp_;
  ibv_qp_attr attr_;
};
std::vector<int> IbvwrapModifyQpMicrotest::results_;
int64_t IbvwrapModifyQpMicrotest::retryCnt_;
int64_t IbvwrapModifyQpMicrotest::sleepMs_;
int64_t IbvwrapModifyQpMicrotest::retryAll_;

TEST_F(IbvwrapModifyQpMicrotest, VerbSucceeds_CallsVerbOnce) {
  EXPECT_EQ(Modify(), ncclSuccess);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapModifyQpMicrotest, TimedOutThenSucceeds_RetriesAndSucceeds) {
  results_ = {ETIMEDOUT, 0};
  EXPECT_EQ(Modify(), ncclSuccess);
  EXPECT_EQ(g_calls, 2);
}

TEST_F(IbvwrapModifyQpMicrotest, NegativeTimedOut_RetriesAndSucceeds) {
  results_ = {-ETIMEDOUT, 0};
  EXPECT_EQ(Modify(), ncclSuccess);
  EXPECT_EQ(g_calls, 2);
}

TEST_F(IbvwrapModifyQpMicrotest, AlwaysTimedOut_GivesUpAfterRetryCountPlusOneAttempts) {
  results_ = {ETIMEDOUT};
  EXPECT_EQ(Modify(), ncclSystemError);
  EXPECT_EQ(g_calls, retryCnt_ + 1);
}

TEST_F(IbvwrapModifyQpMicrotest, ZeroRetryCount_TriesOnce) {
  retryCnt_ = 0;
  results_ = {ETIMEDOUT};
  EXPECT_EQ(Modify(), ncclSystemError);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapModifyQpMicrotest, OtherError_FailsWithoutRetry) {
  results_ = {EINVAL, 0};
  EXPECT_EQ(Modify(), ncclSystemError);
  EXPECT_EQ(g_calls, 1);
}

TEST_F(IbvwrapModifyQpMicrotest, RetryAllSet_RetriesAnyError) {
  retryAll_ = 1;
  results_ = {EINVAL, 0};
  EXPECT_EQ(Modify(), ncclSuccess);
  EXPECT_EQ(g_calls, 2);
}

TEST_F(IbvwrapModifyQpMicrotest, RetryAllSet_StillStopsAtRetryCount) {
  retryAll_ = 1;
  results_ = {EINVAL};
  EXPECT_EQ(Modify(), ncclSystemError);
  EXPECT_EQ(g_calls, retryCnt_ + 1);
}

// The n-th retry sleeps n times the configured interval.
TEST_F(IbvwrapModifyQpMicrotest, TimedOutWithSleepSet_BacksOffLinearly) {
  sleepMs_ = 10;
  retryCnt_ = 2;
  results_ = {ETIMEDOUT};
  auto start = std::chrono::steady_clock::now();
  EXPECT_EQ(Modify(), ncclSystemError);
  EXPECT_GE(std::chrono::steady_clock::now() - start, std::chrono::milliseconds(10 * 1 + 10 * 2));
}

TEST_F(IbvwrapModifyQpMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  ibvSymbols.ibv_internal_modify_qp = nullptr;
  EXPECT_EQ(Modify(), ncclInternalError);
}

class IbvwrapModifyQpLogMicrotest : public IbvwrapModifyQpMicrotest {
 protected:
  std::string Log(int userFlag) {
    char msg[1024];
    ibvModifyQpLog(&qp_, IBV_QPS_RTS, &attr_, userFlag, msg, sizeof(msg));
    return msg;
  }
  void SetGlobalAv() {
    attr_.ah_attr.is_global = 1;
    attr_.ah_attr.grh.sgid_index = 3;
    SetGid(&attr_.ah_attr.grh.dgid, 0x02);
  }
  static void SetGid(union ibv_gid* gid, uint8_t last) {
    std::memset(gid, 0, sizeof(*gid));
    gid->raw[0] = 0xfe;
    gid->raw[1] = 0x80;
    gid->raw[15] = last;
  }
};

TEST_F(IbvwrapModifyQpLogMicrotest, QueryFailsAndNoUserAttr_ReportsStatesOnly) {
  std::string log = Log(0);
  EXPECT_NE(log.find("mlx5_0:-1"), std::string::npos) << log;
  EXPECT_NE(log.find("curr state INIT, next state RTS"), std::string::npos) << log;
  EXPECT_NE(log.find("local GID index -1, local GID N/A, remote GID N/A"), std::string::npos) << log;
}

TEST_F(IbvwrapModifyQpLogMicrotest, QuerySucceeds_ReportsQueriedPort) {
  ibvSymbols.ibv_internal_query_qp = [](ibv_qp*, ibv_qp_attr* attr, int mask, ibv_qp_init_attr*) {
    EXPECT_EQ(mask, IBV_QP_PORT | IBV_QP_AV);
    attr->port_num = 2;
    attr->ah_attr.is_global = 0;
    return 0;
  };
  EXPECT_NE(Log(0).find("on dev mlx5_0:2,"), std::string::npos);
}

TEST_F(IbvwrapModifyQpLogMicrotest, UserSetsPort_ReportsUserPortOverQueried) {
  ibvSymbols.ibv_internal_query_qp = [](ibv_qp*, ibv_qp_attr* attr, int, ibv_qp_init_attr*) {
    attr->port_num = 2;
    return 0;
  };
  attr_.port_num = 5;
  EXPECT_NE(Log(IBV_QP_PORT).find("on dev mlx5_0:5,"), std::string::npos);
}

TEST_F(IbvwrapModifyQpLogMicrotest, UserSetsGlobalAv_ReportsLocalAndRemoteGids) {
  attr_.port_num = 1;
  SetGlobalAv();
  ibvSymbols.ibv_internal_query_gid = [](ibv_context*, uint8_t port, int index, union ibv_gid* gid) {
    EXPECT_EQ(port, 1);
    EXPECT_EQ(index, 3);
    SetGid(gid, 0x01);
    return 0;
  };
  std::string log = Log(IBV_QP_PORT | IBV_QP_AV);
  EXPECT_NE(log.find("local GID index 3, local GID fe80::1, remote GID fe80::2"), std::string::npos) << log;
}

TEST_F(IbvwrapModifyQpLogMicrotest, UserSetsNonGlobalAv_ReportsNoGids) {
  attr_.ah_attr.is_global = 0;
  attr_.ah_attr.grh.sgid_index = 3;
  ibvSymbols.ibv_internal_query_gid = [](ibv_context*, uint8_t, int, union ibv_gid*) {
    ADD_FAILURE() << "local GID queried for a non-global AV";
    return 0;
  };
  std::string log = Log(IBV_QP_AV);
  EXPECT_NE(log.find("local GID index -1, local GID N/A, remote GID N/A"), std::string::npos) << log;
}

TEST_F(IbvwrapModifyQpLogMicrotest, LocalGidQueryFails_ReportsIndexButNoLocalGid) {
  SetGlobalAv();
  ibvSymbols.ibv_internal_query_gid = [](ibv_context*, uint8_t, int, union ibv_gid*) { return EINVAL; };
  std::string log = Log(IBV_QP_AV);
  EXPECT_NE(log.find("local GID index 3, local GID N/A, remote GID fe80::2"), std::string::npos) << log;
}

TEST_F(IbvwrapModifyQpLogMicrotest, NoPdContext_ReportsNoDeviceAndSkipsLocalGid) {
  pd_.context = nullptr;
  SetGlobalAv();
  std::string log = Log(IBV_QP_AV);
  EXPECT_NE(log.find("on dev N/A:"), std::string::npos) << log;
  EXPECT_NE(log.find("local GID index -1, local GID N/A, remote GID fe80::2"), std::string::npos) << log;
}

TEST(IbvwrapQpStateNameMicrotest, EachState_HasItsName) {
  const std::pair<ibv_qp_state, const char*> kNames[] = {
      {IBV_QPS_RESET, "RESET"}, {IBV_QPS_INIT, "INIT"}, {IBV_QPS_RTR, "RTR"}, {IBV_QPS_RTS, "RTS"},
      {IBV_QPS_SQD, "SQD"},     {IBV_QPS_SQE, "SQE"},   {IBV_QPS_ERR, "ERR"}, {IBV_QPS_UNKNOWN, "UNKNOWN"},
  };
  char msg[32];
  for (const auto& [state, name] : kNames) {
    ibvQpStateName(state, msg, sizeof(msg));
    EXPECT_STREQ(msg, name);
  }
  ibvQpStateName(static_cast<ibv_qp_state>(42), msg, sizeof(msg));
  EXPECT_STREQ(msg, "NOT RECOGNIZED (42)");
}

// --- Enum-to-string helpers ---

#define NAMED(e) std::pair<int, const char*>{e, #e}

template <typename Enum>
void ExpectNames(const char* (*str)(Enum), const std::vector<std::pair<int, const char*>>& names,
                 const char* unknown) {
  for (const auto& [value, name] : names) EXPECT_STREQ(str(static_cast<Enum>(value)), name);
  EXPECT_STREQ(str(static_cast<Enum>(999)), unknown);
}

TEST(IbvwrapStrMicrotest, WcStatusStr_EachStatus_HasItsNameAndUnknownIsFlagged) {
  ExpectNames(ibvWcStatusStr,
              {NAMED(IBV_WC_SUCCESS),           NAMED(IBV_WC_LOC_LEN_ERR),       NAMED(IBV_WC_LOC_QP_OP_ERR),
               NAMED(IBV_WC_LOC_EEC_OP_ERR),    NAMED(IBV_WC_LOC_PROT_ERR),      NAMED(IBV_WC_WR_FLUSH_ERR),
               NAMED(IBV_WC_MW_BIND_ERR),       NAMED(IBV_WC_BAD_RESP_ERR),      NAMED(IBV_WC_LOC_ACCESS_ERR),
               NAMED(IBV_WC_REM_INV_REQ_ERR),   NAMED(IBV_WC_REM_ACCESS_ERR),    NAMED(IBV_WC_REM_OP_ERR),
               NAMED(IBV_WC_RETRY_EXC_ERR),     NAMED(IBV_WC_RNR_RETRY_EXC_ERR), NAMED(IBV_WC_LOC_RDD_VIOL_ERR),
               NAMED(IBV_WC_REM_INV_RD_REQ_ERR), NAMED(IBV_WC_REM_ABORT_ERR),    NAMED(IBV_WC_INV_EECN_ERR),
               NAMED(IBV_WC_INV_EEC_STATE_ERR), NAMED(IBV_WC_FATAL_ERR),         NAMED(IBV_WC_RESP_TIMEOUT_ERR),
               NAMED(IBV_WC_GENERAL_ERR)},
              "UNKNOWN_STATUS");
}

TEST(IbvwrapStrMicrotest, WcOpcodeStr_EachOpcode_HasItsNameAndUnknownIsFlagged) {
  ExpectNames(ibvWcOpcodeStr,
              {NAMED(IBV_WC_SEND), NAMED(IBV_WC_RDMA_WRITE), NAMED(IBV_WC_RDMA_READ), NAMED(IBV_WC_COMP_SWAP),
               NAMED(IBV_WC_FETCH_ADD), NAMED(IBV_WC_BIND_MW), NAMED(IBV_WC_RECV), NAMED(IBV_WC_RECV_RDMA_WITH_IMM)},
              "UNKNOWN_OPCODE");
}

TEST(IbvwrapStrMicrotest, WrOpcodeStr_EachOpcode_HasItsNameAndUnknownIsFlagged) {
  ExpectNames(ibvWrOpcodeStr,
              {NAMED(IBV_WR_RDMA_WRITE), NAMED(IBV_WR_RDMA_WRITE_WITH_IMM), NAMED(IBV_WR_SEND),
               NAMED(IBV_WR_SEND_WITH_IMM), NAMED(IBV_WR_RDMA_READ), NAMED(IBV_WR_ATOMIC_CMP_AND_SWP),
               NAMED(IBV_WR_ATOMIC_FETCH_AND_ADD)},
              "UNKNOWN_OPCODE");
}

#undef NAMED

// --- ibvwrap.h inline wrappers: these call through the context's ops table ---

class IbvwrapInlineMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::memset(&context_, 0, sizeof(context_));
    std::memset(&pd_, 0, sizeof(pd_));
    std::memset(&qp_, 0, sizeof(qp_));
    std::memset(&cq_, 0, sizeof(cq_));
    std::memset(&ah_, 0, sizeof(ah_));
    pd_.context = &context_;
    qp_.context = &context_;
    cq_.context = &context_;
    g_ret = 0;
    created_ = &ah_;
    context_.ops.create_ah = [](ibv_pd*, ibv_ah_attr*) { return created_; };
    context_.ops.destroy_ah = [](ibv_ah*) { return g_ret; };
    context_.ops.post_send = [](ibv_qp*, ibv_send_wr*, ibv_send_wr**) { return g_ret; };
    context_.ops.post_recv = [](ibv_qp*, ibv_recv_wr*, ibv_recv_wr**) { return g_ret; };
  }

  static ibv_ah* created_;
  ibv_context context_;
  ibv_pd pd_;
  ibv_qp qp_;
  ibv_cq cq_;
  ibv_ah ah_;
  ibv_ah_attr ahAttr_{};
  ibv_ah* ahOut_ = nullptr;
};
ibv_ah* IbvwrapInlineMicrotest::created_;

TEST_F(IbvwrapInlineMicrotest, PollCq_CompletionsReturned_ReportsCount) {
  context_.ops.poll_cq = [](ibv_cq*, int n, ibv_wc*) {
    EXPECT_EQ(n, 8);
    return 3;
  };
  ibv_wc wc[8];
  int done = -1;
  EXPECT_EQ(wrap_ibv_poll_cq(&cq_, 8, wc, &done), ncclSuccess);
  EXPECT_EQ(done, 3);
}

TEST_F(IbvwrapInlineMicrotest, PollCq_NoCompletions_ReportsZero) {
  context_.ops.poll_cq = [](ibv_cq*, int, ibv_wc*) { return 0; };
  ibv_wc wc[1];
  int done = -1;
  EXPECT_EQ(wrap_ibv_poll_cq(&cq_, 1, wc, &done), ncclSuccess);
  EXPECT_EQ(done, 0);
}

TEST_F(IbvwrapInlineMicrotest, PollCq_NegativeReturn_ReturnsSystemError) {
  context_.ops.poll_cq = [](ibv_cq*, int, ibv_wc*) { return -1; };
  ibv_wc wc[1];
  int done = -1;
  EXPECT_EQ(wrap_ibv_poll_cq(&cq_, 1, wc, &done), ncclSystemError);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_VerbSucceeds_LinksAhToPdAndContext) {
  EXPECT_EQ(wrap_ibv_create_ah(&ahOut_, &pd_, &ahAttr_), ncclSuccess);
  EXPECT_EQ(ahOut_, &ah_);
  EXPECT_EQ(ah_.context, &context_);
  EXPECT_EQ(ah_.pd, &pd_);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_VerbFails_ReturnsSystemErrorAndNullAh) {
  created_ = nullptr;
  ahOut_ = &ah_;
  EXPECT_EQ(wrap_ibv_create_ah(&ahOut_, &pd_, &ahAttr_), ncclSystemError);
  EXPECT_EQ(ahOut_, nullptr);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_NullPd_ReturnsSystemErrorAndNullAh) {
  ahOut_ = &ah_;
  EXPECT_EQ(wrap_ibv_create_ah(&ahOut_, nullptr, &ahAttr_), ncclSystemError);
  EXPECT_EQ(ahOut_, nullptr);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_PdWithoutContext_ReturnsSystemErrorAndNullAh) {
  pd_.context = nullptr;
  ahOut_ = &ah_;
  EXPECT_EQ(wrap_ibv_create_ah(&ahOut_, &pd_, &ahAttr_), ncclSystemError);
  EXPECT_EQ(ahOut_, nullptr);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_NullAttr_ReturnsSystemErrorAndNullAh) {
  ahOut_ = &ah_;
  EXPECT_EQ(wrap_ibv_create_ah(&ahOut_, &pd_, nullptr), ncclSystemError);
  EXPECT_EQ(ahOut_, nullptr);
}

TEST_F(IbvwrapInlineMicrotest, CreateAh_NullRet_ReturnsSystemError) {
  EXPECT_EQ(wrap_ibv_create_ah(nullptr, &pd_, &ahAttr_), ncclSystemError);
}

TEST_F(IbvwrapInlineMicrotest, DestroyAh_VerbReturnsZero_Succeeds) {
  ah_.context = &context_;
  EXPECT_EQ(wrap_ibv_destroy_ah(&ah_), ncclSuccess);
}

TEST_F(IbvwrapInlineMicrotest, DestroyAh_VerbReturnsErrno_ReturnsSystemError) {
  ah_.context = &context_;
  g_ret = EBUSY;
  EXPECT_EQ(wrap_ibv_destroy_ah(&ah_), ncclSystemError);
}

TEST_F(IbvwrapInlineMicrotest, DestroyAh_NullAhOrContext_ReturnsSystemError) {
  EXPECT_EQ(wrap_ibv_destroy_ah(nullptr), ncclSystemError);
  EXPECT_EQ(wrap_ibv_destroy_ah(&ah_), ncclSystemError);
}

TEST_F(IbvwrapInlineMicrotest, PostSend_VerbReturnsZero_Succeeds) {
  ibv_send_wr wr{};
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(wrap_ibv_post_send(&qp_, &wr, &bad), ncclSuccess);
}

TEST_F(IbvwrapInlineMicrotest, PostSend_VerbReturnsErrno_ReturnsSystemError) {
  g_ret = ENOMEM;
  ibv_send_wr wr{};
  ibv_send_wr* bad = nullptr;
  EXPECT_EQ(wrap_ibv_post_send(&qp_, &wr, &bad), ncclSystemError);
}

TEST_F(IbvwrapInlineMicrotest, PostRecv_VerbReturnsZero_Succeeds) {
  ibv_recv_wr wr{};
  ibv_recv_wr* bad = nullptr;
  EXPECT_EQ(wrap_ibv_post_recv(&qp_, &wr, &bad), ncclSuccess);
}

TEST_F(IbvwrapInlineMicrotest, PostRecv_VerbReturnsErrno_ReturnsSystemError) {
  g_ret = ENOMEM;
  ibv_recv_wr wr{};
  ibv_recv_wr* bad = nullptr;
  EXPECT_EQ(wrap_ibv_post_recv(&qp_, &wr, &bad), ncclSystemError);
}

TEST(IbvwrapGidStrMicrotest, ValidGid_FormatsAsIpv6) {
  union ibv_gid gid;
  std::memset(&gid, 0, sizeof(gid));
  gid.raw[10] = 0xff;
  gid.raw[11] = 0xff;
  gid.raw[12] = 10;
  gid.raw[15] = 1;
  char str[INET6_ADDRSTRLEN];
  EXPECT_EQ(ibvGetGidStr(&gid, str, sizeof(str)), str);
  EXPECT_STREQ(str, "::ffff:10.0.0.1");
}

TEST(IbvwrapGidStrMicrotest, BufferTooSmall_ReturnsNullWithEnospc) {
  union ibv_gid gid;
  std::memset(gid.raw, 0xab, sizeof(gid.raw));
  char str[8];
  errno = 0;
  EXPECT_EQ(ibvGetGidStr(&gid, str, sizeof(str)), nullptr);
  EXPECT_EQ(errno, ENOSPC);
}

// Builds once per process, so runs isolated. Relies on ibvsymbols-test.cc's dl_seam.h build.
TEST(IbvwrapSymbolsMicrotestIsolated, LibraryMissing_FailsAndDoesNotRetry) {
  RUN_ISOLATED_TEST("IbvwrapSymbolsMicrotestIsolated_LibraryMissing_FailsAndDoesNotRetry", []() {
    SetMicroEnvAbsent("NCCL_IBVERBS_LIB");
    SetMicroEnvAbsent("NCCL_LIBIBVERBS_SO");
    ScriptedDl dl;
    EXPECT_EQ(wrap_ibv_symbols(), ncclSystemError);
    size_t opens = dl.opens.size();
    EXPECT_GT(opens, 0u);
    dl.AddLibrary("libibverbs.so");
    EXPECT_EQ(wrap_ibv_symbols(), ncclSystemError);
    EXPECT_EQ(dl.opens.size(), opens);
  });
}

}  // namespace
