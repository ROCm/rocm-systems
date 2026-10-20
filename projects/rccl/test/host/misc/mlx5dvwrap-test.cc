/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Host-only tests for src/misc/mlx5dvwrap.cc.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include <gtest/gtest.h>

#include <cerrno>
#include <cstring>

#include "../common/ProcessIsolatedTestRunner.hpp"  // RUN_ISOLATED_TEST
#include "fakes/dl_fakes.h"

#include MLX5DVWRAP_CC_PATH

namespace {

// The wrappers only forward these, so they are never dereferenced.
ibv_device* const kDevice = reinterpret_cast<ibv_device*>(0xB0);
ibv_context* const kCtx = reinterpret_cast<ibv_context*>(0xC0);
ibv_pd* const kPd = reinterpret_cast<ibv_pd*>(0xD0);
ibv_mr* const kMr = reinterpret_cast<ibv_mr*>(0xE0);
ibv_qp* const kQp = reinterpret_cast<ibv_qp*>(0xF0);
mlx5dv_context* const kDvAttrs = reinterpret_cast<mlx5dv_context*>(0x100);
ibv_qp_init_attr_ex* const kQpAttr = reinterpret_cast<ibv_qp_init_attr_ex*>(0x110);
mlx5dv_qp_init_attr* const kMlx5QpAttr = reinterpret_cast<mlx5dv_qp_init_attr*>(0x120);
char* const kBuf = reinterpret_cast<char*>(0x130);

int g_ret;
ibv_mr* g_mr;

void InstallRegDmabufMr() {
  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = [](ibv_pd* pd, uint64_t offset, size_t len, uint64_t iova, int fd,
                                                  int access, int mlx5Access) {
    EXPECT_EQ(pd, kPd);
    EXPECT_EQ(offset, 0x40u);
    EXPECT_EQ(len, 4096u);
    EXPECT_EQ(iova, 0x5000u);
    EXPECT_EQ(fd, 7);
    EXPECT_EQ(access, 3);
    EXPECT_EQ(mlx5Access, 4);
    return g_mr;
  };
}

class Mlx5dvwrapMicrotest : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_ = mlx5dvSymbols;
    std::memset(&mlx5dvSymbols, 0, sizeof(mlx5dvSymbols));
    g_ret = 0;
    g_mr = kMr;
  }
  void TearDown() override { mlx5dvSymbols = saved_; }

 private:
  ncclMlx5dvSymbols saved_;
};

TEST_F(Mlx5dvwrapMicrotest, IsSupported_SymbolNotLoaded_ReturnsFalse) {
  EXPECT_FALSE(wrap_mlx5dv_is_supported(kDevice));
}

TEST_F(Mlx5dvwrapMicrotest, IsSupported_SymbolLoaded_ReturnsVerbAnswer) {
  mlx5dvSymbols.mlx5dv_internal_is_supported = [](ibv_device* d) { return d == kDevice; };
  EXPECT_TRUE(wrap_mlx5dv_is_supported(kDevice));
  EXPECT_FALSE(wrap_mlx5dv_is_supported(nullptr));
}

class Mlx5dvwrapDataDirectMicrotest : public Mlx5dvwrapMicrotest {
 protected:
  void SetUp() override {
    Mlx5dvwrapMicrotest::SetUp();
    mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path = [](ibv_context* c, char* buf, size_t len) {
      EXPECT_EQ(c, kCtx);
      EXPECT_EQ(buf, kBuf);
      EXPECT_EQ(len, 256u);
      return g_ret;
    };
  }
  ncclResult_t Call() { return wrap_mlx5dv_get_data_direct_sysfs_path(kCtx, kBuf, 256); }
};

TEST_F(Mlx5dvwrapDataDirectMicrotest, VerbSucceeds_Succeeds) { EXPECT_EQ(Call(), ncclSuccess); }

// ENODEV only means the device is not data-direct.
TEST_F(Mlx5dvwrapDataDirectMicrotest, VerbReturnsEnodev_ReturnsInvalidArgument) {
  g_ret = ENODEV;
  EXPECT_EQ(Call(), ncclInvalidArgument);
}

TEST_F(Mlx5dvwrapDataDirectMicrotest, VerbReturnsOtherErrno_ReturnsSystemError) {
  g_ret = EIO;
  EXPECT_EQ(Call(), ncclSystemError);
}

TEST_F(Mlx5dvwrapDataDirectMicrotest, SymbolNotLoaded_ReturnsInternalError) {
  mlx5dvSymbols.mlx5dv_internal_get_data_direct_sysfs_path = nullptr;
  EXPECT_EQ(Call(), ncclInternalError);
}

TEST_F(Mlx5dvwrapMicrotest, RegDmabufMr_VerbReturnsMr_SucceedsAndHandsItBack) {
  InstallRegDmabufMr();
  ibv_mr* mr = nullptr;
  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&mr, kPd, 0x40, 4096, 0x5000, 7, 3, 4), ncclSuccess);
  EXPECT_EQ(mr, kMr);
}

TEST_F(Mlx5dvwrapMicrotest, RegDmabufMr_VerbReturnsNull_ReturnsSystemError) {
  InstallRegDmabufMr();
  g_mr = nullptr;
  ibv_mr* mr = kMr;
  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&mr, kPd, 0x40, 4096, 0x5000, 7, 3, 4), ncclSystemError);
  EXPECT_EQ(mr, nullptr);
}

TEST_F(Mlx5dvwrapMicrotest, RegDmabufMr_SymbolNotLoaded_ReturnsInternalError) {
  ibv_mr* mr = nullptr;
  EXPECT_EQ(wrap_mlx5dv_reg_dmabuf_mr(&mr, kPd, 0x40, 4096, 0x5000, 7, 3, 4), ncclInternalError);
}

TEST_F(Mlx5dvwrapMicrotest, DirectRegDmabufMr_SymbolLoaded_ReturnsVerbResult) {
  InstallRegDmabufMr();
  EXPECT_EQ(wrap_direct_mlx5dv_reg_dmabuf_mr(kPd, 0x40, 4096, 0x5000, 7, 3, 4), kMr);
}

// ncclIbDmaBufSupport() reads errno to tell "unsupported" from a real failure.
TEST_F(Mlx5dvwrapMicrotest, DirectRegDmabufMr_SymbolNotLoaded_ReturnsNullWithEopnotsupp) {
  errno = 0;
  EXPECT_EQ(wrap_direct_mlx5dv_reg_dmabuf_mr(kPd, 0x40, 4096, 0x5000, 7, 3, 4), nullptr);
  EXPECT_EQ(errno, EOPNOTSUPP);
}

void InstallQueryDevice() {
  mlx5dvSymbols.mlx5dv_internal_query_device = [](ibv_context* c, mlx5dv_context* attrs) {
    EXPECT_EQ(c, kCtx);
    EXPECT_EQ(attrs, kDvAttrs);
    return g_ret;
  };
}

TEST_F(Mlx5dvwrapMicrotest, QueryDevice_VerbReturnsZero_Succeeds) {
  InstallQueryDevice();
  EXPECT_EQ(wrap_mlx5dv_query_device(kCtx, kDvAttrs), ncclSuccess);
}

TEST_F(Mlx5dvwrapMicrotest, QueryDevice_VerbReturnsErrno_ReturnsSystemError) {
  InstallQueryDevice();
  g_ret = EINVAL;
  EXPECT_EQ(wrap_mlx5dv_query_device(kCtx, kDvAttrs), ncclSystemError);
}

TEST_F(Mlx5dvwrapMicrotest, QueryDevice_SymbolNotLoaded_ReturnsInternalError) {
  EXPECT_EQ(wrap_mlx5dv_query_device(kCtx, kDvAttrs), ncclInternalError);
}

TEST_F(Mlx5dvwrapMicrotest, CreateQp_SymbolLoaded_ReturnsVerbResult) {
  mlx5dvSymbols.mlx5dv_internal_create_qp = [](ibv_context* c, ibv_qp_init_attr_ex* attr,
                                               mlx5dv_qp_init_attr* mlx5Attr) {
    EXPECT_EQ(c, kCtx);
    EXPECT_EQ(attr, kQpAttr);
    EXPECT_EQ(mlx5Attr, kMlx5QpAttr);
    return kQp;
  };
  EXPECT_EQ(wrap_mlx5dv_create_qp(kCtx, kQpAttr, kMlx5QpAttr), kQp);
}

TEST_F(Mlx5dvwrapMicrotest, CreateQp_SymbolNotLoaded_ReturnsNull) {
  EXPECT_EQ(wrap_mlx5dv_create_qp(kCtx, kQpAttr, kMlx5QpAttr), nullptr);
}

// Builds once per process, so runs isolated. Relies on mlx5dvsymbols-test.cc's dl_seam.h build.
TEST(Mlx5dvwrapSymbolsMicrotestIsolated, CalledTwice_BuildsOnceAndCachesResult) {
  RUN_ISOLATED_TEST("Mlx5dvwrapSymbolsMicrotestIsolated_CalledTwice_BuildsOnceAndCachesResult", []() {
    ScriptedDl dl;
    void* lib = dl.AddLibrary("libmlx5.so");
    dl.Export(lib, "mlx5dv_is_supported", "MLX5_1.8", reinterpret_cast<void*>(0x1));
    EXPECT_EQ(wrap_mlx5dv_symbols(), ncclSuccess);
    EXPECT_EQ(reinterpret_cast<void*>(mlx5dvSymbols.mlx5dv_internal_is_supported), reinterpret_cast<void*>(0x1));
    EXPECT_EQ(wrap_mlx5dv_symbols(), ncclSuccess);
    EXPECT_EQ(dl.opens.size(), 1u);
  });
}

}  // namespace
