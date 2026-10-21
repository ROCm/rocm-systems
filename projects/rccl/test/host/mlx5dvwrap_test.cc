/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/
//
// Host-only microtests for wrap_reg_dmabuf_mr (src/misc/mlx5dvwrap.cc).
// No NIC: the mlx5 symbol is installed or cleared directly, and ibv_reg_dmabuf_mr
// is the g_ibvRegDmabufMr seam. Each test checks which entry point produced the MR.

#include <gtest/gtest.h>

#include <cerrno>

#include "fakes/ibvwrap_fakes.h"
#include "mlx5/mlx5dvsymbols.h"
#include "mlx5/mlx5dvwrap.h"

extern struct ncclMlx5dvSymbols mlx5dvSymbols;

namespace {

int g_mlx5Calls = 0;
struct ibv_mr* g_mlx5Mr = nullptr;

struct ibv_mr* TestMlx5RegDmabufMr(struct ibv_pd*, uint64_t, size_t, uint64_t, int, int, int) {
  g_mlx5Calls++;
  if (g_mlx5Mr == nullptr) {
    errno = EINVAL;
    return nullptr;
  }
  return g_mlx5Mr;
}

class Mlx5dvRegDmabufTest : public ::testing::Test {
 protected:
  struct ibv_pd pd_ {};
  struct ibv_mr ibvMr_ {};
  struct ibv_mr mlx5Mr_ {};
  int ibvCalls_ = 0;

  void SetUp() override {
    ResetIbvwrapFakes();
    mlx5dvSymbols = {};
    g_mlx5Calls = 0;
    g_mlx5Mr = nullptr;
    ibvCalls_ = 0;
    g_ibvRegDmabufMr = [this](struct ibv_mr** ret, struct ibv_pd*, uint64_t, size_t, uint64_t, int, int) {
      ibvCalls_++;
      *ret = &ibvMr_;
      return ncclSuccess;
    };
  }

  void TearDown() override { ResetIbvwrapFakes(); }

  ncclResult_t Reg(bool dataDirect, struct ibv_mr** out) {
    return wrap_reg_dmabuf_mr(out, &pd_, 0, 4096, 0, 3, 0, dataDirect);
  }
};

TEST_F(Mlx5dvRegDmabufTest, DataDirectOffUsesIbv) {
  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = TestMlx5RegDmabufMr;
  g_mlx5Mr = &mlx5Mr_;
  struct ibv_mr* mr = nullptr;
  EXPECT_EQ(ncclSuccess, Reg(false, &mr));
  EXPECT_EQ(&ibvMr_, mr);
  EXPECT_EQ(1, ibvCalls_);
  EXPECT_EQ(0, g_mlx5Calls);
}

TEST_F(Mlx5dvRegDmabufTest, MissingMlx5SymbolFallsBackToIbv) {
  struct ibv_mr* mr = nullptr;
  EXPECT_EQ(ncclSuccess, Reg(true, &mr));
  EXPECT_EQ(&ibvMr_, mr);
  EXPECT_EQ(1, ibvCalls_);
  EXPECT_EQ(0, g_mlx5Calls);
}

TEST_F(Mlx5dvRegDmabufTest, DataDirectUsesMlx5WhenItSucceeds) {
  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = TestMlx5RegDmabufMr;
  g_mlx5Mr = &mlx5Mr_;
  struct ibv_mr* mr = nullptr;
  EXPECT_EQ(ncclSuccess, Reg(true, &mr));
  EXPECT_EQ(&mlx5Mr_, mr);
  EXPECT_EQ(0, ibvCalls_);
  EXPECT_EQ(1, g_mlx5Calls);
}

TEST_F(Mlx5dvRegDmabufTest, Mlx5FailureFallsBackToIbv) {
  mlx5dvSymbols.mlx5dv_internal_reg_dmabuf_mr = TestMlx5RegDmabufMr;
  g_mlx5Mr = nullptr;
  struct ibv_mr* mr = nullptr;
  EXPECT_EQ(ncclSuccess, Reg(true, &mr));
  EXPECT_EQ(&ibvMr_, mr);
  EXPECT_EQ(1, ibvCalls_);
  EXPECT_EQ(1, g_mlx5Calls);
}

}  // namespace
