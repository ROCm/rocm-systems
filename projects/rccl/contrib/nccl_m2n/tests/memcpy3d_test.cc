/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 * See LICENSE.txt for more license information.
 ************************************************************************/

#include <gtest/gtest.h>
#include <cstdint>
#include <vector>

#include "cuda_runtime.h"
#include "m2n_checks.h"

namespace {

constexpr size_t kGiB = (size_t)1 << 30;

class Memcpy3DTest : public ::testing::Test {
protected:
  cudaStream_t stream = nullptr;
  std::vector<void*> buffers;

  void SetUp() override {
    ASSERT_EQ(cudaSuccess, cudaSetDevice(0));
    ASSERT_EQ(cudaSuccess, cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
  }

  void TearDown() override {
    for (void* b : buffers) cudaFree(b);
    if (stream != nullptr) cudaStreamDestroy(stream);
  }

  bool haveFree(size_t bytes) {
    size_t freeBytes = 0, total = 0;
    return cudaMemGetInfo(&freeBytes, &total) == cudaSuccess && freeBytes > bytes + kGiB;
  }

  char* alloc(size_t bytes, int value) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) return nullptr;
    buffers.push_back(p);
    if (cudaMemset(p, value, bytes) != cudaSuccess) return nullptr;
    return (char*)p;
  }

  static void put(char* dev, size_t offset, uint64_t value) {
    ASSERT_EQ(cudaSuccess, cudaMemcpy(dev + offset, &value, sizeof(value), cudaMemcpyHostToDevice));
  }

  static uint64_t get(const char* dev, size_t offset) {
    uint64_t value = 0;
    EXPECT_EQ(cudaSuccess, cudaMemcpy(&value, dev + offset, sizeof(value), cudaMemcpyDeviceToHost));
    return value;
  }
};

/* A fully contiguous span past 2 GiB is one row whose pitch is its >= 2^31 width, which hipMemcpy3DAsync rejects. */
TEST_F(Memcpy3DTest, ContiguousRowAbove2GiB) {
  const size_t width = 2 * kGiB + 4096;
  if (!haveFree(2 * width)) GTEST_SKIP() << "needs " << 2 * width << " B of free device memory";
  char* src = alloc(width, 0x5a);
  char* dst = alloc(width, 0);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  const size_t probes[] = {0, kGiB, 2 * kGiB - 8, 2 * kGiB, width - 8};
  for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) put(src, probes[i], 0x1000 + i);

  cudaMemcpy3DParms cp = {};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(src, width, width, 1);
  cp.dstPtr = make_cudaPitchedPtr(dst, width, width, 1);
  cp.extent = make_cudaExtent(width, 1, 1);
  ASSERT_EQ(cudaSuccess, m2nMemcpy3DAsync(&cp, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) EXPECT_EQ(0x1000 + i, get(dst, probes[i]));
  EXPECT_EQ(0x5a5a5a5a5a5a5a5aull, get(dst, 3 * (kGiB / 2)));
}

/* Rows wider than 2 GiB with a destination pitch and position: each row must land at its own pitch. */
TEST_F(Memcpy3DTest, PitchedRowsAbove2GiB) {
  const size_t width = 2 * kGiB + 256;
  const size_t dstPitch = width + 4096;
  const size_t rows = 2;
  const size_t dstX = 512;
  if (!haveFree(rows * width + (rows + 1) * dstPitch)) GTEST_SKIP() << "not enough free device memory";
  char* src = alloc(rows * width, 0x11);
  char* dst = alloc((rows + 1) * dstPitch, 0);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  for (size_t r = 0; r < rows; r++) {
    put(src, r * width, 0xa0 + r);
    put(src, r * width + width - 8, 0xb0 + r);
  }

  cudaMemcpy3DParms cp = {};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(src, width, width, rows);
  cp.dstPtr = make_cudaPitchedPtr(dst, dstPitch, width, rows + 1);
  cp.dstPos = make_cudaPos(dstX, 1, 0);
  cp.extent = make_cudaExtent(width, rows, 1);
  ASSERT_EQ(cudaSuccess, m2nMemcpy3DAsync(&cp, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  EXPECT_EQ(0u, get(dst, 0));
  EXPECT_EQ(0u, get(dst, dstPitch + dstX - 8));
  for (size_t r = 0; r < rows; r++) {
    const size_t row = (1 + r) * dstPitch + dstX;
    EXPECT_EQ(0xa0 + r, get(dst, row));
    EXPECT_EQ(0xb0 + r, get(dst, row + width - 8));
    EXPECT_EQ(0x1111111111111111ull, get(dst, row + kGiB));
  }
}

/* Narrow rows whose source pitch (a >= 2 GiB outer stride) is past INT32_MAX: HIP rejects the pitch, not the width. */
TEST_F(Memcpy3DTest, NarrowRowsWithPitchAbove2GiB) {
  const size_t width = 4096, height = 2, depth = 2;
  const size_t srcPitch = 2 * kGiB + 256;
  const size_t srcX = 128;
  const size_t srcBytes = (height * depth - 1) * srcPitch + srcX + width;
  if (!haveFree(srcBytes)) GTEST_SKIP() << "needs " << srcBytes << " B of free device memory";
  char* src = alloc(srcBytes, 0x33);
  char* dst = alloc(width * height * depth, 0);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  for (size_t r = 0; r < height * depth; r++) {
    put(src, r * srcPitch + srcX, 0xd0 + r);
    put(src, r * srcPitch + srcX + width - 8, 0xe0 + r);
  }

  cudaMemcpy3DParms cp = {};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(src, srcPitch, width, height);
  cp.srcPos = make_cudaPos(srcX, 0, 0);
  cp.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
  cp.extent = make_cudaExtent(width, height, depth);
  ASSERT_EQ(cudaSuccess, m2nMemcpy3DAsync(&cp, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  for (size_t r = 0; r < height * depth; r++) {
    EXPECT_EQ(0xd0 + r, get(dst, r * width));
    EXPECT_EQ(0xe0 + r, get(dst, r * width + width - 8));
  }
  EXPECT_EQ(0x3333333333333333ull, get(dst, width / 2));
}

/* Narrow strided copies keep going through a single cudaMemcpy3DAsync. */
TEST_F(Memcpy3DTest, NarrowStridedCopy) {
  const size_t width = 256, height = 4, depth = 3, pitch = 1024;
  char* src = alloc(pitch * height * depth, 0x22);
  char* dst = alloc(width * height * depth, 0);
  ASSERT_NE(src, nullptr);
  ASSERT_NE(dst, nullptr);
  put(src, (2 * height + 3) * pitch + width - 8, 0xc0ffee);

  cudaMemcpy3DParms cp = {};
  cp.kind = cudaMemcpyDeviceToDevice;
  cp.srcPtr = make_cudaPitchedPtr(src, pitch, width, height);
  cp.dstPtr = make_cudaPitchedPtr(dst, width, width, height);
  cp.extent = make_cudaExtent(width, height, depth);
  ASSERT_EQ(cudaSuccess, m2nMemcpy3DAsync(&cp, stream));
  ASSERT_EQ(cudaSuccess, cudaStreamSynchronize(stream));

  EXPECT_EQ(0xc0ffeeull, get(dst, (2 * height + 3) * width + width - 8));
  EXPECT_EQ(0x2222222222222222ull, get(dst, 0));
}

}  // namespace
