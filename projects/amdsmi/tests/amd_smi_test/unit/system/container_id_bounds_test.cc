// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Buffer-bounds contract of the container-ID extractors: an ID that does not
// fit in out_cap is reported as absent, never truncated, so a caller cannot
// mistake a clipped prefix for a complete ID.

#include <gtest/gtest.h>

#include <cstddef>
#include <string>

#include "amd_smi/impl/amd_smi_container_id_parser.h"
#include "container_id_test_util.h"
#include "guarded_buffer.h"

using amdsmi_test::GuardedBuffer;

TEST(SystemUnit, ContainerIdFullDockerIdIsNullTerminated) {
  constexpr size_t kIdLen = sizeof(amdsmi_test::kDocker64) - 1;
  const std::string line = std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  size_t n = amd::smi::ExtractContainerId(line, "docker", gb.buf, sizeof(gb.buf));
  EXPECT_EQ(n, kIdLen);
  EXPECT_EQ(std::string(gb.buf), amdsmi_test::kDocker64);
  EXPECT_EQ(gb.buf[kIdLen], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

// An ID longer than the destination is a truncation hazard: the caller cannot
// tell a clipped prefix from a complete ID, so the parser reports no ID.
TEST(SystemUnit, ContainerIdExceedingCapacityIsRejected) {
  const std::string line = std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<16> gb;
  size_t n = amd::smi::ExtractContainerId(line, "docker", gb.buf, sizeof(gb.buf));
  EXPECT_EQ(n, 0u);
  EXPECT_EQ(gb.buf[0], '\0');
  EXPECT_TRUE(gb.CanariesIntact());
}

// The accept/reject boundary sits exactly at out_cap - 1.
TEST(SystemUnit, ContainerIdCapacityBoundaryIsExact) {
  const std::string id(15, 'a');
  const std::string line = "0::/docker/" + id;
  {
    GuardedBuffer<16> gb;  // 15 bytes of ID + NUL: fits exactly
    EXPECT_EQ(amd::smi::ExtractContainerId(line, "docker", gb.buf, sizeof(gb.buf)), 15u);
    EXPECT_EQ(std::string(gb.buf), id);
    EXPECT_TRUE(gb.CanariesIntact());
  }
  {
    GuardedBuffer<15> gb;  // one byte short
    EXPECT_EQ(amd::smi::ExtractContainerId(line, "docker", gb.buf, sizeof(gb.buf)), 0u);
    EXPECT_TRUE(gb.CanariesIntact());
  }
}

TEST(SystemUnit, ContainerIdOverlongInputLeavesCanariesIntact) {
  std::string line = "0::/docker/";
  line.append(1024, 'z');
  GuardedBuffer<AMDSMI_MAX_STRING_LENGTH> gb;
  EXPECT_EQ(amd::smi::ExtractContainerId(line, "docker", gb.buf, sizeof(gb.buf)), 0u);
  EXPECT_TRUE(gb.CanariesIntact());
}

// Same refusal-over-truncation contract for the OCI extractor: a 64-char ID
// needs 65 bytes, and a caller that offers fewer gets "no ID", never a prefix.
TEST(SystemUnit, ContainerIdOciCapacityBoundaryIsExact) {
  const std::string line =
      std::string("0::/system.slice/docker-") + amdsmi_test::kDocker64 + ".scope";
  {
    GuardedBuffer<amd::smi::kOciContainerIdLength + 1> gb;
    EXPECT_EQ(amd::smi::ExtractOciContainerId(line, gb.buf, sizeof(gb.buf)),
              amd::smi::kOciContainerIdLength);
    EXPECT_EQ(std::string(gb.buf), amdsmi_test::kDocker64);
    EXPECT_TRUE(gb.CanariesIntact());
  }
  {
    GuardedBuffer<amd::smi::kOciContainerIdLength> gb;  // one byte short
    EXPECT_EQ(amd::smi::ExtractOciContainerId(line, gb.buf, sizeof(gb.buf)), 0u);
    EXPECT_EQ(gb.buf[0], '\0');
    EXPECT_TRUE(gb.CanariesIntact());
  }
  {
    GuardedBuffer<1> gb;
    EXPECT_EQ(amd::smi::ExtractOciContainerId(line, gb.buf, 0), 0u);
    EXPECT_TRUE(gb.CanariesIntact());
  }
}

TEST(SystemUnit, ContainerIdZeroCapacityBufferIsNotWritten) {
  const std::string line = std::string("0::/docker/") + amdsmi_test::kDocker64;
  GuardedBuffer<1> gb;
  EXPECT_EQ(amd::smi::ExtractContainerId(line, "docker", gb.buf, 0), 0u);
  EXPECT_TRUE(gb.CanariesIntact());
}
