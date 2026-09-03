// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pins the SMN loop-back count: N entries must yield exactly N passes,
// and a zero count must not re-enter the loop past the end of the image.

#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rocm_smi/rocm_smi.h"
#include "unit_fixtures.h"

namespace amd::smi {
int present_reg_state(const char* fname, rsmi_reg_type_t reg_type, rsmi_name_value_t** kv,
                      uint32_t* kvnum);
}  // namespace amd::smi

namespace {

// Mirrors xgmi_regs[]: 8B header, 8B instance header, then 16B per SMN entry.
std::string WriteXgmiImage(uint16_t num_smn_regs) {
  const size_t size = 16 + static_cast<size_t>(num_smn_regs) * 16;
  std::vector<uint8_t> buf(size, 0);

  const uint16_t structure_size = static_cast<uint16_t>(size);
  std::memcpy(&buf[0], &structure_size, sizeof(structure_size));
  buf[5] = 1;  // num_instances
  std::memcpy(&buf[12], &num_smn_regs, sizeof(num_smn_regs));

  for (uint16_t i = 0; i < num_smn_regs; ++i) {
    const size_t offset = 16 + static_cast<size_t>(i) * 16;
    const uint64_t addr = static_cast<uint64_t>(i) + 1;
    const uint32_t value = static_cast<uint32_t>(i) * 10;
    std::memcpy(&buf[offset], &addr, sizeof(addr));
    std::memcpy(&buf[offset + 8], &value, sizeof(value));
  }

  char path[] = "/tmp/amdsmi_xgmi_regs_XXXXXX";
  const int fd = mkstemp(path);
  EXPECT_GE(fd, 0);
  EXPECT_EQ(write(fd, buf.data(), size), static_cast<ssize_t>(size));
  close(fd);
  return std::string(path);
}

// 6 header fields + 4 instance-header fields + 3 fields per SMN entry.
uint32_t ExpectedPairs(uint16_t num_smn_regs) {
  return 10U + 3U * static_cast<uint32_t>(num_smn_regs);
}

void ExpectSmnPasses(uint16_t num_smn_regs) {
  const std::string path = WriteXgmiImage(num_smn_regs);
  rsmi_name_value_t* kv = nullptr;
  uint32_t kvnum = 0;

  EXPECT_EQ(amd::smi::present_reg_state(path.c_str(), RSMI_REG_XGMI, &kv, &kvnum), 0);
  EXPECT_EQ(kvnum, ExpectedPairs(num_smn_regs));

  free(kv);
  remove(path.c_str());
}

// Zero entries is the case that regressed: the loop-back test ran before
// the counter reached zero, so it re-read past the end of the image.

// Header plus one instance header per instance, no SMN entries.
std::string WriteXgmiInstancesOnly(uint8_t num_instances) {
  const size_t size = 8 + static_cast<size_t>(num_instances) * 8;
  std::vector<uint8_t> buf(size, 0);
  const uint16_t structure_size = static_cast<uint16_t>(size);
  std::memcpy(&buf[0], &structure_size, sizeof(structure_size));
  buf[5] = num_instances;
  char path[] = "/tmp/amdsmi_xgmi_multi_XXXXXX";
  const int fd = mkstemp(path);
  EXPECT_GE(fd, 0);
  EXPECT_EQ(write(fd, buf.data(), size), static_cast<ssize_t>(size));
  close(fd);
  return std::string(path);
}

// The skip-SMN path must stop on the last instance; it previously
// looped back and read an instance header past the end of the image.
TEST_F(GpuUnit, BinaryParserXgmiTwoInstancesZeroSmnRegs) {
  const std::string path = WriteXgmiInstancesOnly(2);
  rsmi_name_value_t* kv = nullptr;
  uint32_t kvnum = 0;
  EXPECT_EQ(amd::smi::present_reg_state(path.c_str(), RSMI_REG_XGMI, &kv, &kvnum), 0);
  EXPECT_EQ(kvnum, 14U);
  free(kv);
  remove(path.c_str());
}

TEST_F(GpuUnit, BinaryParserXgmiZeroSmnRegs) { ExpectSmnPasses(0); }

TEST_F(GpuUnit, BinaryParserXgmiOneSmnReg) { ExpectSmnPasses(1); }

TEST_F(GpuUnit, BinaryParserXgmiTwoSmnRegs) { ExpectSmnPasses(2); }

TEST_F(GpuUnit, BinaryParserXgmiThreeSmnRegs) { ExpectSmnPasses(3); }

}  // namespace
