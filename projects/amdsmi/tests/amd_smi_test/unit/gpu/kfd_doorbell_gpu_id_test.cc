// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <gtest/gtest.h>

#include <cstdint>

#include "rocm_smi/rocm_smi_kfd.h"

using amd::smi::KfdDoorbellGpuId;
using amd::smi::MapsLine;
using amd::smi::ParseMapsLine;

// /proc/<pid>/maps lines of a HIP process, whose /dev/kfd is inode 4 of the
// file system with device 00:05.
TEST(GpuUnit, ParseMapsLineReadsOffsetDeviceAndInode) {
  MapsLine line;
  ASSERT_TRUE(
      ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s dcb1000000000000 00:05 4"
                    "                          /dev/kfd",
                    &line));
  EXPECT_EQ(line.offset, 0xdcb1000000000000u);
  EXPECT_EQ(line.dev_major, 0u);
  EXPECT_EQ(line.dev_minor, 5u);
  EXPECT_EQ(line.inode, 4u);
  // Device numbers are hexadecimal, the inode decimal; the path may be any.
  ASSERT_TRUE(ParseMapsLine(
      "55d0c8a00000-55d0c8a01000 r--p 00001000 fd:1a 131075 /x/dev/kfd (deleted)", &line));
  EXPECT_EQ(line.offset, 0x1000u);
  EXPECT_EQ(line.dev_major, 0xfdu);
  EXPECT_EQ(line.dev_minor, 0x1au);
  EXPECT_EQ(line.inode, 131075u);
  // An anonymous mapping ends after its inode.
  ASSERT_TRUE(ParseMapsLine("7ffd4a1c0000-7ffd4a1e1000 rw-p 00001000 00:00 0", &line));
  EXPECT_EQ(line.offset, 0x1000u);
  EXPECT_EQ(line.dev_major, 0u);
  EXPECT_EQ(line.dev_minor, 0u);
  EXPECT_EQ(line.inode, 0u);
}

TEST(GpuUnit, ParseMapsLineRejectsOtherLines) {
  MapsLine line;
  EXPECT_FALSE(ParseMapsLine("", &line));
  // A line cut short after its device, or with its newline left on.
  EXPECT_FALSE(ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s dcb1000000000000 00:05", &line));
  EXPECT_FALSE(ParseMapsLine("7ffd4a1c0000-7ffd4a1e1000 rw-p 00000000 00:00 0\n", &line));
  EXPECT_FALSE(ParseMapsLine("/dev/kfd", &line));
  EXPECT_FALSE(ParseMapsLine("not a mapping /dev/kfd", &line));
  // An offset that is not hexadecimal throughout, wider than 64 bits, or empty.
  EXPECT_FALSE(
      ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s dcb100000000000z 00:05 4 /dev/kfd", &line));
  EXPECT_FALSE(
      ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s 1dcb1000000000000 00:05 4 /dev/kfd", &line));
  EXPECT_FALSE(ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s  00:05 4 /dev/kfd", &line));
  // A device without its minor, or an inode that is not decimal.
  EXPECT_FALSE(
      ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s dcb1000000000000 0005 4 /dev/kfd", &line));
  EXPECT_FALSE(
      ParseMapsLine("7f2be0c00000-7f2be0c02000 rw-s dcb1000000000000 00:05 4a /dev/kfd", &line));
}

// A doorbell offset holds type 3 in bits 62-63 and the GPU's KFD id in bits
// 46-61: dcb1000000000000 is (3 << 62) | (29380 << 46).
TEST(GpuUnit, KfdDoorbellGpuIdFromQueueDoorbells) {
  EXPECT_EQ(KfdDoorbellGpuId(0xdcb1000000000000u), 29380u);
  EXPECT_EQ(KfdDoorbellGpuId(0xd092800000000000u), 16970u);
  // Lower bits, as KFD sets for a queue's doorbell inside its process's
  // doorbell pages, leave the id as it is.
  EXPECT_EQ(KfdDoorbellGpuId(0xdcb1000000002008u), 29380u);
  // The largest KFD id that the offset holds.
  EXPECT_EQ(KfdDoorbellGpuId(0xffffc00000000000u), 0xffffu);
}

TEST(GpuUnit, KfdDoorbellGpuIdIgnoresOtherMappings) {
  // KFD also maps every GPU's MMIO page (type 0), reserved memory (type 1)
  // and the events page (type 2), with or without queues.
  EXPECT_EQ(KfdDoorbellGpuId(0x1cb1000000000000u), 0u);
  EXPECT_EQ(KfdDoorbellGpuId(0x5cb1000000000000u), 0u);
  EXPECT_EQ(KfdDoorbellGpuId(0x8000000000000000u), 0u);
  EXPECT_EQ(KfdDoorbellGpuId(0), 0u);
  // A doorbell offset for KFD id 0, which names no GPU.
  EXPECT_EQ(KfdDoorbellGpuId(0xc000000000000000u), 0u);
}
