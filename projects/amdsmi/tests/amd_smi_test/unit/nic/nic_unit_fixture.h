// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Shared gtest fixture for every unit/nic/*.cc file: the amdsmi-test-conventions
// hook requires one suite name (NicUnit) per component/tier, and gtest requires
// every test in a suite to share one fixture *type* (not just the same name
// used in different translation units). This header is the single definition
// both files include.

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstring>

#include "amd_smi/impl/amd_smi_nic_testing.h"

// Positions in amd::smi::nic_info_getters_t, in declaration order.
constexpr size_t kBus = 0;
constexpr size_t kDriver = 1;
constexpr size_t kAsic = 2;
constexpr size_t kNuma = 3;
constexpr size_t kPort = 4;
constexpr size_t kRdma = 5;
constexpr size_t kGetterCount = 6;

// inline (not anonymous-namespace): NicUnit's SetUp/Populate are defined in
// this header and included by multiple .cc files, so the linker folds their
// duplicate inline definitions into one. An anonymous-namespace g_status would
// give every TU its own copy, and whichever TU's fold survives would read a
// copy other TUs' tests never wrote to. inline gives it one true address
// everywhere, matching the single class definition that touches it.
inline std::array<smi_nic_status_t, kGetterCount> g_status;

// A byte unique to each getter position, so a swapped reinterpret_cast
// destination surfaces as the wrong byte instead of an indistinguishable zero.
constexpr unsigned char FillByte(size_t index) { return static_cast<unsigned char>(0xA0 + index); }

template <size_t Index, typename Info>
smi_nic_status_t Stub(smi_nic_ctx_t, uint64_t, Info* info) {
  std::memset(info, FillByte(Index), sizeof(*info));
  return g_status[Index];
}

// The RDMA stub fills like the rest rather than writing nothing: a non-zero fill
// is what the real getter leaves behind when it takes its ports.empty() early
// return without its own `*info = {}`, so the zeroing the tolerated path is
// asserted to perform still has to come from production.
inline const amd::smi::nic_info_getters_t kStubGetters = {
    Stub<kBus, smi_nic_bus_info_t>,   Stub<kDriver, smi_nic_driver_info_t>,
    Stub<kAsic, smi_nic_asic_info_t>, Stub<kNuma, smi_nic_numa_info_t>,
    Stub<kPort, smi_nic_port_info_t>, Stub<kRdma, smi_nic_rdma_devices_info_t>,
};

class NicUnit : public ::testing::Test {
 protected:
  void SetUp() override {
    g_status.fill(SMI_NIC_STATUS_SUCCESS);
    amd::smi::nic_set_info_getters_for_testing(&kStubGetters);
  }

  // Restores the production getters however a test exits.
  void TearDown() override { amd::smi::nic_set_info_getters_for_testing(nullptr); }

  amdsmi_status_t Populate() {
    smi_nic_ctx_t ctx = nullptr;
    info_ = {};
    std::memset(&info_.rdma_dev, 0xFF, sizeof(info_.rdma_dev));
    return amd::smi::populate_amd_ainic_device(ctx, 0x1000, info_);
  }

  amd::smi::AMDSmiAINICDevice::AINICInfo info_ = {};
};
