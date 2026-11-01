// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "amd_smi/impl/nic/amd_smi_ainic_device.h"
#include "amd_smi/impl/nic/amd_smi_nic_interface.h"

// Included bare: the header opens its own extern "C", and wrapping it again
// would extend C linkage over the C++-only declarations at its tail.
#include "amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h"

namespace amd::smi {

// Library-local test seam: swaps the smi_get_nic_* getters that AI-NIC
// discovery calls, so unit tests can drive per-getter status codes (notably the
// optional RDMA path) on a host with no NIC. Not amdsmi_-prefixed, so the
// linker version script keeps these out of libamd_smi.so (no public ABI); tests
// reach them through the static archive.
//
// Shared by the definitions (src/amd_smi/amd_smi_system.cc) and the tests
// (tests/amd_smi_test/unit/nic/rdma_status_test.cc) so the signatures stay in
// sync.
struct nic_info_getters_t {
  smi_nic_status_t (*bus)(smi_nic_ctx_t, uint64_t, smi_nic_bus_info_t*);
  smi_nic_status_t (*driver)(smi_nic_ctx_t, uint64_t, smi_nic_driver_info_t*);
  smi_nic_status_t (*asic)(smi_nic_ctx_t, uint64_t, smi_nic_asic_info_t*);
  smi_nic_status_t (*numa)(smi_nic_ctx_t, uint64_t, smi_nic_numa_info_t*);
  smi_nic_status_t (*port)(smi_nic_ctx_t, uint64_t, smi_nic_port_info_t*);
  smi_nic_status_t (*rdma)(smi_nic_ctx_t, uint64_t, smi_nic_rdma_devices_info_t*);
  smi_nic_status_t (*kind)(smi_nic_ctx_t, uint64_t, uint8_t*);
  smi_nic_status_t (*capability)(smi_nic_ctx_t, uint64_t, uint32_t*);
  smi_nic_status_t (*rdma_mtu)(smi_nic_ctx_t, uint64_t, smi_nic_rdma_devices_info_t*);
  // Not an info getter: listed so a test can observe the state NIC context creation runs in.
  smi_nic_status_t (*create_context)(smi_nic_ctx_t*);
};

// Passing nullptr restores the production getters. Not thread-safe: only the
// single-threaded tests call it; production never does.
void nic_set_info_getters_for_testing(const nic_info_getters_t* getters);

// The getters amdsmi_init uses, so a test can check how they are wired.
const nic_info_getters_t& nic_production_info_getters_for_testing();

// Fills one AI-NIC's info from the getters above. Declared here only so the
// tests can reach it; the sole production caller is populate_amd_ainic_devices().
amdsmi_status_t populate_amd_ainic_device(const smi_nic_ctx_t& ctx, uint64_t bdf_int,
                                          AMDSmiAINICDevice::AINICInfo& ai_nic_info);

// Fills the MTU of the RDMA ports in rdma_dev through the rdma_mtu getter. Declared here so the
// device (which fills on first read) and the tests share it; defined in amd_smi_system.cc.
smi_nic_status_t fill_ainic_rdma_mtu(smi_nic_ctx_t ctx, uint64_t bdf_int,
                                     amdsmi_nic_rdma_devices_info_t& rdma_dev);

}  // namespace amd::smi
