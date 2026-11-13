// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
#define AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_

#include <cstdint>
#include <functional>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_test_flags.h"

// The per-component step of amdsmi_get_cuid_components(): fill(i, &component)
// for each of |total| components, keeping in |found| those it fills. A
// component whose fill fails is left out; AMDSMI_STATUS_NO_DATA (no
// machine-id), AMDSMI_STATUS_INVAL and AMDSMI_STATUS_INSUFFICIENT_SIZE fail
// the call with |found| empty. Declared here so the unit tests can make one
// component fail, which no fake sysfs tree can do between the listing and the
// read.
amdsmi_status_t amdsmi_cuid_collect_components(
    uint32_t total, const std::function<amdsmi_status_t(uint32_t, amdsmi_cuid_component_t*)>& fill,
    std::vector<amdsmi_cuid_component_t>* found);

// Internal test-only wrapper around rsmi_test_sleep. Acquires the device mutex
// for |seconds| seconds and returns an amdsmi_status_t so tests do not need to
// extern-declare the rsmi_status_t function directly.
amdsmi_status_t amdsmi_test_sleep(amdsmi_processor_handle processor_handle, uint32_t seconds);

// Internal test-only hook: registers |node_handle| as a valid handle in the
// same node-handle registry that amdsmi_get_npm_info()/amdsmi_set_npm_limit()
// consult (via is_registered_node_handle()) before dereferencing their
// node_handle argument. In production, that registry is only ever populated
// by amdsmi_get_node_handle(); unit tests that fabricate a raw node_handle
// directly (bypassing amdsmi_get_node_handle(), whose oam_id == 0 gate is not
// satisfiable in the test environment) must call this first so the
// registered-handle check does not reject their forged-but-otherwise-valid
// handle. Not declared in amdsmi.h -- internal use via
// amd_smi_test_internal.h only.
// Unlike the other hooks in this header, this one lets a caller mark an
// arbitrary pointer as a "valid" node handle, so it is compiled only for
// BUILD_TESTS=ON builds (see src/CMakeLists.txt) rather than shipped in
// production builds.
#ifdef BUILD_TESTS
amdsmi_status_t amdsmi_test_register_node_handle(amdsmi_node_handle node_handle);
#endif  // BUILD_TESTS

#endif  // AMD_SMI_INCLUDE_AMD_SMI_TEST_INTERNAL_H_
