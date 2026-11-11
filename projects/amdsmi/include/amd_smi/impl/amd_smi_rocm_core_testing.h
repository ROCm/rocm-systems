// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

// Library-local test seam for amdsmi_get_rocm_version()'s librocm-core.so
// discovery and status mapping. Not amdsmi_-prefixed, so the linker version
// script keeps it out of libamd_smi.so; tests reach it through the static
// archive. Shared by the definition (src/amd_smi/amd_smi.cc) and the unit
// tests so the signature stays in sync.

// Ordered librocm-core.so search candidates: the directory this library
// itself loaded from (relocatable installs), ROCM_HOME/ROCM_PATH, then
// /opt/rocm/lib, each with both soname variants, falling back to the bare
// sonames for the dynamic linker's own search path.
std::vector<std::string> rocm_core_candidates();

// Maps getROCmVersion()'s raw output to the amdsmi_version_t/status pair
// amdsmi_get_rocm_version() returns. build_str must outlive version->build,
// which points into it.
amdsmi_status_t rocm_core_status_to_amdsmi(uint32_t rocm_core_status, uint32_t major,
                                           uint32_t minor, uint32_t release,
                                           amdsmi_version_t* version, std::string* build_str);
