/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Guards amd-smi's vendored view of the librocdxg ABI against upstream drift.
// librocdxg writes into structs amd-smi allocates, so a silent layout change
// would corrupt memory rather than fail to compile. Every check below is a
// static_assert, so compiling this target is the test.

#include <cstddef>

// Vendored copy under test.
#include "amd_smi/impl/wsl/rocdxg_abi.h"

// Upstream definitions. Pulled into a namespace so the two sets of identically
// named types can coexist in one translation unit.
namespace upstream {
#include "hsakmt/rocdxg_smi.h"
}  // namespace upstream

#define ABI_ASSERT_SIZE(type)                                    \
  static_assert(sizeof(type) == sizeof(upstream::type), #type    \
                " size differs from upstream librocdxg; update " \
                "include/amd_smi/impl/wsl/rocdxg_abi.h")

#define ABI_ASSERT_FIELD(type, field)                                     \
  static_assert(offsetof(type, field) == offsetof(upstream::type, field), \
                #type "." #field                                          \
                      " offset differs from upstream librocdxg; "         \
                      "update include/amd_smi/impl/wsl/rocdxg_abi.h")

ABI_ASSERT_SIZE(HsaSystemProperties);
ABI_ASSERT_FIELD(HsaSystemProperties, NumNodes);

ABI_ASSERT_SIZE(rocdxg_smi_bdf_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_asic_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_board_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_vram_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_vram_usage_t);
ABI_ASSERT_SIZE(rocdxg_smi_power_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_clock_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_pcie_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_driver_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_vbios_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_gpu_metrics_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_process_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_cache_entry_t);
ABI_ASSERT_SIZE(rocdxg_smi_cache_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_fw_entry_t);
ABI_ASSERT_SIZE(rocdxg_smi_fw_info_t);
ABI_ASSERT_SIZE(rocdxg_smi_device_info_t);

// Sub-struct offsets inside the aggregate: catches a field added to an earlier
// member, which shifts everything after it without changing any single size.
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, struct_size);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, bdf);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, asic);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, board);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, vram);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, driver);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, vbios);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, cache);
ABI_ASSERT_FIELD(rocdxg_smi_device_info_t, fw);

// struct_size only protects callers if it sits at offset 0, where a library
// built from any layout can still read it.
static_assert(offsetof(rocdxg_smi_device_info_t, struct_size) == 0,
              "struct_size must be the first member of rocdxg_smi_device_info_t");

ABI_ASSERT_FIELD(rocdxg_smi_asic_info_t, num_xcc);
ABI_ASSERT_FIELD(rocdxg_smi_asic_info_t, target_graphics_version);

// Status codes are exchanged by value; mismatched numbering would silently
// remap errors.
static_assert(static_cast<int>(HSAKMT_STATUS_SUCCESS) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_SUCCESS),
              "HSAKMT_STATUS_SUCCESS differs from upstream");
static_assert(static_cast<int>(HSAKMT_STATUS_NOT_SUPPORTED) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_NOT_SUPPORTED),
              "HSAKMT_STATUS_NOT_SUPPORTED differs from upstream");
static_assert(static_cast<int>(HSAKMT_STATUS_INVALID_PARAMETER) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_INVALID_PARAMETER),
              "HSAKMT_STATUS_INVALID_PARAMETER differs from upstream");
static_assert(static_cast<int>(HSAKMT_STATUS_INVALID_NODE_UNIT) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_INVALID_NODE_UNIT),
              "HSAKMT_STATUS_INVALID_NODE_UNIT differs from upstream");
static_assert(static_cast<int>(HSAKMT_STATUS_KERNEL_ALREADY_OPENED) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_KERNEL_ALREADY_OPENED),
              "HSAKMT_STATUS_KERNEL_ALREADY_OPENED differs from upstream");
static_assert(static_cast<int>(HSAKMT_STATUS_BUFFER_TOO_SMALL) ==
                  static_cast<int>(upstream::HSAKMT_STATUS_BUFFER_TOO_SMALL),
              "HSAKMT_STATUS_BUFFER_TOO_SMALL differs from upstream");

int main() { return 0; }
