/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef ROCSHMEM_AMDSMI_LOADER_HPP_
#define ROCSHMEM_AMDSMI_LOADER_HPP_

#include <cstdint>

namespace rocshmem {

// Forward declarations and type definitions to avoid compile-time dependency on AMD SMI headers
// These definitions match the ABI of libamd_smi.so

// AMD SMI status type
typedef uint32_t amdsmi_status_t;

// AMD SMI processor handle (opaque pointer)
typedef void* amdsmi_processor_handle;

// AMD SMI GPU fabric info structure (version 1)
// Based on AMD SMI library interface
typedef struct {
  uint32_t ppod_id;          // Physical pod ID
  uint32_t vpod_id;          // Virtual pod ID
  uint32_t reserved[62];     // Reserved fields to match full structure size
} amdsmi_gpu_fabric_info_v1_t;

// AMD SMI GPU fabric info union wrapper
typedef struct {
  union {
    amdsmi_gpu_fabric_info_v1_t v1;
  } info;
} amdsmi_gpu_fabric_info_t;

// AMD SMI initialization flags
#define AMDSMI_INIT_AMD_GPUS (1 << 1)

// AMD SMI status codes
#define AMDSMI_STATUS_SUCCESS 0

/**
 * Structure to hold dynamically loaded AMD SMI function pointers
 */
struct AmdsmiLoader {
  void* amdsmi_handle;

  // Function pointers
  amdsmi_status_t (*init)(uint64_t init_flags);
  amdsmi_status_t (*shut_down)();
  amdsmi_status_t (*get_processor_handle_from_bdf)(const char* bdf, amdsmi_processor_handle* processor_handle);
  amdsmi_status_t (*get_gpu_fabric_info)(amdsmi_processor_handle processor_handle, amdsmi_gpu_fabric_info_t* fabric_info);

  AmdsmiLoader();
  ~AmdsmiLoader();

  int init_function_table();
  bool isLoaded() const;
};

}  // namespace rocshmem

#endif  // ROCSHMEM_AMDSMI_LOADER_HPP_
