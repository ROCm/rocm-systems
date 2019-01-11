/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_COMMON_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_COMMON_H_

#define DBG_FILE_ERROR(STR) \
  if (env_->debug_output_bitfield & RSMI_DEBUG_SYSFS_FILE_PATHS) { \
    std::cout << "*****" << __FUNCTION__ << std::endl; \
    std::cout << "*****Opening file: " << (STR) << std::endl; \
    std::cout << " at " << __FILE__ << ":" << __LINE__ << std::endl;\
  }

// Add different debug filters here, as powers of 2; e.g, 1, 2, 4, 8, ...
#define RSMI_DEBUG_SYSFS_FILE_PATHS 1

struct RocmSMI_env_vars {
    // Store env. variables here
    uint32_t debug_output_bitfield;
    uint32_t enum_override;
    const char *path_DRM_root_override;
    const char *path_HWMon_root_override;
    const char *path_power_root_override;
};

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_COMMON_H_
