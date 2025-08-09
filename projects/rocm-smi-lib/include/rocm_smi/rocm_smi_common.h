/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2018-2025, Advanced Micro Devices, Inc.
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

#include <cstdint>
#include <memory>
#include <map>
#include <vector>
#include <string>
#include <unordered_set>

#define CHECK_DV_IND_RANGE \
    amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance(); \
    if (dv_ind >= smi.devices().size()) { \
      return RSMI_STATUS_INVALID_ARGS; \
    } \

#define GET_DEV_FROM_INDX  \
  CHECK_DV_IND_RANGE \
  std::shared_ptr<amd::smi::Device> dev = smi.devices()[dv_ind]; \
  assert(dev != nullptr);


#define GET_DEV_AND_KFDNODE_FROM_INDX \
  GET_DEV_FROM_INDX \
  std::shared_ptr<amd::smi::KFDNode> kfd_node; \
  if (smi.kfd_node_map().find(dev->kfd_gpu_id()) == \
                                                 smi.kfd_node_map().end()) { \
    return RSMI_INITIALIZATION_ERROR; \
  } \
  kfd_node = smi.kfd_node_map()[dev->kfd_gpu_id()];

#define REQUIRE_ROOT_ACCESS \
    if (amd::smi::RocmSMI::getInstance().euid()) { \
      return RSMI_STATUS_PERMISSION; \
    }

#define DEVICE_MUTEX \
    amd::smi::pthread_wrap _pw(*amd::smi::GetMutex(dv_ind)); \
    amd::smi::RocmSMI& smi_ = amd::smi::RocmSMI::getInstance(); \
    bool blocking_ = !(smi_.init_options() & \
                          static_cast<uint64_t>(RSMI_INIT_FLAG_RESRV_TEST1)); \
    amd::smi::ScopedPthread _lock(_pw, blocking_); \
    if (!blocking_ && _lock.mutex_not_acquired()) { \
      return RSMI_STATUS_BUSY; \
    }

/* This group of macros is used to facilitate checking of support for rsmi_dev*
 * "getter" functions. When the return buffer is set to nullptr, the macro will
 * check the previously gathered device support data to see if the function,
 * with possible variants (e.g., memory types, firmware types,...) and
 * subvariants (e.g. monitors/sensors) are supported.
 */
// This macro assumes dev already available
#define CHK_API_SUPPORT_ONLY(RT_PTR, VR, SUB_VR) \
    if ((RT_PTR) == nullptr) { \
      try { \
        if (!dev->DeviceAPISupported(__FUNCTION__, (VR), (SUB_VR))) { \
          return RSMI_STATUS_NOT_SUPPORTED; \
        }  \
        return RSMI_STATUS_INVALID_ARGS; \
      } catch (const amd::smi::rsmi_exception& e) { \
        debug_print( \
             "Exception caught when checking if API is supported %s.\n", \
                                                                  e.what()); \
        return RSMI_STATUS_INVALID_ARGS; \
      } \
    }

#define CHK_SUPPORT(RT_PTR, VR, SUB_VR)  \
    GET_DEV_FROM_INDX \
    CHK_API_SUPPORT_ONLY((RT_PTR), (VR), (SUB_VR))

#define CHK_SUPPORT_NAME_ONLY(RT_PTR) \
    CHK_SUPPORT((RT_PTR), RSMI_DEFAULT_VARIANT, RSMI_DEFAULT_VARIANT) \

#define CHK_SUPPORT_VAR(RT_PTR, VR) \
    CHK_SUPPORT((RT_PTR), (VR), RSMI_DEFAULT_VARIANT) \

#define CHK_SUPPORT_SUBVAR_ONLY(RT_PTR, SUB_VR) \
    CHK_SUPPORT((RT_PTR), RSMI_DEFAULT_VARIANT, (SUB_VR)) \

#define DBG_FILE_ERROR(FN, WR_STR) \
  if (env_ && env_->debug_output_bitfield & RSMI_DEBUG_SYSFS_FILE_PATHS) { \
    std::cout << "*****" << __FUNCTION__ << std::endl; \
    std::cout << "*****Opening file: " << (FN) << std::endl; \
    if ((WR_STR) != nullptr) { \
       std::cout << "***** for writing. Writing: \"" << (WR_STR) << "\""; \
    } else { std::cout << "***** for reading.";} \
    std::cout << std::endl; \
    std::cout << " at " << __FILE__ << ":" << std::dec << __LINE__ << \
                                                                   std::endl;\
  }

#define DEBUG_LOG(WR_STR, VR) \
  amd::smi::RocmSMI& smi = amd::smi::RocmSMI::getInstance(); \
  if (smi.getEnv().debug_output_bitfield & RSMI_DEBUG_VAL) { \
      if ((WR_STR) != nullptr)  \
          std::cout << (WR_STR) << " " << (VR) << std::endl;\
  }

// Add different debug filters here, as powers of 2; e.g, 1, 2, 4, 8, ...
#define RSMI_DEBUG_SYSFS_FILE_PATHS 1<<0
#define RSMI_DEBUG_VAL 1<<1

struct rsmi_func_id_iter_handle {
    uintptr_t func_id_iter;
    uintptr_t container_ptr;
    uint32_t id_type;
};

struct RocmSMI_env_vars {
    // If RSMI_DEBUG_INFINITE_LOOP is non-zero, rsmi_init() will go into
    // an infinite loop in debug builds. For release builds, this is
    // ignored. This is useful for debugging RSMI applications with
    // gdb. After attaching with gdb, the inf. loop can be exited and
    // RSMI can be debugged.
    uint32_t debug_inf_loop;

    // Bitfield that is AND'd with various RSMI_DEBUG_* bits to determine
    // which debugging information should be turned on. Env. variable
    // RSMI_DEBUG_BITFIELD is used to set all the debug info bits.
    uint32_t debug_output_bitfield;

    // The integer value of sysfs field enum that is to be over-ridden.
    // Env. variable RSMI_DEBUG_ENUM_OVERRIDE is used to specify this.
    // A set of enum overrides, RSMI_DEBUG_ENUM_OVERRIDE now supports
    // comma delimited values.
    std::unordered_set<uint32_t> enum_overrides;

    // If RSMI_LOGGING is set, enables logging.
    // Otherwise unset values, signify logging is turned off.
    uint32_t logging_on;

    // Sysfs path overrides

    // Env. var. RSMI_DEBUG_DRM_ROOT_OVERRIDE
    const char *path_DRM_root_override;

    // Env. var. RSMI_DEBUG_HWMON_ROOT_OVERRIDE
    const char *path_HWMon_root_override;

    // Env. var. RSMI_DEBUG_PP_ROOT_OVERRIDE
    const char *path_power_root_override;
};

// Use this bit offset to store the label-mapped file index
#define MONITOR_TYPE_BIT_POSITION 16
#define MONITOR_IND_BIT_MASK ((1 << MONITOR_TYPE_BIT_POSITION) - 1)

// Support information data structures
typedef std::vector<uint64_t> SubVariant;
typedef SubVariant::const_iterator SubVariantIt;

typedef std::map<uint64_t, std::shared_ptr<SubVariant>> VariantMap;
typedef VariantMap::const_iterator VariantMapIt;

typedef std::map<std::string, std::shared_ptr<VariantMap>> SupportedFuncMap;
typedef SupportedFuncMap::const_iterator SupportedFuncMapIt;

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_COMMON_H_
