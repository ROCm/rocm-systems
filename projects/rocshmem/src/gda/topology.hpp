/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *****************************************************************************/

#pragma once
#include <algorithm>
#include <cstring>
#include <future>
#include <map>
#include "numa_wrapper.hpp"
#include <random>
#include <set>
#include <sstream>
#include <stdarg.h>
#include <thread>
#include <unistd.h>
#include <vector>
#include <iostream>

#include <infiniband/verbs.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <arpa/inet.h>
#include <fcntl.h>
#include <unistd.h>
#include <filesystem>
#include <fstream>

#include <hip/hip_ext.h>
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>

#include "util.hpp"

namespace rocshmem
{
  using std::map;
  using std::pair;
  using std::set;
  using std::vector;

  /**
   * Enumeration of GID priority
   *
   * @note These are the GID types ordered in priority from lowest (0) to highest
   */
  enum GidPriority
  {
    UNKNOWN           = -1,                      ///< Default
    ROCEV1_LINK_LOCAL = 0,                       ///< RoCEv1 Link-local
    ROCEV2_LINK_LOCAL = 1,                       ///< RoCEv2 Link-local fe80::/10
    ROCEV1_IPV6       = 2,                       ///< RoCEv1 IPv6
    ROCEV2_IPV6       = 3,                       ///< RoCEv2 IPv6
    ROCEV1_IPV4       = 4,                       ///< RoCEv1 IPv4-mapped IPv6
    ROCEV2_IPV4       = 5,                       ///< RoCEv2 IPv4-mapped IPv6 ::ffff:192.168.x.x
  };


  /**
   * Enumeration of supported memory types
   *
   * @note These are possible types of memory to be used as sources/destinations
   */
  enum MemType
  {
    MEM_CPU          = 0,                       ///< Coarse-grained pinned CPU memory
    MEM_GPU          = 1,                       ///< Coarse-grained global GPU memory
  };

 /**
   * Enumeration of supported Executor types
   *
   * @note The Executor is the device used to perform a Transfer
   * @note IBVerbs executor is currently not implemented yet
   */

  enum DeviceType
  {
    EXE_CPU          = 0,
    EXE_GPU          = 1,
    EXE_NIC          = 2
  };

  inline bool IsCpuExeType(DeviceType e){ return e == EXE_CPU; }
  inline bool IsGpuExeType(DeviceType e){ return e == EXE_GPU; }
  inline bool IsNicExeType(DeviceType e){ return e == EXE_NIC; }

  /**
   * A ExeDevice defines a specific Executor
   */
  struct ExeDevice
  {
    DeviceType exeType;                         ///< Device type
    int32_t exeIndex;                           ///< Device index

    bool operator<(ExeDevice const& other) const {
      return (exeType < other.exeType) || (exeType == other.exeType && exeIndex < other.exeIndex);
    }
  };


  /**
   * A MemDevice indicates a memory type on a specific device
   */
  struct MemDevice
  {
    MemType memType;                            ///< Memory type
    int32_t memIndex;                           ///< Device index

    bool operator<(MemDevice const& other) const {
      return (memType < other.memType) || (memType == other.memType && memIndex < other.memIndex);
    }
  };

  inline bool IsCpuMemType(MemType m) { return (m == MEM_CPU); }
  inline bool IsGpuMemType(MemType m) { return (m == MEM_GPU); }

  /**
   * Returns the index of the NUMA node closest to the given GPU
   *
   * @param[in] gpuIndex Index of the GPU to query
   * @returns NUMA node index closest to GPU gpuIndex, or -1 if unable to detect
   */
  int GetClosestCpuNumaToGpu(int gpuIndex);

  /**
   * Returns the index of the NUMA node closest to the given NIC
   *
   * @param[in] nicIndex Index of the NIC to query
   * @returns NUMA node index closest to the NIC nicIndex, or -1 if unable to detect
   */
  int GetClosestCpuNumaToNic(int nicIndex);

  /**
   * Returns the index of the NIC closest to the given GPU
   *
   * @param[in] gpuIndex Index of the GPU to query
   * @param[in] hca_list Include list of device names that can be used (Exclude if prefixed by ^)
   * @param[out] dev_name Name of of IB Verbs capable NIC index closest to GPU gpuIndex
   * @returns index of IB Verbs capable NIC index closest to GPU gpuIndex, or -1 if unable to detect
   */
  int GetClosestNicToGpu(int gpuIndex, const char* hca_list, const char** dev_name);

  /**
   * Returns information about number of available Devices
   *
   * @param[in]  Type    Hardware Device type to query
   * @returns    Number of detected Devices of type Type
   */
  int GetNumDevices(DeviceType Type);

  void DisplayTopology(bool outputToCsv);

};

//==========================================================================================
// End of rocshmem API
//==========================================================================================

// Error check macros
#define ROCSHMEM_SUCCESS 0

#define ERR_CHECK(cmd)            \
  do {                            \
    int error = cmd;                                                      \
    if (error != 0) {                                                \
      fprintf(stderr, "error: %d at %s:%d\n", error, __FILE__, __LINE__);     \
      exit(EXIT_FAILURE);                                                     \
    }                                                                         \
} while (0)

// Helper macros for calling RDMA functions and reporting errors
#ifdef VERBS_DEBUG
#define IBV_CALL(__func__, ...)                                         \
  do {                                                                  \
    int error = __func__(__VA_ARGS__);                                  \
    if (error != 0) {                                                   \
      fprintf(stderr,"Encountered IbVerbs error (%d) at line (%d) "        \
              "and function (%s)", (error), __LINE__, #__func__);       \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define IBV_PTR_CALL(__ptr__, __func__, ...)                               \
  do {                                                                     \
    __ptr__ = __func__(__VA_ARGS__);                                       \
    if (__ptr__ == nullptr) {                                              \
      fprintf(stderr, "Encountered IbVerbs nullptr error at line (%d) " \
              "and function (%s)", __LINE__, #__func__);                   \
      exit(EXIT_FAILURE);                                               \
    }                                                                      \
  } while (0)
#else
#define IBV_CALL(__func__, ...)                                         \
  do {                                                                  \
    int error = __func__(__VA_ARGS__);                                  \
    if (error != 0) {                                                   \
      fprintf(stderr, "Encountered IbVerbs error (%d) in func (%s) " \
              , error, #__func__);                                      \
      exit(EXIT_FAILURE);                                               \
    }                                                                   \
  } while (0)

#define IBV_PTR_CALL(__ptr__, __func__, ...)                               \
  do {                                                                     \
    __ptr__ = __func__(__VA_ARGS__);                                       \
    if (__ptr__ == nullptr) {                                              \
      fprintf(stderr, "Encountered IbVerbs nullptr error in func (%s) ",   \
               #__func__);                                                \
      exit(EXIT_FAILURE);                                               \
    }                                                                      \
  } while (0)
#endif

