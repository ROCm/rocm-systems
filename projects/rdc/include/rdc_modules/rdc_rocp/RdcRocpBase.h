/*
Copyright (c) 2022 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#ifndef RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
#define RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
#include <rocmtools.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <map>
#include <string>
#include <typeinfo>
#include <unordered_map>

#include "rdc/rdc.h"

namespace amd {
namespace rdc {

/**
 * @brief Map of RDC fields to rocmtools counters
 *
 * See metrics.xml in rocmtools for more info.
 * RDC_CALC fields are calculated over time by RDC.
 */
static const std::unordered_map<rdc_field_t, const char*> counter_map_k = {
    {RDC_FI_PROF_ELAPSED_CYCLES, "GRBM_COUNT"},
    {RDC_FI_PROF_ACTIVE_WAVES, "SQ_WAVES"},
    {RDC_FI_PROF_ACTIVE_CYCLES, "SQ_BUSY_CU_CYCLES"},
    {RDC_FI_PROF_CU_OCCUPANCY, "CU_OCCUPANCY"},
    {RDC_FI_PROF_CU_UTILIZATION, "CU_UTILIZATION"},
    {RDC_FI_PROF_FETCH_SIZE, "FETCH_SIZE"},
    {RDC_FI_PROF_WRITE_SIZE, "WRITE_SIZE"},
    {RDC_FI_PROF_FLOPS_16, "TOTAL_16_OPS"},
    {RDC_FI_PROF_FLOPS_32, "TOTAL_32_OPS"},
    {RDC_FI_PROF_FLOPS_64, "TOTAL_64_OPS"},
    // fields below require special handling
    {RDC_FI_PROF_GFLOPS_16, "TOTAL_16_OPS"},
    {RDC_FI_PROF_GFLOPS_32, "TOTAL_32_OPS"},
    {RDC_FI_PROF_GFLOPS_64, "TOTAL_64_OPS"},
    {RDC_FI_PROF_MEMR_BW_KBPNS, "FETCH_SIZE"},
    {RDC_FI_PROF_MEMW_BW_KBPNS, "WRITE_SIZE"},
};

/// Common interface for RocP tests and samples
class RdcRocpBase {
  typedef std::pair<uint32_t, rdc_field_t> pair_gpu_field_t;
  typedef struct session_info_t {
    rocmtools_session_id_t id{};
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> start_time;
    std::chrono::time_point<std::chrono::system_clock, std::chrono::nanoseconds> stop_time;
  } session_info_t;

 public:
  RdcRocpBase();
  RdcRocpBase(const RdcRocpBase&) = default;
  RdcRocpBase(RdcRocpBase&&) = delete;
  RdcRocpBase& operator=(const RdcRocpBase&) = delete;
  RdcRocpBase& operator=(RdcRocpBase&&) = delete;
  ~RdcRocpBase();

  /**
   * @brief Lookup ROCProfiler counter
   *
   * @param[in] field An existing field already added to sessions dictionary
   * @param[out] value A pointer that will be populated with returned value
   *
   * @retval ::ROCMTOOLS_STATUS_SUCCESS The function has been executed
   * successfully.
   */
  rdc_status_t rocp_lookup(pair_gpu_field_t gpu_field, double* value);

  /**
   * @brief Destroy ROCmTools session responsible for monitoring a given
   * field
   *
   * @details While rocmtools supports multiple fields per ID - it has a
   * limit to how many counters it can query internally.
   * To avoid concerning ourselves with said limit, we limit each session to
   * 1 field.
   * In the future this can be optimized to allow for multiple fields per
   * session.
   *
   * @param[in] field A field to start monitoring
   *
   * @retval ::ROCMTOOLS_STATUS_SUCCESS The function has been executed
   * successfully.
   */
  rdc_status_t create_session(pair_gpu_field_t gpu_field);

  /**
   * @brief Destroy ROCmTools session responsible for monitoring a given
   * field
   *
   * @param[in] field A field to stop monitoring
   *
   * @retval ::ROCMTOOLS_STATUS_SUCCESS The function has been executed
   * successfully.
   */
  rdc_status_t destroy_session(pair_gpu_field_t gpu_field);

 protected:
 private:
  std::map<pair_gpu_field_t, session_info_t> sessions;

  /**
   * @brief Convert from rocmtools status into RDC status
   */
  rdc_status_t Rocp2RdcError(rocmtools_status_t rocm_status);
};

}  // namespace rdc
}  // namespace amd

#endif  // RDC_MODULES_RDC_ROCP_RDCROCPBASE_H_
