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

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_processor.h"

extern "C" {
#include "ualoe_lib/ualoe_lib.h"
}

#include <map>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

// decouple the dependency to ualoe_lib.h by using a typedef for the handle
typedef int ualoe_handle_t;

namespace amd::smi {

/*
 * UALoE Link Information
 * -----------------------------------------------------------------------------------------------
 *  - link_type: ualink or ualoe
 *  - accel_id: size is architecture dependent, 0-255 on MI4xx
 *  - bandwidth: station bandwidth share, units TBD
 *  - latency: latency depends on switch presence/type, unit TBD
 *  - ppod_id: physical pod id: 64-bit hexadecimal 0x...
 *  - ppod_size: physical pod size
 *  - vpod_id: virtual pod id, size architecture dependent, decimal
 *  - vpod_size: virtual pod size
 *  - vpod_active_accels: list of active accelerator ids
 *  - local_accels: list of local accelerator ids
 *  - addr_mode: source aliasing or source identification
 *  - accel_state: this accelerator state: unconfigured, configured, ready, active, error
 */

constexpr auto kIFOE_DRIVER_BASE_PATH = std::string_view("/sys/bus/pci/drivers/ifoe/");
constexpr auto kUALOE_BASE_PATH = std::string_view("/sys/class/drm/");
constexpr auto kUALOE_UALINK_DIRECTORY = std::string_view("/device/ualink");
constexpr auto kUALOE_LINK_TYPE = std::string_view("link_type");
constexpr auto kUALOE_ACCEL_ID = std::string_view("accel_id");
constexpr auto kUALOE_BANDWIDTH = std::string_view("bandwidth");
constexpr auto kUALOE_LATENCY = std::string_view("latency");
constexpr auto kUALOE_PPOD_ID = std::string_view("ppod_id");
constexpr auto kUALOE_PPOD_SIZE = std::string_view("ppod_size");
constexpr auto kUALOE_VPOD_ID = std::string_view("vpod_id");
constexpr auto kUALOE_VPOD_SIZE = std::string_view("vpod_size");
constexpr auto kUALOE_VPOD_ACTIVE_ACCELS = std::string_view("vpod_active_accels");
constexpr auto kUALOE_LOCAL_ACCELS = std::string_view("local_accels");
constexpr auto kUALOE_ADDR_MODE = std::string_view("addr_mode");
constexpr auto kUALOE_ACCEL_STATE = std::string_view("accel_state");
constexpr auto kUALOE_BDF_OFFSET = std::uint16_t(0x01);  // TODO: Example offset - TBD

enum class UALoeLinkInfo_t : std::uint16_t {
  LINK_TYPE = 0,
  ACCEL_ID,
  BANDWIDTH,
  LATENCY,
  PPOD_ID,
  PPOD_SIZE,
  VPOD_ID,
  VPOD_SIZE,
  VPOD_ACTIVE_ACCELS,
  LOCAL_ACCELS,
  ADDR_MODE,
  ACCEL_STATE,
  ALL_LINK_INFO
};

using UALoeLinkInfoMap_t = std::map<UALoeLinkInfo_t, std::string_view>;
inline const auto UALoeLinkInfoMap = UALoeLinkInfoMap_t{
    {UALoeLinkInfo_t::LINK_TYPE, kUALOE_LINK_TYPE},
    {UALoeLinkInfo_t::ACCEL_ID, kUALOE_ACCEL_ID},
    {UALoeLinkInfo_t::BANDWIDTH, kUALOE_BANDWIDTH},
    {UALoeLinkInfo_t::LATENCY, kUALOE_LATENCY},
    {UALoeLinkInfo_t::PPOD_ID, kUALOE_PPOD_ID},
    {UALoeLinkInfo_t::PPOD_SIZE, kUALOE_PPOD_SIZE},
    {UALoeLinkInfo_t::VPOD_ID, kUALOE_VPOD_ID},
    {UALoeLinkInfo_t::VPOD_SIZE, kUALOE_VPOD_SIZE},
    {UALoeLinkInfo_t::VPOD_ACTIVE_ACCELS, kUALOE_VPOD_ACTIVE_ACCELS},
    {UALoeLinkInfo_t::LOCAL_ACCELS, kUALOE_LOCAL_ACCELS},
    {UALoeLinkInfo_t::ADDR_MODE, kUALOE_ADDR_MODE},
    {UALoeLinkInfo_t::ACCEL_STATE, kUALOE_ACCEL_STATE},
};
using UALoeLinkInfoType_t = std::underlying_type_t<UALoeLinkInfo_t>;
using UALoeLinkInfoLine_t = std::string;
using UALoeLinkInfoLines_t = std::vector<UALoeLinkInfoLine_t>;

/*
 * Ordering for fabric BDF sets (amdsmi_bdf_t has no operator<).
 */
struct FabricBdfLessOp {
  bool operator()(const amdsmi_bdf_t& a, const amdsmi_bdf_t& b) const noexcept {
    return (a.as_uint < b.as_uint);
  }
};
using FabricBDFList_t = std::set<amdsmi_bdf_t, FabricBdfLessOp>;

// PID, amdsmi_proc_info_t
using GPUComputeProcessList_t = std::map<amdsmi_process_handle_t, amdsmi_proc_info_t>;
using ComputeProcessListClassType_t = uint16_t;

enum class ComputeProcessListType_t : ComputeProcessListClassType_t {
  kAllProcesses,
  kAllProcessesOnDevice,
};

class AMDSmiGPUDevice : public AMDSmiProcessor {
 public:
  // UALoE requires a matching IFoE BDF under /sys/bus/pci/drivers/ifoe/;
  // otherwise ualoe_open is skipped and fabric queries return NOT_SUPPORTED.
  AMDSmiGPUDevice(uint32_t gpu_id, std::string path, amdsmi_bdf_t bdf, AMDSmiDrm& drm)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU),
        gpu_id_(gpu_id),
        path_(path),
        bdf_(bdf),
        drm_(drm) {
    populate_ifoe_fabric_bdf_list();
    if (has_ifoe_related_bdf() && device_has_ualink()) {
      auto ifoe_bdf_str = get_ifoe_bdf_string();
      if (auto ualoe_status = ualoe_open(ifoe_bdf_str.c_str(), &ualoe_handle_); ualoe_status != 0) {
        ualoe_handle_ = (-1);
      }
    }
  }

  AMDSmiGPUDevice(uint32_t gpu_id, AMDSmiDrm& drm)
      : AMDSmiProcessor(AMDSMI_PROCESSOR_TYPE_AMD_GPU), gpu_id_(gpu_id), drm_(drm) {
    if (check_if_drm_is_supported()) this->get_drm_data();

    populate_ifoe_fabric_bdf_list();
    if (has_ifoe_related_bdf() && device_has_ualink()) {
      auto ifoe_bdf_str = get_ifoe_bdf_string();
      if (auto ualoe_status = ualoe_open(ifoe_bdf_str.c_str(), &ualoe_handle_); ualoe_status != 0) {
        ualoe_handle_ = (-1);
      }
    }
  }

  ~AMDSmiGPUDevice() {
    if (ualoe_handle_ != -1) {
      ualoe_close(ualoe_handle_);
      ualoe_handle_ = -1;
    }
  }

  amdsmi_status_t get_drm_data();
  pthread_mutex_t* get_mutex();
  uint32_t get_gpu_id() const;
  uint32_t get_card_id();           // -e feature + we can get card_id for our internal functions
  uint32_t get_drm_render_minor();  // -e feature + we can get card_id for our internal functions
  uint64_t get_kfd_gpu_id();        // Used to decode vram usage for KFD processes
  std::string& get_gpu_path();
  const std::string& get_gpu_path() const;
  amdsmi_bdf_t get_bdf();
  bool check_if_drm_is_supported() { return drm_.check_if_drm_is_supported(); }
  uint32_t get_vendor_id();
  const GPUComputeProcessList_t& amdgpu_get_compute_process_list(
      ComputeProcessListType_t list_type = ComputeProcessListType_t::kAllProcessesOnDevice);
  amdsmi_status_t amdgpu_query_cpu_affinity(std::string& cpu_affinity) const;

  // New methods for -e feature
  std::string bdf_to_string() const;  // -e feature
  std::vector<uint64_t> get_bitmask_from_numa_node(int32_t node_id, uint32_t size) const;
  std::vector<uint64_t> get_bitmask_from_local_cpulist(uint32_t drm_card, uint32_t size) const;

  // Get the UALoE handle
  ualoe_handle_t get_ualoe_handle() const { return ualoe_handle_; }

  /** UALoE fabric sysfs:
   *    - partial reads; see amdsmi_get_gpu_fabric_info() for status info
   */
  auto get_fabric_info_from_ualoe(
      amdsmi_fabric_info_t& fabric_info,
      UALoeLinkInfo_t link_info_type = UALoeLinkInfo_t::ALL_LINK_INFO) const -> amdsmi_status_t;

  auto has_ifoe_related_bdf() const -> bool;
  auto get_ifoe_bdf_string() const -> std::string;
  bool device_has_ualink() const;

 private:
  uint32_t gpu_id_;
  std::string path_;
  amdsmi_bdf_t bdf_;
  FabricBDFList_t fabric_bdf_list_;
  uint32_t vendor_id_;
  AMDSmiDrm& drm_;
  uint32_t card_index_;
  uint32_t drm_render_minor_;
  uint64_t kfd_gpu_id_;  // Used to decode vram usage for KFD processes
  GPUComputeProcessList_t compute_process_list_;
  std::string gpu_uuid_;  // Device UUID for UALoE identification
  int32_t get_compute_process_list_impl(GPUComputeProcessList_t& compute_process_list,
                                        ComputeProcessListType_t list_type);
  void populate_ifoe_fabric_bdf_list();
  // UALoE
  ualoe_handle_t ualoe_handle_ = (-1);
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
