// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
#define AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_

#include <map>
#include <mutex>
#include <set>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_drm.h"
#include "amd_smi/impl/amd_smi_gpu_backend.h"
#include "amd_smi/impl/amd_smi_processor.h"

// Forward declaration of UALoE handle type to keep ualoe_lib/ualoe_lib.h out
// of the public install tree. Implementation file pulls in ualoe_lib.h.
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
constexpr auto kUALOE_STATION_FLAGS = std::string_view("station_flags");
constexpr auto kUALOE_NUM_STATIONS = std::string_view("num_stations");
constexpr auto kUALOE_LANE_EN_BITMAP = std::string_view("lane_en_bitmap");

constexpr auto kUALOE_UALINK_SETUP_SUBDIR = std::string_view("setup/");
constexpr auto kUALOE_UALINK_CONFIG_SUBDIR = std::string_view("config/");
constexpr auto kUALOE_UALINK_STATIONS_SUBDIR = std::string_view("stations/");
constexpr auto kUALOE_UALINK_COMMIT_FILE = std::string_view("commit");

/**
 *  Flat surface: the legacy field files that live directly under the ualink root
 *  (no write-subtree prefix). Reads here use the same field filenames as the
 *  subtree readers, letting a caller diff flat vs setup/config/df to confirm a
 *  commit propagated into the flat surface.
 */
constexpr auto kUALOE_UALINK_FLAT_SUBDIR = std::string_view("");

enum class UALoeLinkInfo_t : std::uint16_t { LINK_TYPE = 0, ACCEL_STATE, ALL_LINK_INFO };

using UALoeLinkInfoMap_t = std::map<UALoeLinkInfo_t, std::string_view>;

/**
 *  Flat surface owned by get_fabric_info_from_ualoe()
 *      - fabric_type and accel_state only
 *  The Ppod/Vpod/Station payloads are sourced authoritatively from their
 *  respective subtrees via the 'query_fabric_*_config()' readers
 */
inline const auto UALoeLinkInfoMap = UALoeLinkInfoMap_t{
    {UALoeLinkInfo_t::LINK_TYPE, kUALOE_LINK_TYPE},
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
  AMDSmiGPUDevice(uint32_t gpu_id, std::string path, amdsmi_bdf_t bdf, AMDSmiDrm& drm);
  AMDSmiGPUDevice(uint32_t gpu_id, AMDSmiDrm& drm);
  ~AMDSmiGPUDevice();

  // Non-null only on non-Linux backends (e.g. WSL). amd_smi.cc checks this
  // before falling through to the Linux rsmi/libdrm path.
  IGPUBackend* backend() const { return backend_; }
  void set_backend(IGPUBackend* b) { backend_ = b; }
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
  // Set the bit for every CPU named by a sysfs cpulist ("0-3,8") that the
  // bitmask can hold. Malformed entries are skipped.
  static void parse_cpulist(const std::string& cpulist, std::vector<uint64_t>& bitmask);

  // Get the UALoE handle, opening the IFoE/UALoE session on first use.
  // Deferred out of the constructor so amdsmi_init() and non-fabric queries
  // never block on the IFoE driver; a wedged IFoE driver would otherwise hang
  // initialization in an uninterruptible generic-netlink wait.
  ualoe_handle_t get_ualoe_handle();

  /**
   *    UALoE fabric sysfs:
   *        - partial reads (see amdsmi_get_gpu_fabric_info() for status info)
   */
  auto get_fabric_info_from_ualoe(
      amdsmi_fabric_info_t& fabric_info,
      UALoeLinkInfo_t link_info_type = UALoeLinkInfo_t::ALL_LINK_INFO) const -> amdsmi_status_t;

  auto apply_fabric_ppod_config(const amdsmi_fabric_ppod_config_t& config) const -> amdsmi_status_t;
  auto apply_fabric_vpod_config(const amdsmi_fabric_vpod_config_t& config) const -> amdsmi_status_t;
  auto apply_fabric_station_config(const amdsmi_fabric_station_config_t& config) const
      -> amdsmi_status_t;

  /**
   *    UALoE fabric write-subtree readback (live, post-commit state):
   *        - config.mask selects fields to read. On return it reports fields actually populated
   *        - absent/empty masked fields keep a sentinel value and clear their readback bit
   */
  auto query_fabric_ppod_config(amdsmi_fabric_ppod_config_t& config) const -> amdsmi_status_t;
  auto query_fabric_vpod_config(amdsmi_fabric_vpod_config_t& config) const -> amdsmi_status_t;
  auto query_fabric_station_config(amdsmi_fabric_station_config_t& config) const -> amdsmi_status_t;

  /**
   *    UALoE fabric flat-surface readback (legacy files directly under the ualink
   *    root). Same fields/semantics as the subtree readers above, but sourced from
   *    the flat surface so callers can diff the two and confirm commit propagation.
   */
  auto query_fabric_ppod_config_flat(amdsmi_fabric_ppod_config_t& config) const -> amdsmi_status_t;
  auto query_fabric_vpod_config_flat(amdsmi_fabric_vpod_config_t& config) const -> amdsmi_status_t;
  auto query_fabric_station_config_flat(amdsmi_fabric_station_config_t& config) const
      -> amdsmi_status_t;

  auto has_ifoe_related_bdf() const -> bool;
  auto get_ifoe_bdf_string() const -> std::string;
  bool device_has_ualink() const;

 private:
  IGPUBackend* backend_ = nullptr;
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

  // UALoE — session is opened lazily on the first get_ualoe_handle() call
  void open_ualoe_session();
  ualoe_handle_t ualoe_handle_ = (-1);
  std::once_flag ualoe_open_once_;

  /**
   *  UALoE sysfs ualink root: /sys/class/drm/<gpu_path>/device/ualink
   */
  auto get_ualink_directory_path() const -> std::string;
};

}  // namespace amd::smi

#endif  // AMD_SMI_INCLUDE_IMPL_AMD_SMI_GPU_DEVICE_H_
