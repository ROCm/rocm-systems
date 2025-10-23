/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017-2025, Advanced Micro Devices, Inc.
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
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_

#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <map>
#include <mutex>  // NOLINT
#include <utility>

#include "rocm_smi/rocm_smi_io_link.h"
#include "rocm_smi/rocm_smi_kfd.h"
#include "rocm_smi/rocm_smi_device.h"
#include "rocm_smi/rocm_smi_monitor.h"
#include "rocm_smi/rocm_smi_power_mon.h"
#include "rocm_smi/rocm_smi_common.h"

namespace amd {
namespace smi {

class RocmSMI {
 public:
    explicit RocmSMI(uint64_t flags);
    ~RocmSMI(void);

    static RocmSMI& getInstance(uint64_t flags = 0);
    void Initialize(uint64_t flags);
    void Cleanup(void);

    std::vector<std::shared_ptr<amd::smi::Device>>&
                                                  devices() {return devices_;}

    uint32_t DiscoverAmdgpuDevices(void);
    int DiscoverAMDPowerMonitors(bool force_update = false);

    // Will execute "func" for every Device object known about, or until func
    // returns non-zero;
    uint32_t IterateSMIDevices(
      std::function<uint32_t(std::shared_ptr<Device>&, void *)> func, void *);

    void set_init_options(uint64_t options) {init_options_ = options;}
    uint64_t init_options() const {return init_options_;}
    uint64_t is_thread_only_mutex() const {
      return init_options_ & RSMI_INIT_FLAG_THRAD_ONLY_MUTEX;
    }

    uint32_t euid() const {return euid_;}

    std::map<uint64_t, std::shared_ptr<KFDNode>> & kfd_node_map(void) {
      return kfd_node_map_;}

    int kfd_notif_evt_fh(void) const {return kfd_notif_evt_fh_;}
    void set_kfd_notif_evt_fh(int fd) {kfd_notif_evt_fh_ = fd;}
    std::mutex *kfd_notif_evt_fh_mutex(void) {return &kfd_notif_evt_fh_mutex_;}
    std::mutex *bootstrap_mutex(void) {return &bootstrap_mutex_;}

    uint32_t ref_count(void) const {return ref_count_;}
    uint32_t ref_count_inc(void) {return ++ref_count_;}
    uint32_t ref_count_dec(void) {return --ref_count_;}

    uint32_t kfd_notif_evt_fh_refcnt(void) const {
                                             return kfd_notif_evt_fh_refcnt_;}
    uint32_t kfd_notif_evt_fh_refcnt_inc(void) {
                                           return ++kfd_notif_evt_fh_refcnt_;}
    uint32_t kfd_notif_evt_fh_refcnt_dec(void) {
                                           return --kfd_notif_evt_fh_refcnt_;}
    int get_io_link_weight(uint32_t node_from, uint32_t node_to,
                           uint64_t *weight);
    int get_node_index(uint32_t dv_ind, uint32_t *node_ind);
    const RocmSMI_env_vars& getEnv(void);
    std::string getRSMIEnvVarInfo(void);
    void debugRSMIEnvVarInfo();
    bool isLoggingOn(void);
    uint32_t getLogSetting(void);

 private:
    std::vector<std::shared_ptr<Device>> devices_;
    std::map<uint64_t, std::shared_ptr<KFDNode>> kfd_node_map_;
    std::vector<std::shared_ptr<Monitor>> monitors_;
    std::vector<std::shared_ptr<PowerMon>> power_mons_;
    std::set<std::string> amd_monitor_types_;
    std::map<std::pair<uint32_t, uint32_t>, std::shared_ptr<IOLink>>
      io_link_map_;
    std::map<uint32_t, uint32_t> dev_ind_to_node_ind_map_;
    void AddToDeviceList(std::string dev_name, uint64_t bdfid = 0);
    typedef struct {
      uint32_t card_index = std::numeric_limits<uint32_t>::max();
      std::string dev_name = "";
      std::string drm_render_path = "";
      std::string drm_card_path = "";
      uint32_t drm_render_minor = std::numeric_limits<uint32_t>::max();
      uint64_t bdfid = std::numeric_limits<uint64_t>::max();
    } rsmi_device_enumeration_t;
    rsmi_status_t AddToDeviceList2(rsmi_device_enumeration_t device);
    void GetEnvVariables(void);
    std::shared_ptr<Monitor> FindMonitor(std::string monitor_path);

    RocmSMI_env_vars env_vars_;
    uint64_t init_options_;
    uint32_t euid_;

    int kfd_notif_evt_fh_;
    std::mutex kfd_notif_evt_fh_mutex_;
    uint32_t kfd_notif_evt_fh_refcnt_;  // Access to this should be protected
                                        // by kfd_notif_evt_fh_mutex_
    std::mutex bootstrap_mutex_;
    uint32_t ref_count_;  // Access to this should be protected
                          // by bootstrap_mutex_
};

}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_MAIN_H_
