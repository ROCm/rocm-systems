/*
Copyright (c) 2024 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCCONFIGSETTINGSIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCCONFIGSETTINGSIMPL_H_

#include <atomic>
#include <condition_variable>
#include <mutex>  // NOLINT
#include <thread>
#include <unordered_map>

#include "rdc_lib/RdcConfigSettings.h"
#include "rdc_lib/RdcGroupSettings.h"

namespace amd {
namespace rdc {

class RdcConfigSettingsImpl : public RdcConfigSettings {
 public:
  // Set one configure
  rdc_status_t rdc_config_set(rdc_gpu_group_t group_id, rdc_config_setting_t setting) override;

  // Get the setting
  rdc_status_t rdc_config_get(rdc_gpu_group_t group_id,
                              rdc_config_setting_list_t* settings) override;

  // clear the setting
  rdc_status_t rdc_config_clear(rdc_gpu_group_t group_id) override;

  explicit RdcConfigSettingsImpl(const RdcGroupSettingsPtr& group_settings);

 private:
  RdcGroupSettingsPtr group_settings_;
  std::unordered_map<rdc_gpu_group_t, std::unordered_map<rdc_config_type_t, rdc_config_setting_t>>
      cached_group_settings_;
  std::thread monitor_thread_;
  std::mutex mutex_;              // Mutex for cached_group_settings_
  std::atomic<bool> is_running_;  // Bool for  if the thread should keep running
  std::condition_variable cv_;

  // monitorSettings() is kicked off from the RdcConfigSettingsImpl constructor as it's own thread
  // Every minute, it will check if gpu settings from amdsmi are the same as inside
  // cached_group_settings_ If not, it sets the mismatched values to the value in
  // cached_group_settings_
  void monitorSettings();
  uint64_t wattsToMicrowatts(uint64_t watts) const;
  uint64_t microwattsToWatts(int microwatts) const;
  rdc_status_t get_group_info(rdc_gpu_group_t group_id, rdc_group_info_t* rdc_group_info);
};

}  // namespace rdc
}  // namespace amd

#endif  // INCLUDE_RDC_LIB_IMPL_RDCCONFIGSETTINGSIMPL_H_
