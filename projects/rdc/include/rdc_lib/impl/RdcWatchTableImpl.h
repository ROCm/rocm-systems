/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

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
#ifndef INCLUDE_RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_
#define INCLUDE_RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_

#include <string>
#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <mutex>  // NOLINT
#include <atomic>
#include "rdc_lib/RdcWatchTable.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcCacheManager.h"
#include "rdc_lib/RdcMetricFetcher.h"
#include "rdc_lib/RdcModuleMgr.h"
#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace rdc {

//!< The settings for a field or a group of field in the watch table.
struct FieldSettings {
    uint64_t  update_freq;
    uint32_t  max_keep_samples;
    double  max_keep_age;
    bool is_watching;
    uint64_t last_update_time;
};

struct JobWatchTableEntry {
    uint32_t group_id;
    std::vector<RdcFieldKey> fields;  //< store fields for faster query
};


class RdcWatchTableImpl : public RdcWatchTable {
 public:
    rdc_status_t rdc_job_start_stats(rdc_gpu_group_t group_id,
                    const char job_id[64], uint64_t update_freq,
                    const rdc_gpu_gauges_t& gpu_gauge) override;
    rdc_status_t rdc_job_stop_stats(const char job_id[64],
                    const rdc_gpu_gauges_t& gpu_gauge) override;
    rdc_status_t rdc_job_remove(const char job_id[64]) override;
    rdc_status_t rdc_job_remove_all() override;

    rdc_status_t rdc_field_watch(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double  max_keep_age, uint32_t max_keep_samples) override;

    //!< rdc_field_unwatch() will not remove the entry from watch_table.
    //!< The unwatched entry is still kept until the max_keep_age of the entry
    //!< is reached, which will be handled in the clean_up() function.
    rdc_status_t rdc_field_unwatch(rdc_gpu_group_t group_id,
                rdc_field_grp_t field_group_id) override;

    //!< When the RDC is running as RDC_OPERATION_MODE_MANUAL, the user will
    //!< call this function periodically. Instead of providing other APIs to
    //!< cleanup the cache, this function will update and cleanup the cache.
    //!<
    //!< This function may be called very frequently, and the cache cleanup
    //!< is expensive. Internally, this function will throttle the cleanup to
    //!< once per second.
    rdc_status_t rdc_field_update_all() override;

    RdcWatchTableImpl(const RdcGroupSettingsPtr& group_settings,
        const RdcCacheManagerPtr& cache_mgr,
        const RdcModuleMgrPtr& module_mgr);

 private:
    //!< Helper function to Update the fields_in_table when unwatch tables
    rdc_status_t update_field_in_table_when_unwatch(
                                               const RdcFieldGroupKey& entry);

    //!< Helper function to clean up the watch table and cache
    void clean_up();

    //!< Helper function for debug information in watch table and cache
    void debug_status();

    //!< Helper function to get the fields using the group and the field group.
    rdc_status_t get_fields_from_group(rdc_gpu_group_t group_id,
         rdc_field_grp_t field_group_id,
         std::vector<RdcFieldKey> & fields); // NOLINT

    bool is_job_watch_field(uint32_t gpu_index, rdc_field_t field_id,
                                        std::string& job_id) const;  // NOLINT

    rdc_status_t initialize_rsmi_handles(RdcFieldKey fk);

    //!< The function will be pass as the callback for bulk fetch
    static rdc_status_t handle_fields(rdc_gpu_field_value_t*  values,
            uint32_t num_values, void*  user_data);

    RdcGroupSettingsPtr group_settings_;
    RdcCacheManagerPtr cache_mgr_;
    RdcModuleMgrPtr rdc_module_mgr_;

    //!< The watch table to store the watch settings.
    std::map<RdcFieldGroupKey, FieldSettings> watch_table_;

    //!< <job_id, gpu_group_id> pairs
    std::map<std::string, JobWatchTableEntry> job_watch_table_;


    //!< The settings for each field can be deduced from watch_table. But every
    //!< rdc_field_update_all() call needs to deduce them. To improve the
    //!< performance, the fields_to_watch_ is used to track the field settings.
    //!< Those settings will only be updated when watching or unwatching.
    std::map<RdcFieldKey, FieldSettings> fields_to_watch_;

    //!< The last clean up time
    std::atomic<uint64_t> last_cleanup_time_;
    std::mutex watch_mutex_;
};

}  // namespace rdc
}  // namespace amd


#endif  // INCLUDE_RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_
