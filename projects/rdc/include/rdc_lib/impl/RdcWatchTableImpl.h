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
#ifndef RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_
#define RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_

#include <map>
#include <vector>
#include <utility>
#include <memory>
#include <mutex>
#include <atomic>
#include "rdc_lib/RdcWatchTable.h"
#include "rdc_lib/RdcGroupSettings.h"
#include "rdc_lib/RdcCacheManager.h"
#include "rdc_lib/RdcMetricFetcher.h"

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


class RdcWatchTableImpl : public RdcWatchTable {
 public:
    rdc_status_t rdc_job_start_stats(rdc_gpu_group_t group_id,
                    char job_id[64]) override;
    rdc_status_t rdc_watch_job_fields(rdc_gpu_group_t group_id,
                    uint64_t update_freq, double  max_keep_age,
                    uint32_t  max_keep_samples)  override;

    rdc_status_t rdc_watch_fields(rdc_gpu_group_t group_id,
        rdc_field_grp_t field_group_id, uint64_t update_freq,
        double  max_keep_age, uint32_t max_keep_samples) override;

    //!< rdc_unwatch_fields() will not remove the entry from watch_table.
    //!< The unwatched entry is still kept until the max_keep_age of the entry
    //!< is reached, which will be handled in the clean_up() function.
    rdc_status_t rdc_unwatch_fields(rdc_gpu_group_t group_id,
                rdc_field_grp_t field_group_id) override;

    //!< When the RDC is running as RDC_OPERATION_MODE_MANUAL, the user will
    //!< call this function periodically. Instead of providing other APIs to
    //!< cleanup the cache, this function will update and cleanup the cache.
    //!<
    //!< This function may be called very frequently, and the cache cleanup
    //!< is expensive. Internally, this function will throttle the cleanup to
    //!< once per second.
    rdc_status_t rdc_update_all_fields() override;

    RdcWatchTableImpl(const RdcGroupSettingsPtr& group_settings,
        const RdcCacheManagerPtr& cache_mgr,
        const RdcMetricFetcherPtr& metric_fetcher);

 private:
    //!< Helper function to Update the fields_in_table when unwatch tables
    rdc_status_t update_field_in_table_when_unwatch(const RdcFieldKey& entry);

    //!< Helper function to clean up the watch table and cache
    void clean_up();

    //!< Helper function to get the fields using the group and the field group.
    rdc_status_t get_fields_from_group(rdc_gpu_group_t group_id,
         rdc_field_grp_t field_group_id, std::vector<RdcFieldKey> & fields);


    RdcGroupSettingsPtr group_settings_;
    RdcCacheManagerPtr cache_mgr_;
    RdcMetricFetcherPtr metric_fetcher_;

    //!< The watch table to store the watch settings.
    std::map<RdcFieldKey, FieldSettings> watch_table_;

    //!< The settings for each field can be deduced from watch_table. But every
    //!< rdc_update_all_fields() call needs to deduce them. To improve the
    //!< performance, the fields_to_watch_ is used to track the field settings.
    //!< Those settings will only be updated when watching or unwatching.
    std::map<RdcFieldKey, FieldSettings> fields_to_watch_;

    //!< The last clean up time
    std::atomic<uint64_t> last_cleanup_time_;

    std::mutex watch_mutex_;
};

}  // namespace rdc
}  // namespace amd


#endif  // RDC_LIB_IMPL_RDCWATCHTABLEIMPL_H_
