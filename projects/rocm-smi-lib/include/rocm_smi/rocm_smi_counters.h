/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2025, Advanced Micro Devices, Inc.
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
#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_

#include <linux/perf_event.h>

#include <cstdint>
#include <vector>
#include <unordered_set>
#include <string>

#include "rocm_smi/rocm_smi.h"

namespace amd {
namespace smi {
namespace evt {

class RSMIEventGrpHashFunction {
 public:
    size_t operator()(const rsmi_event_group_t& grp) const {
      return static_cast<size_t>(grp);
    }
};

typedef std::unordered_set<rsmi_event_group_t, RSMIEventGrpHashFunction>
                                                            dev_evt_grp_set_t;
void
GetSupportedEventGroups(uint32_t dev_ind, dev_evt_grp_set_t*supported_grps);

struct evnt_info_t {
    uint8_t start_bit;
    uint8_t field_size;
    uint64_t value;
};

struct perf_read_format_t {
    union {
        struct {
            uint64_t value;
            uint64_t enabled_time;
            uint64_t run_time;
        };
        uint64_t values[3];
    };
};

class Event {
 public:
    explicit Event(rsmi_event_type_t event, uint32_t dev_ind);
    ~Event(void);

    int32_t openPerfHandle();
    int32_t startCounter(void);
    int32_t stopCounter(void);
    uint32_t getValue(rsmi_counter_value_t *val);
    uint32_t dev_file_ind(void) const {return dev_file_ind_;}
    uint32_t dev_ind(void) const {return dev_ind_;}

 private:
    // perf_event_attr fields
    std::vector<evnt_info_t> event_info_;

    std::string evt_path_root_;

    rsmi_event_type_t event_type_;
    uint32_t dev_file_ind_;
    uint32_t dev_ind_;
    int32_t fd_;
    perf_event_attr attr_;
    uint64_t prev_cntr_val_;
    int32_t get_event_file_info(void);
    int32_t get_event_type(uint32_t *ev_type);
};


}  // namespace evt
}  // namespace smi
}  // namespace amd

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_

