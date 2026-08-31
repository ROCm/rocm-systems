// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_
#define INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_

#include <linux/perf_event.h>

#include <cstdint>
#include <string>
#include <unordered_set>
#include <vector>

#include "rocm_smi/rocm_smi.h"

namespace amd::smi::evt {

class RSMIEventGrpHashFunction {
 public:
  size_t operator()(const rsmi_event_group_t& grp) const { return static_cast<size_t>(grp); }
};

typedef std::unordered_set<rsmi_event_group_t, RSMIEventGrpHashFunction> dev_evt_grp_set_t;
void GetSupportedEventGroups(uint32_t dev_ind, dev_evt_grp_set_t* supported_grps);

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
  uint32_t getValue(rsmi_counter_value_t* val);
  uint32_t dev_file_ind(void) const { return dev_file_ind_; }
  uint32_t dev_ind(void) const { return dev_ind_; }

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
  int32_t get_event_type(uint32_t* ev_type);
};

}  // namespace amd::smi::evt

#endif  // INCLUDE_ROCM_SMI_ROCM_SMI_COUNTERS_H_
