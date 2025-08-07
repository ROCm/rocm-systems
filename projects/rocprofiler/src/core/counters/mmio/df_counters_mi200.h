#ifndef DF_COUNTERS_MI200_H
#define DF_COUNTERS_MI200_H

#include <cstdint>
#include "mmio.h"
#include "perfmon.h"

namespace rocprofiler {

/*
  One perfmon per GPU.
  Only one instance per GPU, per process
*/

class DFPerfMonMI200 : public PerfMon {
 public:
  DFPerfMonMI200(const HSAAgentInfo& info);
  ~DFPerfMonMI200();
  void Start() override;
  void Stop(){};
  void Read(std::vector<rocprofiler_counters_sampler_counter_output_t>& values){};
  void SetCounterNames(std::vector<std::string>& counter_names);
  mmio::mmap_type_t Type() override { return mmio::mmap_type_t::DF_PERFMON; }

 private:
  void writeRegister(uint32_t reg_offset, uint32_t value);
  void readRegister(uint32_t reg_offset, uint32_t& value);

  // outboud bandwidth for xgmi nodes
  void SetFicaNodeOutboundBw(uint64_t node_instance, uint32_t& ficaa_in_val);
  uint64_t GetFicaNodeOutboundBw(uint32_t ficaa_val);


 private:
  mmio::DFPerfmonMMIO* mmio_;
  static std::mutex mutex_;  // should be an MMIO member
  static DFPerfMonMI200* instance_;
  uint64_t instance_id_;
};

}  // namespace rocprofiler

#endif  // DF_COUNTERS_MI200_H