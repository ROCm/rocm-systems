#ifndef PCIE_COUNTERS_MI200_H
#define PCIE_COUNTERS_MI200_H

#include "mmio.h"
#include "perfmon.h"

namespace rocprofiler {

/*
  One perfmon per GPU.
  Only one instance per GPU, per process
*/

class PciePerfMonMI200 : public PerfMon {
 public:
  PciePerfMonMI200(const HSAAgentInfo& info);
  ~PciePerfMonMI200();
  void SetCounterNames(std::vector<std::string>& counter_names) override;
  void Start() override;
  void Stop() override;
  void Read(std::vector<rocprofiler_counters_sampler_counter_output_t>& values) override;
  mmio::mmap_type_t Type() override { return mmio::mmap_type_t::PCIE_PERFMON; }

 private:
  // TODO : check google coding std
  void writeRegister(uint32_t reg_offset, uint32_t value);
  void readRegister(uint32_t reg_offset, uint32_t& value);

  void Start_RX_TILE_TXCLK(uint32_t event);
  void Read_RX_TILE_TXCLK(uint64_t& result);

  void Start_RX_TILE_SCLK(uint32_t event);
  void Read_RX_TILE_SCLK(uint64_t& result);

 private:
  mmio::PciePerfmonMMIO* mmio_;
  std::vector<std::string> counter_names_;
  int event_id_;
};

}  // namespace rocprofiler

#endif