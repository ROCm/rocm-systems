#include "pcie_counters_mi200.h"
#include "pcie_perfmon_registers_mi200.h"
#include "perfmon.h"

namespace rocprofiler {

PciePerfMonMI200::PciePerfMonMI200(const HSAAgentInfo& info) : PerfMon(), mmio_(nullptr) {
  mmio_ =
      dynamic_cast<mmio::PciePerfmonMMIO*>(mmio::MMIOManager::CreateMMIO(mmio::PCIE_PERFMON, info));
}

PciePerfMonMI200::~PciePerfMonMI200() {
  mmio::MMIOManager::DestroyMMIOInstance(dynamic_cast<mmio::MMIO*>(mmio_));
}

void PciePerfMonMI200::writeRegister(uint32_t reg_offset, uint32_t value) {
  // mmio or ioctl approaches
  mmio_->RegisterWriteAPI(reg_offset, value);
}

void PciePerfMonMI200::readRegister(uint32_t reg_offset, uint32_t& value) {
  // mmio or ioctl approaches
  mmio_->RegisterReadAPI(reg_offset, value);
}

void PciePerfMonMI200::SetCounterNames(std::vector<std::string>& counter_names) {
  counter_names_ = counter_names;
  // TODO: only one event at a time is supported at the moment
  auto it = PCIE_MI200::pcie_events_table.find(counter_names[0]);
  if (it != PCIE_MI200::pcie_events_table.end()) {
    PCIE_MI200::pcie_event_t event_desc = it->second;
    if (event_desc.event_category == PCIE_MI200::RX_TILE_SCLK) {
      event_id_ = event_desc.event_id;
    }
  }
}

void PciePerfMonMI200::Start() {
  // TODO: make sure values stored in table
  // in registers header are dec and not hex

  Start_RX_TILE_SCLK(event_id_);
}

void PciePerfMonMI200::Stop() {
  // TODO: revisit correct value to stop
  writeRegister(PCIE_MI200::PCIE_PERF_COUNT_CNTL, 0x2);  // stop
}

void PciePerfMonMI200::Read(std::vector<rocprofiler_counters_sampler_counter_output_t>& values) {
  uint64_t val = 0;
  Read_RX_TILE_SCLK(val);
  rocprofiler_counters_sampler_counter_output_t value = {ROCPROFILER_COUNTERS_SAMPLER_PCIE_COUNTERS,
                                                         static_cast<double>(val)};
  values.push_back(value);
}

void PciePerfMonMI200::Start_RX_TILE_TXCLK(uint32_t event) {
  // Step 1: PORT SEL update
  writeRegister(PCIE_MI200::PCIE_PERF_CNTL_EVENT_CI_PORT_SEL, 0x0);

  // Step 2: EVENT SEL update
  uint32_t value = event;  // last 8 bits for event
  writeRegister(PCIE_MI200::PCIE_PERF_CNTL_TXCLK3, value);

  // Steps 3 & 4: Performance counters initialization, enable:
  // TODO: revisit. Just a single write with 0x3 might be enough (check with pcie team)
  writeRegister(PCIE_MI200::PCIE_PERF_COUNT_CNTL, 0x5);
}

void PciePerfMonMI200::Read_RX_TILE_TXCLK(uint64_t& result) {
  // Step 5: Performance counters read:
  uint32_t lo_val, hi_val;
  readRegister(PCIE_MI200::PCIE_PERF_COUNT0_TXCLK3, lo_val);
  readRegister(PCIE_MI200::PCIE_PERF_COUNT0_UPVAL_TXCLK3, hi_val);

  // Combine the lo and hi values and put them in result
  uint64_t val = (hi_val & 0xFFFFUL);
  val = val << 32;
  result = val | lo_val;
}

void PciePerfMonMI200::Start_RX_TILE_SCLK(uint32_t event) {
  // Step 1: PORT SEL update
  writeRegister(PCIE_MI200::PCIE_PERF_CNTL_EVENT_CI_PORT_SEL, 0x0);

  // Step 2: EVENT SEL update
  uint32_t value = event;  // last 8 bits for event
  writeRegister(PCIE_MI200::PCIE_PERF_CNTL_LCLK1, value);

  // Steps 3 & 4: Performance counters initialization, enable:
  // TODO: revisit. Just a single write with 0x3 might be enough (check with pcie team)
  writeRegister(PCIE_MI200::PCIE_PERF_COUNT_CNTL, 0x5);
}

void PciePerfMonMI200::Read_RX_TILE_SCLK(uint64_t& result) {
  // Step 5: Performance counters read:
  uint32_t lo_val, hi_val;
  readRegister(PCIE_MI200::PCIE_PERF_COUNT0_LCLK1, lo_val);
  readRegister(PCIE_MI200::PCIE_PERF_COUNT0_UPVAL_LCLK1, hi_val);

  // Combine the lo and hi values and put them in result
  uint64_t val = (hi_val & 0xFFFFUL);
  val = val << 32;
  result = val | lo_val;
}

}  // namespace rocprofiler
