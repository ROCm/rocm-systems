#include "df_counters_mi200.h"
#include "df_perfmon_registers_mi200.h"
#include "mmio.h"
#include "src/core/hsa/hsa_support.h"

namespace rocprofiler {

#define DF_V3_6_MAX_COUNTERS 4

/* get flags from df perfmon config */
#define DF_V3_6_GET_EVENT(x) (x & 0xFFUL)
#define DF_V3_6_GET_INSTANCE(x) ((x >> 8) & 0xFFUL)
#define DF_V3_6_GET_UNITMASK(x) ((x >> 16) & 0xFFUL)
#define DF_V3_6_PERFMON_OVERFLOW 0xFFFFFFFFFFFFULL

/* get ficaa value for accessing CakeDlwmActiveTransferCount */
#define AMDGPU_PMU_SET_FICAA(o) ((o << 16) | 0x1AF5)

DFPerfMonMI200::DFPerfMonMI200(const HSAAgentInfo& info) : PerfMon(), mmio_(nullptr) {
  mmio_ = dynamic_cast<mmio::DFPerfmonMMIO*>(mmio::MMIOManager::CreateMMIO(mmio::DF_PERFMON, info));
}

DFPerfMonMI200::~DFPerfMonMI200() {
  mmio::MMIOManager::DestroyMMIOInstance(dynamic_cast<mmio::MMIO*>(mmio_));
}

void DFPerfMonMI200::SetCounterNames(std::vector<std::string>& counter_names) {
  counter_names_ = counter_names;
  // TODO: only one event at a time is supported at the moment
  auto it = DF_MI200::xgmi_events_table.find(counter_names[0]);
  if (it != DF_MI200::xgmi_events_table.end()) {
    instance_id_ = it->second;
  }
}

void DFPerfMonMI200::writeRegister(uint32_t reg_offset, uint32_t value) {
  mmio_->RegisterWriteAPI(reg_offset, value);
}

void DFPerfMonMI200::readRegister(uint32_t reg_offset, uint32_t& value) {
  mmio_->RegisterReadAPI(reg_offset, value);
}

uint64_t DFPerfMonMI200::GetFicaNodeOutboundBw(uint32_t ficaa_val) {
  uint32_t ficadl_val, ficadh_val;

  // setting up FICAA for address
  writeRegister(DF_MI200::smnDF_PIE_AON_FabricIndirectConfigAccessAddress3, ficaa_val);

  // setting up FICADL for data
  readRegister(DF_MI200::smnDF_PIE_AON_FabricIndirectConfigAccessDataLo3, ficadl_val);

  // setting up FICADH for data
  readRegister(DF_MI200::smnDF_PIE_AON_FabricIndirectConfigAccessDataHi3, ficadh_val);

  return (((ficadh_val & 0xFFFFFFFFFFFFFFFF) << 32) | ficadl_val);
}

void DFPerfMonMI200::SetFicaNodeOutboundBw(uint64_t node_instance, uint32_t& ficaa_in_val) {
  uint32_t instance = node_instance;
  ficaa_in_val = AMDGPU_PMU_SET_FICAA(instance);
}


void DFPerfMonMI200::Start() {
  uint32_t ficaa_in_val;
  uint32_t ficaa_out_val;

  SetFicaNodeOutboundBw(instance_id_, ficaa_in_val);

  ficaa_out_val = GetFicaNodeOutboundBw(ficaa_in_val);
  printf("CakeDlwmActiveTransferCount_1=%u\n", ficaa_out_val);
  ficaa_out_val = ficaa_out_val & 0xFFFFFFFF;
  printf("CakeDlwmActiveTransferCount_2=%u\n", ficaa_out_val);
}

}  // namespace rocprofiler
