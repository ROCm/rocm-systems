#ifndef DF_PERFMON_REGISTERS_MI200_H
#define DF_PERFMON_REGISTERS_MI200_H

#include <stdint.h>
#include <map>

namespace DF_MI200 {

const static uint32_t smnDF_PIE_AON_FabricIndirectConfigAccessAddress3 = 0x1d05cUL;
const static uint32_t smnDF_PIE_AON_FabricIndirectConfigAccessDataLo3 = 0x1d098UL;
const static uint32_t smnDF_PIE_AON_FabricIndirectConfigAccessDataHi3 = 0x1d09cUL;

#define NUM_EVENT_TYPES_ALDEBARAN 1
#define NUM_EVENTS_ALDEBARAN_XGMI 8
#define NUM_EVENTS_ALDEBARAN_MAX NUM_EVENTS_ALDEBARAN_XGMI

#define mmPCIE_INDEX2 0x000e
#define mmPCIE_INDEX2_BASE_IDX 0
#define mmPCIE_DATA2 0x000f
#define mmPCIE_DATA2_BASE_IDX 0

/* MI200 events */
const static std::map<std::string, uint64_t> xgmi_events_table = {
    {"xgmi_link0_data_outbound", 0x4b}, {"xgmi_link1_data_outbound", 0x4c},
    {"xgmi_link2_data_outbound", 0x4d}, {"xgmi_link3_data_outbound", 0x4e},
    {"xgmi_link4_data_outbound", 0x4f}, {"xgmi_link5_data_outbound", 0x50},
    {"xgmi_link6_data_outbound", 0x51}, {"xgmi_link7_data_outbound", 0x52}};

}  // namespace DF_MI200


#endif  // DF_PERFMON_REGISTERS_MI200_H