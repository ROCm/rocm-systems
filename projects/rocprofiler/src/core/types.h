#ifndef SRC_CORE_TYPES_H_
#define SRC_CORE_TYPES_H_

#include <hsa_ven_amd_aqlprofile.h>

namespace rocprofiler {
typedef hsa_ven_amd_aqlprofile_1_00_pfn_t pfn_t;
typedef hsa_ven_amd_aqlprofile_event_t event_t;
typedef hsa_ven_amd_aqlprofile_parameter_t parameter_t;
typedef hsa_ven_amd_aqlprofile_profile_t profile_t;
typedef hsa_ext_amd_aql_pm4_packet_t packet_t;
typedef uint32_t packet_word_t;
}  // namespace rocprofiler

#endif  // SRC_CORE_TYPES_H_
