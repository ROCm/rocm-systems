#include "lib/rocprofiler/counters/id_decode.hpp"

#include <string>
#include <unordered_map>

#include "lib/common/utility.hpp"

namespace rocprofiler
{
namespace counters
{
const std::unordered_map<rocprofiler_profile_counter_instance_types, std::string>&
dimension_map()
{
    static std::unordered_map<rocprofiler_profile_counter_instance_types, std::string> map = {
        {ROCPROFILER_DIMENSION_NONE, "DIMENSION_NONE"},
        {ROCPROFILER_DIMENSION_XCC, "DIMENSION_XCC"},
        {ROCPROFILER_DIMENSION_SHADER_ENGINE, "DIMENSION_SHADER_ENGINE"},
        {ROCPROFILER_DIMENSION_AGENT, "DIMENSION_AGENT"},
        {ROCPROFILER_DIMENSION_PMC_CHANNEL, "DIMENSION_PMC_CHANNEL"},
        {ROCPROFILER_DIMENSION_CU, "DIMENSION_CU"},
    };
    return map;
}

}  // namespace counters
}  // namespace rocprofiler