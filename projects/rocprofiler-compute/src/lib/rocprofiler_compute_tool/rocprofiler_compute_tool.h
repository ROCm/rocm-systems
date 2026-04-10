#pragma once
#include "input_parameters.h"
#include "sdk_wrapper.h"

#include <memory>

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id);

namespace rocprof_compute_tool::test_knobs
{
void set_input_parameters(std::shared_ptr<InputParameters> parameters);
}