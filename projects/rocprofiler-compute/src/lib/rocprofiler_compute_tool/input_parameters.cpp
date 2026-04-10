#include "input_parameters.h"
#include <stdlib.h>

using namespace rocprof_compute_tool;

char* EnvInputParameters::get_output_path()
{
    return getenv("ROCPROF_OUTPUT_PATH");
}

char* EnvInputParameters::get_requested_counters()
{
    return getenv("ROCPROF_COUNTERS");
}

char* EnvInputParameters::get_iteration_multiplexing_mode()
{
    return getenv("ROCPROF_ITERATION_MULTIPLEXING");
}

char* EnvInputParameters::get_kernel_filter_include_regex()
{
    return getenv("ROCPROF_KERNEL_FILTER_INCLUDE_REGEX");
}

char* EnvInputParameters::get_kernel_filter_range()
{
    return getenv("ROCPROF_KERNEL_FILTER_RANGE");
}
