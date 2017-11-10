#include "core/metrics.h"

namespace rocprofiler {
MetricsDict::map_t* MetricsDict::map_ = NULL;
MetricsDict::mutex_t MetricsDict::mutex_;
}
