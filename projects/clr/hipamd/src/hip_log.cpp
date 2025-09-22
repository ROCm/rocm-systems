#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_log.hpp"

namespace hip {

// Static instance pointer for LoggingInfo singleton
LoggingInfo* LoggingInfo::lginfo_;

void LoggingInfo::init() {
  amd::ScopedLock lock(lg_lock_);
  log_level_ = 0; log_size_ = 0; log_mask_ = 0;
}

hipError_t hipExtEnableLogging() {
  HIP_INIT_API(hipExtEnableLogging);
  amd::ScopedLock lock(LoggingInfo::instance().lg_lock_);
  AMD_LOG_LEVEL = LoggingInfo::instance().log_level_;
  AMD_LOG_MASK = LoggingInfo::instance().log_mask_;
  HIP_RETURN(hipSuccess);
}

hipError_t hipExtDisableLogging() {
  HIP_INIT_API(hipExtDisableLogging);
  amd::ScopedLock lock(LoggingInfo::instance().lg_lock_);
  AMD_LOG_LEVEL = 0;
  HIP_RETURN(hipSuccess);
}

hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask) {
  HIP_INIT_API(hipExtSetLoggingParams, log_level, log_size, log_mask);
  amd::ScopedLock lock(LoggingInfo::instance().lg_lock_);
  // Store logging parameters for later activation
  LoggingInfo::instance().log_level_ = log_level;
  LoggingInfo::instance().log_size_ = log_size;
  LoggingInfo::instance().log_mask_ = log_mask;
  HIP_RETURN(hipSuccess);
}
} // namespace::hip