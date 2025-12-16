#include <hip/hip_runtime.h>
#include "hip_internal.hpp"
#include "hip_platform.hpp"

namespace hip {

hipError_t hipExtEnableLogging() {
  HIP_INIT_API(hipExtEnableLogging);
  amd::ScopedLock lock(PlatformState::instance().getLogLock());
  AMD_LOG_LEVEL = PlatformState::instance().log_level_;
  AMD_LOG_MASK = PlatformState::instance().log_mask_;
  HIP_RETURN(hipSuccess);
}

hipError_t hipExtDisableLogging() {
  HIP_INIT_API(hipExtDisableLogging);
  amd::ScopedLock lock(PlatformState::instance().getLogLock());
  AMD_LOG_LEVEL = 0;
  HIP_RETURN(hipSuccess);
}

hipError_t hipExtSetLoggingParams(size_t log_level, size_t log_size, size_t log_mask) {
  HIP_INIT_API(hipExtSetLoggingParams, log_level, log_size, log_mask);
  amd::ScopedLock lock(PlatformState::instance().getLogLock());
  // Store logging parameters for later activation
  PlatformState::instance().log_level_ = log_level;
  PlatformState::instance().log_size_ = log_size;
  PlatformState::instance().log_mask_ = log_mask;
  HIP_RETURN(hipSuccess);
}
} // namespace::hip