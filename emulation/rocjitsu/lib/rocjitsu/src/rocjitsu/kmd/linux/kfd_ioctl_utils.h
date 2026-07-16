// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_KMD_LINUX_KFD_IOCTL_UTILS_H_
#define ROCJITSU_KMD_LINUX_KFD_IOCTL_UTILS_H_

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "linux/uapi/kfd_ioctl.h"
RJ_DIAGNOSTIC_POP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace rocjitsu {

constexpr size_t ioctl_arg_size(unsigned long request) { return _IOC_SIZE(request); }

constexpr unsigned long ioctl_without_size(unsigned long request) {
  return request & ~(_IOC_SIZEMASK << _IOC_SIZESHIFT);
}

constexpr unsigned long ioctl_with_size(unsigned long request, size_t size) {
  return ioctl_without_size(request) |
         ((static_cast<unsigned long>(size) & _IOC_SIZEMASK) << _IOC_SIZESHIFT);
}

// SVM is the one KFD ioctl we see with a runtime-sized payload. The UAPI macro
// encodes sizeof(kfd_ioctl_svm_args), while libhsakmt rebuilds the request with
// enough _IOC_SIZE space for the trailing kfd_ioctl_svm_attribute array. Dispatch
// and name lookup therefore need to compare the command bits without the size.
constexpr bool is_svm_ioctl(unsigned long request) {
  return ioctl_without_size(request) == ioctl_without_size(AMDKFD_IOC_SVM);
}

constexpr unsigned long canonical_ioctl_request(unsigned long request) {
  return is_svm_ioctl(request) ? AMDKFD_IOC_SVM : request;
}

inline bool svm_ioctl_required_size(uint32_t nattr, size_t &required_size) {
  constexpr size_t base_size = sizeof(kfd_ioctl_svm_args);
  constexpr size_t attr_size = sizeof(kfd_ioctl_svm_attribute);
  constexpr auto max_nattr = (std::numeric_limits<size_t>::max() - base_size) / attr_size;
  if (static_cast<size_t>(nattr) > max_nattr)
    return false;
  required_size = base_size + static_cast<size_t>(nattr) * attr_size;
  return true;
}

inline bool validate_ioctl_arg_size(unsigned long request, const void *arg, size_t &arg_size) {
  arg_size = ioctl_arg_size(request);
  if (!is_svm_ioctl(request))
    return true;

  if (arg_size < sizeof(kfd_ioctl_svm_args))
    return false;
  const auto *svm_args = static_cast<const kfd_ioctl_svm_args *>(arg);
  size_t required_size = 0;
  return svm_ioctl_required_size(svm_args->nattr, required_size) && arg_size >= required_size;
}

} // namespace rocjitsu

#endif // ROCJITSU_KMD_LINUX_KFD_IOCTL_UTILS_H_
