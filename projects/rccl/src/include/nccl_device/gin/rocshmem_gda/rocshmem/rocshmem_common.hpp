// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP
#define LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP

namespace rocshmem {

enum ROCSHMEM_STATUS {
  ROCSHMEM_SUCCESS = 0,
  ROCSHMEM_ERROR = 1,
};

enum class BackendType { GDA_BACKEND, };

}  // namespace rocshmem

#endif  // LIBRARY_INCLUDE_ROCSHMEM_COMMON_HPP
