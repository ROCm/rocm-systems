/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Framework-independent helpers for the rocshmem4py binding.  This header
 * MUST NOT include or depend on any binding framework: it only wraps rocSHMEM
 * ABI behavior, keeping the nanobind registration layer (rocshmem4py.cc) thin.
 */

#ifndef ROCSHMEM4PY_COMMON_HPP
#define ROCSHMEM4PY_COMMON_HPP

#include <rocshmem/rocshmem.hpp>
#include <cstdint>
#include <sstream>
#include <stdexcept>

namespace rocshmem4py {

// Translate the Python team sentinels to rocSHMEM ABI handles:
//   0   -> rocshmem::ROCSHMEM_TEAM_WORLD (runtime pointer set at init)
//  -1   -> rocshmem::ROCSHMEM_TEAM_INVALID    (rocSHMEM ABI nullptr)
// Anything else is a raw rocshmem_team_t (intptr_t-cast pointer) returned by
// a previous successful split and passed back through.
inline rocshmem::rocshmem_team_t resolve_team_handle(intptr_t team) {
  if (team == 0) return rocshmem::ROCSHMEM_TEAM_WORLD;
  if (team == -1) return rocshmem::ROCSHMEM_TEAM_INVALID;
  return reinterpret_cast<rocshmem::rocshmem_team_t>(team);
}

}  // namespace rocshmem4py

// Throw std::runtime_error on a non-success rocSHMEM status.  nanobind
// auto-translates std::runtime_error to a Python RuntimeError.
#define CHECK_ROCSHMEM(expr)                                            \
  do {                                                                  \
    int status = expr;                                                  \
    if (status != ROCSHMEM_SUCCESS) {                                   \
      std::ostringstream err_msg;                                       \
      err_msg << "ROCSHMEM error in " << __FILE__ << ":" << __LINE__;   \
      throw std::runtime_error(err_msg.str());                         \
    }                                                                   \
  } while (0)

#endif  // ROCSHMEM4PY_COMMON_HPP
