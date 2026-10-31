/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef RCCL_TESTS_MPICH_CONFIG_H_
#define RCCL_TESTS_MPICH_CONFIG_H_

#include <cctype>
#include <string>

// Pure host helper for MPICH's MPI_Get_library_version() configure line.
// Tested from test/mpich_config_test.cpp; no GPU or MPI required.

namespace rccl_tests_mpi {

// True when `version` contains `option` as a configure flag that enables the
// feature. A delimiter has to follow the option so that "--with-ze" does not
// match "--with-zelda", and "--with-hip" is not found inside "--without-hip".
// Autoconf spells "--without-hip" as "--with-hip=no", which is not a match.
inline bool mpichConfiguredWith(const std::string& version, const std::string& option) {
  for (size_t pos = version.find(option); pos != std::string::npos;
       pos = version.find(option, pos + 1)) {
    if (pos > 0 && std::isspace(static_cast<unsigned char>(version[pos - 1])) == 0)
      continue;
    const size_t end = pos + option.size();
    if (end == version.size() ||
        std::isspace(static_cast<unsigned char>(version[end])) != 0)
      return true;
    if (version[end] != '=')
      continue;
    size_t valEnd = end + 1;
    while (valEnd < version.size() &&
           std::isspace(static_cast<unsigned char>(version[valEnd])) == 0)
      ++valEnd;
    if (version.compare(end + 1, valEnd - (end + 1), "no") == 0)
      continue;
    return true;
  }
  return false;
}

}  // namespace rccl_tests_mpi

#endif
