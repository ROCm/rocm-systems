/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef SYMMETRIC_KERNEL_LOG_HPP
#define SYMMETRIC_KERNEL_LOG_HPP

#include <cctype>
#include <string>

namespace RcclUnitTesting
{
  // Whitespace-delimited token that follows key in line, or "" when key is absent.
  inline std::string TokenAfter(const std::string& line, const std::string& key)
  {
    size_t p = line.find(key);
    if (p == std::string::npos)
    {
      return "";
    }
    p += key.size();
    while (p < line.size() && isspace(static_cast<unsigned char>(line[p])))
    {
      p++;
    }
    size_t e = p;
    while (e < line.size() && !isspace(static_cast<unsigned char>(line[e])))
    {
      e++;
    }
    return line.substr(p, e - p);
  }

  // Kernel named by a "<func> [Symmetric]: ... Kernel <name>" tuning line, or "" for any other line.
  inline std::string SymmetricKernelOnLine(const std::string& line, const std::string& func)
  {
    const size_t tag = line.find(func + " [Symmetric]:");
    return tag == std::string::npos ? "" : TokenAfter(line.substr(tag), "Kernel ");
  }

  // Tensor Data Mover kernels (AllGather_TmaST, ...) are the ones reported as SymTDM.
  inline bool IsTdmSymmetricKernel(const std::string& kernel)
  {
    return kernel.find("Tma") != std::string::npos;
  }

  // Algorithm names rcclGetAlgoName and rcclSymKGetInfo report for a symmetric kernel.
  inline bool IsSymmetricAlgoName(const std::string& algo)
  {
    return algo == "SYM" || algo == "SymTDM";
  }
}  // namespace RcclUnitTesting

#endif  // SYMMETRIC_KERNEL_LOG_HPP
