/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdlib>
#include <string>

namespace RcclUnitTesting
{
  class ScopedEnvVar
  {
  public:
    ScopedEnvVar(const char* name, const char* value) : name_(name)
    {
      if (const char* current = std::getenv(name))
      {
        hadPreviousValue_ = true;
        previousValue_ = current;
      }
      setenv(name, value, 1);
    }

    ~ScopedEnvVar()
    {
      if (hadPreviousValue_)
        setenv(name_.c_str(), previousValue_.c_str(), 1);
      else
        unsetenv(name_.c_str());
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
    std::string name_;
    std::string previousValue_;
    bool hadPreviousValue_ = false;
  };
}
