/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <cstdlib>
#include <optional>
#include <string>

namespace RcclUnitTesting
{
  // Sets (or, when value is nullptr, unsets) an environment variable for the
  // lifetime of the object and restores the previous state on destruction.
  class ScopedEnvVar
  {
  public:
    ScopedEnvVar(const char* name, const char* value) : name_(name)
    {
      if (const char* current = std::getenv(name))
        previousValue_ = current;
      if (value != nullptr)
        setenv(name, value, 1);
      else
        unsetenv(name);
    }

    ~ScopedEnvVar()
    {
      if (previousValue_)
        setenv(name_.c_str(), previousValue_->c_str(), 1);
      else
        unsetenv(name_.c_str());
    }

    ScopedEnvVar(const ScopedEnvVar&) = delete;
    ScopedEnvVar& operator=(const ScopedEnvVar&) = delete;

  private:
    std::string name_;
    std::optional<std::string> previousValue_;
  };

  // Sends NCCL_DEBUG=INFO output for one debug subsystem to debugFile for the lifetime of the object.
  struct ScopedDebugLog
  {
    ScopedDebugLog(const char* subsystem, std::string const& debugFile)
      : subsys("NCCL_DEBUG_SUBSYS", subsystem), file("NCCL_DEBUG_FILE", debugFile.c_str())
    {
    }

    ScopedEnvVar level{"NCCL_DEBUG", "INFO"};
    ScopedEnvVar subsys;
    ScopedEnvVar file;
  };
}
