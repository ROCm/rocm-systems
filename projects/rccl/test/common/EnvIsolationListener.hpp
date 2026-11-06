/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#pragma once

#include <gtest/gtest.h>

#include <cstdlib>
#include <map>
#include <string>

extern char** environ;

namespace RcclUnitTesting
{
  // Restores the process environment after every test and fails any test that changed it.
  // Workers and isolated-test children copy this process's environment when they start, so a
  // variable left behind by one test would silently reconfigure every test that runs after it.
  // Tests that need a variable should hold a ScopedEnvVar for the duration of the test.
  class EnvIsolationListener : public ::testing::EmptyTestEventListener
  {
  public:
    // Call once, after InitGoogleTest.
    static void Install()
    {
      ::testing::UnitTest::GetInstance()->listeners().Append(new EnvIsolationListener);
    }

    void OnTestStart(::testing::TestInfo const&) override
    {
      snapshot_ = Capture();
    }

    void OnTestEnd(::testing::TestInfo const&) override
    {
      std::string const leaked = Restore(snapshot_);
      if (!leaked.empty())
        ADD_FAILURE() << "Test left environment variables changed (now restored; use ScopedEnvVar): " << leaked;
    }

  private:
    using Environment = std::map<std::string, std::string>;

    static Environment Capture()
    {
      Environment env;
      for (char** entry = environ; *entry != nullptr; ++entry)
      {
        std::string const assignment(*entry);
        size_t const eq = assignment.find('=');
        if (eq != std::string::npos)
          env.emplace(assignment.substr(0, eq), assignment.substr(eq + 1));
      }
      return env;
    }

    // Returns the changed variables as "NAME (set|changed|unset)", comma-separated.
    static std::string Restore(Environment const& saved)
    {
      Environment const current = Capture();
      std::string leaked;
      auto report = [&leaked](std::string const& name, char const* how) {
        leaked += (leaked.empty() ? "" : ", ") + name + " (" + how + ")";
      };

      for (auto const& [name, value] : current)
      {
        auto const it = saved.find(name);
        if (it == saved.end())
        {
          unsetenv(name.c_str());
          report(name, "set");
        }
        else if (it->second != value)
        {
          setenv(name.c_str(), it->second.c_str(), 1);
          report(name, "changed");
        }
      }
      for (auto const& [name, value] : saved)
      {
        if (current.count(name) == 0)
        {
          setenv(name.c_str(), value.c_str(), 1);
          report(name, "unset");
        }
      }
      return leaked;
    }

    Environment snapshot_;
  };
}
