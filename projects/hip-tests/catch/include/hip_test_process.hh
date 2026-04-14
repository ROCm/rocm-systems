/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hip_test_common.hh"
#include "hip_test_filesystem.hh"

#include <string>
#include <array>
#include <cstdlib>
#include <random>
#include <fstream>
#include <streambuf>
#include <thread>
#include <future>

namespace hip {
/*
Class to spawn a process in isolation and test its standard output and return status
Good for printf tests and environment variable tests

How to use:
Have the stand alone exe in the same folder
Init a class using hip::SpawnProc proc("ExeName", yes_or_no_to_capture_output);
proc.run("Optional command line args");
*/
class SpawnProc {
  std::string exeName;
  std::string resultStr;
  std::string tmpFileName;
  std::future<int> ret_from_run;
  bool captureOutput;

  std::string getRandomString(size_t len = 6) {
    std::random_device dev;
    std::mt19937 rng(dev());
    std::uniform_int_distribution<std::mt19937::result_type> dist(0, 25);

    std::string res;
    for (size_t i = 0; i < len; i++) {
      res += 'a' + dist(rng);
    }
    return res;
  }

 public:
  SpawnProc(std::string exeName_, bool captureOutput_ = false)
      : exeName(exeName_), captureOutput(captureOutput_) {
    auto dir = fs::path(TestContext::get().currentPath());
    dir /= exeName;
    exeName = dir.string();
    // On Windows, fs::exists returns false without extension.
    if (TestContext::get().isWindows()) {
      if (fs::path(exeName).extension().empty()) {
        exeName += ".exe";
      }
    }
    INFO("Testing that exe exists: " << exeName);
    REQUIRE(fs::exists(exeName));

    if (captureOutput) {
      auto path = fs::temp_directory_path();
      path /= getRandomString();
      tmpFileName = path.string();
      INFO("Testing that capture file does not exist already: " << tmpFileName);
      REQUIRE(!fs::exists(tmpFileName));
    }
    if (TestContext::get().isWindows()) {
      exeName = (exeName.find(" ", 0) == std::string::npos) ? exeName : ("\"" + exeName + "\"");
      tmpFileName = (tmpFileName.find(" ", 0) == std::string::npos) ? tmpFileName : ("\"" + tmpFileName + "\"");
    }
  }

  int run(std::string commandLineArgs = "") {
    std::string execCmd = exeName;

    // Append command line args
    if (commandLineArgs.size() > 0) {
      execCmd += " ";  // Add space for command line args
      execCmd += commandLineArgs;
    }

    if (captureOutput) {
      execCmd += " > ";
      execCmd += tmpFileName;
    }
    if (TestContext::get().isWindows()) {
      execCmd = (execCmd.find(" ", 0) == std::string::npos) ? execCmd : ("\"" + execCmd + "\"");
    }
    auto res = std::system(execCmd.c_str());

    if (captureOutput) {
      std::ifstream t(tmpFileName.c_str());
      resultStr =
          std::string((std::istreambuf_iterator<char>(t)), std::istreambuf_iterator<char>());
      t.close();
    }
#if HT_LINUX
    return WEXITSTATUS(res);
#else
    return res;
#endif
  }

  void run_async(std::string commandLineArgs = "") {
    ret_from_run = std::async(std::launch::async, &hip::SpawnProc::run, this, commandLineArgs);
  }

  int wait() {
    ret_from_run.wait();
    return ret_from_run.get();
  }

  std::string getOutput() { return resultStr; }
};
}  // namespace hip
