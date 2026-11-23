/*
Copyright (c) 2026 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

// field_id_string(), get_rocm_path() and RDC_LOG, all from librdc_bootstrap.

#include <gtest/gtest.h>
#include <link.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <regex>
#include <set>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "rdc/rdc.h"
#include "rdc_lib/RdcLogger.h"

namespace {

// No field has this ID (rdc.h), and it is within rdc_field_t's range.
constexpr rdc_field_t kUnknownField = static_cast<rdc_field_t>(9999);

// A library no process maps, so get_rocm_path() falls back to ROCM_PATH or /opt/rocm.
constexpr char kNoLibrary[] = "librdc_no_such_library.so";

// field_id_string() of an unknown ID read past the field table; it runs in a child process,
// which that can crash.
TEST(FieldIdStringTest, NamesAnUnknownFieldInvalid) {
  EXPECT_STREQ(field_id_string(RDC_FI_GPU_COUNT), "GPU_COUNT");
  EXPECT_STREQ(field_id_string(RDC_FI_INVALID), "INVALID");
  EXPECT_EXIT(_exit(strcmp(field_id_string(kUnknownField), "INVALID") == 0 ? 0 : 1),
              ::testing::ExitedWithCode(0), "");
}

// Both ROCM_PATH values are short enough for one std::string's own buffer, so a result that
// shares storage with a later call's shows the later value.
TEST(GetRocmPathTest, ResultsKeepTheirValues) {
  ASSERT_EQ(setenv("ROCM_PATH", "/first/rocm", 1), 0);
  const char* first = get_rocm_path(kNoLibrary);
  ASSERT_NE(first, nullptr);
  EXPECT_STREQ(first, "/first/rocm");

  ASSERT_EQ(setenv("ROCM_PATH", "/second/rocm", 1), 0);
  const char* second = get_rocm_path(kNoLibrary);
  ASSERT_NE(second, nullptr);
  EXPECT_STREQ(second, "/second/rocm");
  EXPECT_STREQ(first, "/first/rocm");

  ASSERT_EQ(unsetenv("ROCM_PATH"), 0);
  EXPECT_STREQ(get_rocm_path(kNoLibrary), "/opt/rocm");
}

// The canonical form of `path`, or "" if it does not resolve.
std::string canonical(const std::string& path) {
  char* resolved = realpath(path.c_str(), nullptr);
  if (resolved == nullptr) {
    return "";
  }
  std::string result(resolved);
  free(resolved);
  return result;
}

// The directory of the loaded object whose file name starts with `prefix`, or "".
std::string loaded_library_dir(const std::string& prefix) {
  std::pair<std::string, std::string> search{prefix, ""};
  dl_iterate_phdr(
      [](struct dl_phdr_info* info, size_t, void* data) {
        auto* s = static_cast<std::pair<std::string, std::string>*>(data);
        const std::string path = info->dlpi_name != nullptr ? info->dlpi_name : "";
        const size_t slash = path.find_last_of('/');
        if (slash == std::string::npos || path.compare(slash + 1, s->first.size(), s->first) != 0) {
          return 0;
        }
        s->second = path.substr(0, slash);
        return 1;
      },
      &search);
  return search.second;
}

// A mapped library gives the parent of its directory.
TEST(GetRocmPathTest, FindsAMappedLibrary) {
  const std::string dir = loaded_library_dir("librdc_bootstrap.so");
  ASSERT_FALSE(dir.empty()) << "librdc_bootstrap.so is not loaded";

  ASSERT_EQ(unsetenv("ROCM_PATH"), 0);
  const char* path = get_rocm_path("librdc_bootstrap.so");
  ASSERT_NE(path, nullptr);
  const std::string rocm(path);
  ASSERT_GE(rocm.size(), 3u);
  EXPECT_EQ(rocm.substr(rocm.size() - 3), "/..") << rocm;
  EXPECT_EQ(canonical(rocm), canonical(dir + "/..")) << rocm;
}

// A null search string crashed in std::string::find(); it runs in a child process.
TEST(GetRocmPathTest, AcceptsANullSearchString) {
  ASSERT_EQ(unsetenv("ROCM_PATH"), 0);
  EXPECT_EXIT(
      {
        const char* path = get_rocm_path(nullptr);
        _exit(path != nullptr && strcmp(path, "/opt/rocm") == 0 ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}

// Calls from several threads at once; ThreadSanitizer checks how they share state.
TEST(GetRocmPathTest, ConcurrentCallsAgree) {
  ASSERT_EQ(setenv("ROCM_PATH", "/concurrent/rocm", 1), 0);
  constexpr int kThreads = 8;
  constexpr int kCalls = 50;
  std::atomic<int> wrong{0};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([&wrong] {
      for (int i = 0; i < kCalls; i++) {
        const char* path = get_rocm_path(kNoLibrary);
        if (path == nullptr || strcmp(path, "/concurrent/rocm") != 0) {
          wrong++;
        }
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  EXPECT_EQ(wrong.load(), 0);
  ASSERT_EQ(unsetenv("ROCM_PATH"), 0);
}

// RDC_LOG lines from 8 threads at once must each come out whole. The logger writes to
// std::cout, so fd 1 goes to a file during the test.
TEST(RdcLogTest, ConcurrentLinesDoNotInterleave) {
  constexpr int kThreads = 8;
  constexpr int kLines = 200;
  std::string path = ::testing::TempDir() + "rdc_log_XXXXXX";
  const int file = mkstemp(&path[0]);
  ASSERT_GE(file, 0);
  std::cout.flush();
  fflush(stdout);
  const int saved_stdout = dup(STDOUT_FILENO);
  ASSERT_GE(saved_stdout, 0);
  ASSERT_EQ(dup2(file, STDOUT_FILENO), STDOUT_FILENO);
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; t++) {
    threads.emplace_back([t] {
      for (int i = 0; i < kLines; i++) {
        RDC_LOG(RDC_ERROR, "thread " << t << " line " << i << " end");
      }
    });
  }
  for (auto& thread : threads) {
    thread.join();
  }
  std::cout.flush();
  fflush(stdout);
  ASSERT_EQ(dup2(saved_stdout, STDOUT_FILENO), STDOUT_FILENO);
  close(saved_stdout);
  close(file);

  std::ifstream log(path);
  const std::regex whole(
      R"(\d+\.\d{3} ERROR test_bootstrap_checks\.cc\(\d+\): thread (\d+) line (\d+) end)");
  std::set<std::pair<int, int>> lines;
  std::vector<std::string> broken;
  std::string line;
  while (std::getline(log, line)) {
    std::smatch match;
    if (std::regex_match(line, match, whole)) {
      lines.insert({std::stoi(match[1].str()), std::stoi(match[2].str())});
    } else if (broken.size() < 3) {
      broken.push_back(line);
    }
  }
  unlink(path.c_str());
  std::string examples;
  for (const auto& b : broken) {
    examples += "\n  " + b;
  }
  EXPECT_TRUE(broken.empty()) << "broken lines, for example:" << examples;
  EXPECT_EQ(lines.size(), static_cast<size_t>(kThreads * kLines));
}

}  // namespace
