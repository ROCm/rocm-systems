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

// The race-test binary checks itself: ThreadSanitizer must be active, or every
// race test in it would pass without looking.

#include <gtest/gtest.h>

#include <cstdlib>
#include <thread>

namespace {

int g_unsynchronized = 0;

__attribute__((noinline)) void write_unsynchronized() { g_unsynchronized = g_unsynchronized + 1; }

// Two threads write one int without synchronization, then the process exits.
// ThreadSanitizer reports the race and exits with 66, by default and with
// halt_on_error=1 alike; an uninstrumented binary exits with 0.
void plant_a_race_and_exit() {
  std::thread a(write_unsynchronized);
  std::thread b(write_unsynchronized);
  a.join();
  b.join();
  std::exit(0);
}

}  // namespace

TEST(RaceSanity, ThreadSanitizerReportsAPlantedRace) {
  EXPECT_EXIT(plant_a_race_and_exit(), testing::ExitedWithCode(66), "ThreadSanitizer: data race");
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  // Death tests re-execute the binary ("threadsafe") instead of forking a
  // threaded process.
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  return RUN_ALL_TESTS();
}
