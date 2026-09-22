/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

#ifndef ROCRTST_COMMON_HSATIMER_H_
#define ROCRTST_COMMON_HSATIMER_H_

#include <stdint.h>
#include <iostream>
#include <vector>
#include <string>
/// \file
/// Timer related class.

namespace rocrtst {

class PerfTimer {
 private:
  struct Timer {
    std::string name; /* < name name of time object*/
    uint64_t _freq; /* < _freq frequency*/
    uint64_t _clocks; /* < _clocks number of ticks at end*/
    uint64_t _start; /* < _start start point ticks*/
  };

  std::vector<Timer*> _timers; /*< _timers vector to Timer objects */
  double freq_in_100mhz;

 public:
  PerfTimer(void);
  ~PerfTimer(void);

  /// Create a new timer.
  /// \returns A new timer instantance index
  int CreateTimer(void);

  /// Start the timer associated with the given index
  /// \param[in] index Index of the timer to start
  /// \returns int 0 for success, non-zero otherwise
  int StartTimer(int index);

  /// Stop the timer associated with the given index
  /// \param[in] Index Index of the timer to stop
  /// \returns int 0 for success, non-zero otherwise
  int StopTimer(int index);

  /// Reset the timer to 0
  /// param[in] Index of the timer to reset
  /// \returns void
  void ResetTimer(int index);

  /// Read the time value of the timer associated with the provided index.
  /// Units are seconds
  /// \param[in] index Index of the timer to read
  /// \returns double Value of the timer
  double ReadTimer(int index);

 private:
  void Error(std::string str);
  uint64_t CoarseTimestampUs();
  uint64_t MeasureTSCFreqHz();
};

}  // namespace rocrtst
#endif  // ROCRTST_COMMON_HSATIMER_H_

