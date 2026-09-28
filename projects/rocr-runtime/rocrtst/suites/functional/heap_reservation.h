/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2026, Advanced Micro Devices, Inc.
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

#ifndef ROCRTST_SUITES_FUNCTIONAL_HEAP_RESERVATION_H_
#define ROCRTST_SUITES_FUNCTIONAL_HEAP_RESERVATION_H_

#include <stdint.h>

#include "common/base_rocr.h"
#include "hsa/hsa.h"
#include "suites/test_common/test_base.h"

// WSL/DXG only. The DXG thunk reserves two pools of virtual address space
// during hsa_init(), sized from VRAM and from host RAM. They cost no memory,
// but RLIMIT_AS can still refuse them.
//
// ROCM-30547: when the host-RAM pool was refused the thunk reported success
// anyway and left that pool's allocator null. The next system-memory
// allocation - ROCr's shared signal pool, during hsa_init() itself -
// dereferenced it and the process died with SIGSEGV. The fix retries at 1/2
// and 1/4 of RAM and returns an error if nothing fits.
//
// Each budget runs in a forked child, since RLIMIT_AS cannot be raised once
// lowered. Children run silently; use -v 2 to see their logging.
class HeapReservationTest : public TestBase {
 public:
  HeapReservationTest();
  virtual ~HeapReservationTest();

  virtual void SetUp();
  virtual void Run();
  virtual void Close();
  virtual void DisplayResults() const;
  virtual void DisplayTestInfo();

  void HeapReservationSurvivesConstrainedAddressSpace();

 private:
  // Address space a child may map on top of what it already has.
  struct Budgets {
    uint64_t generous = 0;   // every pool size fits, including 1x RAM
    uint64_t ladder = 0;     // 1x refused, a smaller one fits
    uint64_t tight = 0;      // nothing fits
    bool valid = false;
    bool ladder_possible = false;  // false when 1/2 RAM is below the floor
  };

  Budgets ComputeBudgets(uint64_t vram, uint64_t ram) const;

  uint64_t vram_ = 0;
  uint64_t ram_ = 0;
};

#endif  // ROCRTST_SUITES_FUNCTIONAL_HEAP_RESERVATION_H_
