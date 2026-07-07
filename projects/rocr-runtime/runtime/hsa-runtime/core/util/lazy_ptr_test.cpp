////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// Regression test for the lazy_ptr<T> move-assignment operator.
//
// lazy_ptr::operator=(lazy_ptr&&) is declared to return lazy_ptr& but
// historically had no `return` statement. Falling off the end of a non-void
// function is undefined behavior: the returned reference is whatever garbage
// is in the return register. When the result is subsequently used (e.g. the
// std::vector relocation done during GpuAgent::InitDma()), a later reset()
// operates on a corrupted std::function and jumps to an invalid address,
// crashing hsa_init(). See ROCm/TheRock#5985.
//
// These tests pin the observable contract of move-assignment so the missing
// `return *this;` cannot regress silently.

#include "gtest/gtest.h"

#include "core/util/lazy_ptr.h"

namespace rocr {
namespace {

struct Widget {
  explicit Widget(int v) : value(v) {}
  int value;
};

// The core regression: operator= must return a reference to the assigned-to
// object. With the missing `return *this;` the returned reference points at
// garbage, so its address does not match &dst.
TEST(LazyPtrTest, MoveAssignReturnsSelf) {
  lazy_ptr<Widget> dst;
  lazy_ptr<Widget> src([]() { return new Widget(7); });

  lazy_ptr<Widget>& result = (dst = std::move(src));

  EXPECT_EQ(&result, &dst);
}

// Feeding the result of one move-assignment into another relies on the
// returned reference being the real object. If operator= returns garbage,
// std::move(dst = ...) reads a bogus obj/func and the second assignment
// corrupts or crashes.
TEST(LazyPtrTest, MoveAssignResultIsUsable) {
  lazy_ptr<Widget> a;
  lazy_ptr<Widget> b;
  lazy_ptr<Widget> c([]() { return new Widget(42); });

  a = std::move(b = std::move(c));

  EXPECT_TRUE(a.empty());          // constructor not yet run
  EXPECT_EQ((*a)->value, 42);      // forces construction; must not crash
}

// Reproduces the crashing sequence: move-assign into a target, then reset().
// A garbage return corrupts the internal std::function such that the reset()
// (which does `func = std::move(...)` and destroys the prior target) faults.
TEST(LazyPtrTest, ResetAfterMoveAssignIsSafe) {
  lazy_ptr<Widget> dst;
  lazy_ptr<Widget> src([]() { return new Widget(1); });

  dst = std::move(src);
  dst.reset([]() { return new Widget(2); });

  ASSERT_TRUE(dst.empty());        // constructor not yet run
  EXPECT_EQ((*dst)->value, 2);     // forces construction; must not crash
  EXPECT_TRUE(dst.created());
}

// Self move-assignment must remain well-formed and return self.
TEST(LazyPtrTest, SelfMoveAssignReturnsSelf) {
  lazy_ptr<Widget> p([]() { return new Widget(3); });

  lazy_ptr<Widget>& result = (p = std::move(p));

  EXPECT_EQ(&result, &p);
}

}  // namespace
}  // namespace rocr
