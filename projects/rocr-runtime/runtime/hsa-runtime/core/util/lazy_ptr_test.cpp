// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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

// Self move-assignment must remain well-formed and return self. operator=
// guards against self-assignment (&rhs == this), so p = std::move(p) does not
// reduce to a self-move of the underlying obj/func (which would leave func in a
// valid-but-unspecified state).
TEST(LazyPtrTest, SelfMoveAssignReturnsSelf) {
  lazy_ptr<Widget> p([]() { return new Widget(3); });

  // The self-move is deliberate here; -Wself-move would otherwise flag it.
  // GCC only knows -Wself-move from GCC 13; guard the version so older GCC does
  // not emit a -Wpragmas "unknown option" warning (fatal under -Werror).
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wself-move"
#elif defined(__GNUC__) && __GNUC__ >= 13
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wself-move"
#endif
  lazy_ptr<Widget>& result = (p = std::move(p));
#if defined(__clang__)
#pragma clang diagnostic pop
#elif defined(__GNUC__) && __GNUC__ >= 13
#pragma GCC diagnostic pop
#endif

  EXPECT_EQ(&result, &p);
  EXPECT_EQ((*p)->value, 3);  // guard preserved the object across self-move
}

}  // namespace
}  // namespace rocr
