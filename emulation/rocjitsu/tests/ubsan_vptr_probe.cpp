// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

struct Base {
  virtual ~Base() = default;
  int value = 0;
};

struct Derived : Base {};

int main(int argc, char **) {
  Base base;
  Derived derived;
  Base *object = argc > 1 ? &base : &derived;
  // The argument selects a live Base object, so the downcast requires vptr
  // instrumentation to diagnose the dynamic-type mismatch.
  return static_cast<Derived *>(object)->value;
}
