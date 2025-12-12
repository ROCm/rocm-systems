/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>

#pragma clang diagnostic ignored "-Wunused-function"

struct BoxStruct {
  int count;
  BoxStruct() { INFO("Constructor called for Struct!\n"); }
  ~BoxStruct() { INFO("Destructor called for Struct!\n"); }
};

class BoxClass {
 public:
  int count;
  BoxClass() { INFO("Constructor called for Class!\n"); }
  ~BoxClass() { INFO("Destructor called for Class!\n"); }
};

namespace {

void destroyStructObj(void* ptr) {
  BoxStruct* ptr1 = reinterpret_cast<BoxStruct*>(ptr);
  delete ptr1;
}

void destroyClassObj(void* ptr) {
  BoxClass* ptr2 = reinterpret_cast<BoxClass*>(ptr);
  delete ptr2;
}

void destroyIntObj(void* ptr) {
  int* ptr2 = reinterpret_cast<int*>(ptr);
  delete ptr2;
}

void destroyFloatObj(void* ptr) {
  float* ptr2 = reinterpret_cast<float*>(ptr);
  delete ptr2;
}

}  // anonymous namespace
