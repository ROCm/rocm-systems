/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#ifndef OCL_TEST_MODULE_H
#define OCL_TEST_MODULE_H

#include <string>

#include "OCLTest.h"
#include "OCLTestList.h"

struct Module {
  std::string name;
  ModuleHandle hmodule;
  TestCountFuncPtr get_count;
  TestNameFuncPtr get_name;
  CreateTestFuncPtr create_test;
  DestroyTestFuncPtr destroy_test;
  TestVersionFuncPtr get_version;
  TestLibNameFuncPtr get_libname;
  OCLTest** cached_test;

  Module()
      : name(""),
        hmodule(0),
        get_count(0),
        get_name(0),
        create_test(0),
        destroy_test(0),
        get_version(0),
        get_libname(0),
        cached_test(0) {
    // EMPTY!
  }
};

#endif  // OCL_TEST_MODULE_H
