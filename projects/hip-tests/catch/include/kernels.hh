/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <hip_test_common.hh>
#include <map>

#ifndef RTC_TESTING

__global__ void Set(int* Ad, int val);

/* Kernel Templates */
#include "vectorADD.inl"

#else

/*
 * Wrapper Macros that create a string representation of the kernel name.
 * In the case of kernel templates, a variadic template is used to ensure compatibility with
 * the launchKernel template when RTC is not enabled. If the kernel is inside a namespace, use the
 * "_NS" version of the Macro.
 */
#define FUNCTION_WRAPPER(param)                                                                    \
  std::string param() { return #param; }
#define TEMPLATE_WRAPPER(param)                                                                    \
  template <typename...> std::string param() { return #param; }
#define FUNCTION_WRAPPER_NS(param, namespace)                                                      \
  std::string param() { return #namespace "::" #param; }
#define TEMPLATE_WRAPPER_NS(param, namespace)                                                      \
  template <typename...> std::string param() { return #namespace "::" #param; }

FUNCTION_WRAPPER(Set);

namespace HipTest {
TEMPLATE_WRAPPER_NS(vectorADD, HipTest);
}

#endif
