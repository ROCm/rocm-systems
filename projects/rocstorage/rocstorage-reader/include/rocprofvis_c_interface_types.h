// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

/*
* When built with CFFI, headers must not include directives.
* This is wrapper around InterfaceType.h and must be used from C/C++ code only.
* CFFI compiler will use rocprofvis_interface_types.h directly.
*/

#include <cstdint>

#include "rocprofvis_interface_types.h"
