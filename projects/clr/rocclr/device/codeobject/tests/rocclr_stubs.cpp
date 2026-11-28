/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// librocclr.a leaves the ICD dispatch table to the runtime that links it (OpenCL's cl_icd.cpp,
// or HIP's inline definition); the tests link only rocclr.

#include "platform/object.hpp"

cl_icd_dispatch amd::ICDDispatchedObject::icdVendorDispatch_[1] = {};
