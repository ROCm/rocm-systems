/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Reusable stubs for the RCCL device-runtime layer -- the `devr`, `gin`,
// `rma`, `team`, shadow-pool, space-allocator and symmetric-kernel symbols a
// host-only micro-test references when it #includes a production TU such as
// src/dev_runtime.cc.
//
// These are not p2p-specific and carry no controllable seams: every entry is
// an inert success stub (out-params zeroed) that lets the binary link without
// librccl.so or a GPU. Generic `nccl*`/`bootstrap*` symbols live in
// nccl_fakes.{h,cc}; HIP runtime seams live in hip_fakes.{h,cc}.
//
// There is intentionally no reset/seam API here -- if a future test needs to
// control one of these, promote it to a std::function seam following the
// pattern in hip_fakes.h / nccl_fakes.h.

#ifndef RCCL_TEST_HOST_DEVR_FAKES_H_
#define RCCL_TEST_HOST_DEVR_FAKES_H_

// This header documents the devr fakes layer; all definitions live in
// devr_fakes.cc and are declared by the production headers they stand in for.

#endif  // RCCL_TEST_HOST_DEVR_FAKES_H_
