/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for the symbols defined by src/misc/rocmwrap.cc. Deliberately does NOT
// include rocmwrap.h: init.cc already declares pfn_hsa_amd_portable_export_dmabuf
// through it, and pulling checks.h in here would reach the non-hipified
// recorder.h alongside the hipified copy init.cc includes.

#ifndef RCCL_TEST_HOST_ROCMWRAP_FAKES_H_
#define RCCL_TEST_HOST_ROCMWRAP_FAKES_H_

// Resets the dmaBufSupported gate (pfn_hsa_amd_portable_export_dmabuf) to NULL, i.e. unsupported.
void ResetRocmWrapFakes();

#endif  // RCCL_TEST_HOST_ROCMWRAP_FAKES_H_
