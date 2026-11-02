/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Stand-in for roctracer's <roctracer/roctx.h>, used only by the standalone
// rccl-UnitTestsMicroRoctxNvml-roctx target so roctx.cc's ROCTX_ENABLE arm
// compiles whether or not roctracer is installed. Same C-linkage declarations
// as the real header for the two entry points roctx.cc calls;
// roctx-nvml-stub-test.cc defines them to record each call.

#ifndef RCCL_TEST_HOST_FAKES_ROCTRACER_ROCTX_H_
#define RCCL_TEST_HOST_FAKES_ROCTRACER_ROCTX_H_

#ifdef __cplusplus
extern "C" {
#endif

int roctxRangePushA(const char* message);
int roctxRangePop(void);

#ifdef __cplusplus
}
#endif

#endif // RCCL_TEST_HOST_FAKES_ROCTRACER_ROCTX_H_
