/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* The profiler ABI headers in src/include/plugin need only ncclPid_t from
 * os.h. */

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#if defined(NCCL_OS_WINDOWS)
typedef unsigned long ncclPid_t;
#else
#include <sys/types.h>
typedef pid_t ncclPid_t;
#endif

#endif
