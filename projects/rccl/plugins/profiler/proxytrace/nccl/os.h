/*************************************************************************
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * See LICENSE.txt for more license information
 *************************************************************************/

/* Minimal os.h for plugin standalone builds: only ncclPid_t is needed by
 * the profiler headers. The full RCCL os.h pulls in nccl.h and many
 * RCCL-internal APIs that plugins cannot use. */

#ifndef NCCL_OS_H_
#define NCCL_OS_H_

#if defined(NCCL_OS_WINDOWS)
#include "os/windows.h"
#elif defined(NCCL_OS_LINUX)
#include "os/linux.h"
#else
/* Default: assume Linux */
#include "os/linux.h"
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#endif
