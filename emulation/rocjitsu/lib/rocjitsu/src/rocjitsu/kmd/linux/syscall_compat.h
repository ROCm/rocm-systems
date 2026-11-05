// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <sys/syscall.h>

// Some build hosts have older headers than their running kernel. These syscall
// numbers are shared by all architectures ROCm targets.
#ifndef SYS_pidfd_open
#ifdef __NR_pidfd_open
#define SYS_pidfd_open __NR_pidfd_open
#else
#define SYS_pidfd_open 434
#endif
#endif

#ifndef SYS_close_range
#ifdef __NR_close_range
#define SYS_close_range __NR_close_range
#else
#define SYS_close_range 436
#endif
#endif
