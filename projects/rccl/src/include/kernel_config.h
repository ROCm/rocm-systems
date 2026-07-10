/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#ifndef NCCL_KERNEL_CONFIG_H_
#define NCCL_KERNEL_CONFIG_H_

#include <stddef.h>
#include <string>

// True when content contains the exact option substring (e.g. "CONFIG_FOO=y").
bool ncclKernelConfigContentHasOption(const std::string& content, const char* option);

// Read a plain kernel config file. Returns false when the file cannot be read.
bool ncclKernelConfigReadFile(const char* path, std::string* content);

// Read /proc/config.gz via zcat. Returns false when unavailable or empty.
bool ncclKernelConfigReadGzip(const char* path, std::string* content);

// Scan standard kernel config locations for the running kernel.
bool ncclKernelHasConfigOption(const char* option);

// Read the first available kernel config from the standard search paths.
// Returns false when no config file could be read.
bool ncclKernelConfigReadFirstAvailable(std::string* content, char* pathOut,
                                        size_t pathOutLen);

// True when IOMMU passthrough is configured via cmdline or kernel build default.
// A null cmdline skips the cmdline check (e.g. when /proc/cmdline could not be read).
bool ncclIommuPassthroughOk(const char* cmdline);

#endif
