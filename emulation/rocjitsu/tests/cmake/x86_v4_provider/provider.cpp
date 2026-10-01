// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef PROVIDER_FLAG_VALUE
#define PROVIDER_FLAG_VALUE 0
#endif

extern "C" const char *provider_config() { return PROVIDER_CONFIG; }
extern "C" const char *provider_digest() { return PROVIDER_DIGEST; }
extern "C" int provider_flag() { return PROVIDER_FLAG_VALUE; }
