// Stub hip_version.h — version high enough to select simple codepaths
// in rccl_float8.h (>= 60300000 uses hip_fp8.h types) but low enough
// to avoid ROCM_VERSION-gated GPU features.
#pragma once
#define HIP_VERSION_MAJOR 6
#define HIP_VERSION_MINOR 3
#define HIP_VERSION_PATCH 0
#define HIP_VERSION 60300000
#define HIP_VERSION_GITHASH "stub"
