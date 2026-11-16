/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdrwrap_fakes.h"

// Keep the positional seam's trampoline declarations checked against these
// definitions without leaving its macro renames active in this file.
#include "fakes/gdrwrap_seam.h"
#include "fakes/gdrwrap_seam_undef.h"

#include "fail_loud.h"
#include "signature-drift.h"

static ncclResult_t DefaultGdrCudaMapPointer(void*, size_t, gdr_mh_t*, void**, size_t*, gdr_info_t*, uint32_t) {
  FailLoudUnfaked("gdrwrap_fakes", "ncclGdrCudaMapPointer");
}

static ncclResult_t DefaultGdrCudaUnmapPointer(gdr_mh_t, void*, size_t) {
  FailLoudUnfaked("gdrwrap_fakes", "ncclGdrCudaUnmapPointer");
}

static ncclResult_t DefaultWrapGdrCopyToMapping(gdr_mh_t, void*, const void*, size_t) {
  FailLoudUnfaked("gdrwrap_fakes", "wrap_gdr_copy_to_mapping");
}

static ncclResult_t DefaultWrapGdrCopyFromMapping(gdr_mh_t, void*, const void*, size_t) {
  FailLoudUnfaked("gdrwrap_fakes", "wrap_gdr_copy_from_mapping");
}

std::function<ncclResult_t(void*, size_t, gdr_mh_t*, void**, size_t*, gdr_info_t*, uint32_t)>
    g_ncclGdrCudaMapPointer = DefaultGdrCudaMapPointer;
std::function<ncclResult_t(gdr_mh_t, void*, size_t)> g_ncclGdrCudaUnmapPointer =
    DefaultGdrCudaUnmapPointer;
std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_wrapGdrCopyToMapping =
    DefaultWrapGdrCopyToMapping;
std::function<ncclResult_t(gdr_mh_t, void*, const void*, size_t)> g_wrapGdrCopyFromMapping =
    DefaultWrapGdrCopyFromMapping;

ASSERT_HOOK_MATCHES_PROD(g_ncclGdrCudaMapPointer, ncclGdrCudaMapPointer);
ASSERT_HOOK_MATCHES_PROD(g_ncclGdrCudaUnmapPointer, ncclGdrCudaUnmapPointer);
ASSERT_HOOK_MATCHES_PROD(g_wrapGdrCopyToMapping, wrap_gdr_copy_to_mapping);
ASSERT_HOOK_MATCHES_PROD(g_wrapGdrCopyFromMapping, wrap_gdr_copy_from_mapping);
#undef ASSERT_HOOK_MATCHES_PROD

ncclResult_t micro_ncclGdrCudaMapPointer(void* data, size_t size, gdr_mh_t* mh, void** map, size_t* mapSize,
                                         gdr_info_t* info, uint32_t pinFlags) {
  return g_ncclGdrCudaMapPointer(data, size, mh, map, mapSize, info, pinFlags);
}

ncclResult_t micro_ncclGdrCudaUnmapPointer(gdr_mh_t mh, void* map, size_t mapSize) {
  return g_ncclGdrCudaUnmapPointer(mh, map, mapSize);
}

ncclResult_t micro_wrapGdrCopyToMapping(gdr_mh_t mh, void* map, const void* host, size_t size) {
  return g_wrapGdrCopyToMapping(mh, map, host, size);
}

ncclResult_t micro_wrapGdrCopyFromMapping(gdr_mh_t mh, void* host, const void* map, size_t size) {
  return g_wrapGdrCopyFromMapping(mh, host, map, size);
}

void ResetGdrwrapFakes() {
  g_ncclGdrCudaMapPointer = DefaultGdrCudaMapPointer;
  g_ncclGdrCudaUnmapPointer = DefaultGdrCudaUnmapPointer;
  g_wrapGdrCopyToMapping = DefaultWrapGdrCopyToMapping;
  g_wrapGdrCopyFromMapping = DefaultWrapGdrCopyFromMapping;
}
