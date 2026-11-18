// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/base/rj_version.h"

#include "rocjitsu/version.h"

namespace {

constexpr char kVersionString[] =
    "rocjitsu " ROCJITSU_VERSION "\n"
    "git revision: " ROCJITSU_GIT_REVISION "\n"
    "git commit: " ROCJITSU_GIT_COMMIT_DATE " " ROCJITSU_GIT_COMMIT_TITLE;

} // namespace

extern "C" const char *rj_get_version_string(void) { return kVersionString; }
