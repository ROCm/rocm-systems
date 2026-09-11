// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "aca_version.h"

/* Implementation of version functions */

int aca_get_version_major(void) { return RAS_DECODE_VERSION_MAJOR; }

int aca_get_version_minor(void) { return RAS_DECODE_VERSION_MINOR; }

int aca_get_version_patch(void) { return RAS_DECODE_VERSION_PATCH; }

const char* aca_get_version_string(void) { return RAS_DECODE_VERSION_STRING; }

aca_version_info_t aca_get_version_info(void) {
  aca_version_info_t info;

  info.major = RAS_DECODE_VERSION_MAJOR;
  info.minor = RAS_DECODE_VERSION_MINOR;
  info.patch = RAS_DECODE_VERSION_PATCH;
  info.string = RAS_DECODE_VERSION_STRING;

  return info;
}
