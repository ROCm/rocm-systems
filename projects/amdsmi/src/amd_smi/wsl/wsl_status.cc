/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

#include "amd_smi/impl/wsl/wsl_status.h"

#include <ntstatus.h>

namespace wsl {

amdsmi_status_t ToAmdsmiStatus(NTSTATUS status) {
  switch (status) {
    case STATUS_SUCCESS:
      return AMDSMI_STATUS_SUCCESS;
    case STATUS_PENDING:
      return AMDSMI_STATUS_RETRY;
    case STATUS_NO_MEMORY:
      return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case STATUS_DEVICE_REMOVED:
      return AMDSMI_STATUS_NOT_FOUND;
    case STATUS_GRAPHICS_NO_VIDEO_MEMORY:
      return AMDSMI_STATUS_OUT_OF_RESOURCES;
    case STATUS_TIMEOUT:
      return AMDSMI_STATUS_TIMEOUT;
    case STATUS_INVALID_PARAMETER:
      return AMDSMI_STATUS_INVAL;
    case STATUS_INVALID_HANDLE:
      return AMDSMI_STATUS_INVAL;
    case STATUS_NOT_SUPPORTED:
      return AMDSMI_STATUS_NOT_SUPPORTED;
    case STATUS_NOT_IMPLEMENTED:
      return AMDSMI_STATUS_NOT_YET_IMPLEMENTED;
    case STATUS_BUFFER_TOO_SMALL:
      return AMDSMI_STATUS_INSUFFICIENT_SIZE;
    default:
      return AMDSMI_STATUS_API_FAILED;
  }
}

}  // namespace wsl
