/*
Copyright (c) 2020 - present Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "rocm_smi/rocm_smi.h"
#include "rdc/rdc.h"

namespace amd {
namespace rdc {

rdc_status_t Rsmi2RdcError(rsmi_status_t rsmi) {
  switch (rsmi) {
    case RSMI_STATUS_SUCCESS:
      return RDC_ST_OK;

    case RSMI_STATUS_INVALID_ARGS:
      return RDC_ST_BAD_PARAMETER;

    case RSMI_STATUS_NOT_SUPPORTED:
      return RDC_ST_NOT_SUPPORTED;

    case RSMI_STATUS_NOT_FOUND:
      return RDC_ST_NOT_FOUND;

    case RSMI_STATUS_OUT_OF_RESOURCES:
      return RDC_ST_INSUFF_RESOURCES;

    case RSMI_STATUS_FILE_ERROR:
      return RDC_ST_FILE_ERROR;

    case RSMI_STATUS_NO_DATA:
      return RDC_ST_NO_DATA;

    case RSMI_STATUS_PERMISSION:
      return RDC_ST_PERM_ERROR;

    case RSMI_STATUS_BUSY:
    case RSMI_STATUS_UNKNOWN_ERROR:
    case RSMI_STATUS_INTERNAL_EXCEPTION:
    case RSMI_STATUS_INPUT_OUT_OF_BOUNDS:
    case RSMI_STATUS_INIT_ERROR:
    case RSMI_STATUS_NOT_YET_IMPLEMENTED:
    case RSMI_STATUS_INSUFFICIENT_SIZE:
    case RSMI_STATUS_INTERRUPT:
    case RSMI_STATUS_UNEXPECTED_SIZE:
    case RSMI_STATUS_UNEXPECTED_DATA:
    case RSMI_STATUS_REFCOUNT_OVERFLOW:
    default:
      return RDC_ST_UNKNOWN_ERROR;
  }
}

}  // namespace rdc
}  // namespace amd

