/*
 * Copyright © 2020 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

#include <stdlib.h>
#include <stdio.h>

HSAKMT_STATUS HSAKMTAPI hsaKmtSPMAcquire(HSAuint32 PreferredNode) {
  CHECK_DXG_OPEN();
  // Used for profiling tools
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSPMSetDestBuffer(
    HSAuint32 PreferredNode, HSAuint32 SizeInBytes, HSAuint32 *timeout,
    HSAuint32 *SizeCopied, void *DestMemoryAddress, bool *isSPMDataLoss) {
  CHECK_DXG_OPEN();
  // Used for profiling tools
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtSPMRelease(HSAuint32 PreferredNode) {
  CHECK_DXG_OPEN();
  // Used for profiling tools
  pr_warn_once("not supported\n");
  return HSAKMT_STATUS_NOT_SUPPORTED;
}
