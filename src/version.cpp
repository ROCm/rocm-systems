/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
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

#include "libhsakmt.h"

const char rocr4wslbuildid[] __attribute__((used)) = "ROCR4WSL BUILD ID: " STRING(ROCR4WSL_VERSION);

HSAKMT_STATUS HSAKMTAPI hsaKmtGetVersion(HsaVersionInfo *VersionInfo) {
  CHECK_DXG_OPEN();

  VersionInfo->KernelInterfaceMajorVersion = 1;
  VersionInfo->KernelInterfaceMinorVersion = 17;

  return HSAKMT_STATUS_SUCCESS;
}

HSAKMT_STATUS HSAKMTAPI
hsaKmtGetVersionCapInfo(HsaVersionCapability *VersionCapInfo) {
  CHECK_DXG_OPEN();

  VersionCapInfo->Value = 0;
  VersionCapInfo->ui64.Sdma = 1;
  /*VersionCapInfo->ui64.InterruptSignal = 0;
  VersionCapInfo->ui64.SdmaUserQueue = 0;
  VersionCapInfo->ui64.SdmaPeer = 0;
  VersionCapInfo->ui64.SdmaGang = 0;
  VersionCapInfo->ui64.Image = 0;
  VersionCapInfo->ui64.EventAge = 0;
  VersionCapInfo->ui64.Scratch = 0;
  VersionCapInfo->ui64.XNack = 0;
  VersionCapInfo->ui64.FineGrainPcie = 0;*/

  VersionCapInfo->ui64.VendorPacket = enable_vendor_packet;

  return HSAKMT_STATUS_SUCCESS;
}
