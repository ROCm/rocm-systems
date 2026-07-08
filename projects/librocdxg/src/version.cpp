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


const char rocdxgbuildid[] __attribute__((used)) = "ROCDXG BUILD ID: " STRING(ROCDXG_VERSION);

HSAKMT_STATUS HSAKMTAPI hsaKmtGetVersion(HsaVersionInfo *VersionInfo) {
  CHECK_DXG_OPEN();

  VersionInfo->KernelInterfaceMajorVersion = 1;
  VersionInfo->KernelInterfaceMinorVersion = 17;

  return HSAKMT_STATUS_SUCCESS;
}

HsaStructureSizes detected_abi_{};
HSAKMT_STATUS HSAKMTAPI DxgAbiCheck(HsaStructureSizes *actual)
{
  if (actual == nullptr)
  {
    fprintf(stderr, "[ROCr/DXG] DxgAbiCheck: received nullptr.\n");
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }

  if (actual->StructureSizes != sizeof(HsaStructureSizes)) {
    bool is_larger = actual->StructureSizes > sizeof(HsaStructureSizes);
    fprintf(stderr,
            "[ROCr/DXG] DxgAbiCheck: StructureSizes=%u is too %s (expected %zu). Please update your "
            "%s version.\n",
            actual->StructureSizes, is_larger ? "large" : "small", sizeof(HsaStructureSizes),
            is_larger ? "librocdxg" : "ROCr");

    return HSAKMT_STATUS_INVALID_PARAMETER;
  }

  detected_abi_ = *actual;

  return HSAKMT_STATUS_SUCCESS;
}