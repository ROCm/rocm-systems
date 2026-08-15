/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* Compiles as an independently-built consumer of the installed dev package
 * would: <hsakmt/hsakmt.h> reached through the install prefix alone, with no
 * path into this source tree. If the install rule ever ships a header without
 * something that header includes, this stops building.
 *
 * The entry point is defined here rather than linked from librocdxg. That
 * keeps the test honest - the WIN_SDK link needs libwkmi.a, which is not part
 * of this checkout - and it is the stronger check anyway: a definition has to
 * agree with the declaration it is compiled against, so a signature that
 * drifts is a compile error here rather than undefined behaviour in whatever
 * dlsym()s the symbol.
 */

#include <hsakmt/hsakmt.h>

HSAKMT_STATUS HSAKMTAPI DxgAbiCheck(HsaStructureSizes* StructureSizes) {
  if (!StructureSizes ||
      StructureSizes->StructureSizes != (HSAuint16)sizeof(HsaStructureSizes))
    return HSAKMT_STATUS_INVALID_PARAMETER;

  return StructureSizes->SizeOfHsaNodeProperties == (HSAuint16)sizeof(HsaNodeProperties)
             ? HSAKMT_STATUS_SUCCESS
             : HSAKMT_STATUS_DRIVER_MISMATCH;
}

int main(void) {
  /* Filled the way ThunkLoader::CheckThunkAbi() fills it. */
  HsaStructureSizes sizes = {0};
  sizes.StructureSizes = (HSAuint16)sizeof(HsaStructureSizes);
  sizes.SizeOfHsaNodeProperties = (HSAuint16)sizeof(HsaNodeProperties);
  sizes.SizeOfHsaExternalHandleDesc = (HSAuint16)sizeof(HsaHandleImportDesc);

  if (DxgAbiCheck(&sizes) != HSAKMT_STATUS_SUCCESS) return 1;

  sizes.SizeOfHsaNodeProperties = (HSAuint16)(sizeof(HsaNodeProperties) - 1);
  return DxgAbiCheck(&sizes) == HSAKMT_STATUS_DRIVER_MISMATCH ? 0 : 1;
}
