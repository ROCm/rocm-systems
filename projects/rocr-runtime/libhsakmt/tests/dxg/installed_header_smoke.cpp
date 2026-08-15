/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* The C++ half of the same check - see installed_header_smoke.c. Both
 * languages are covered because the header is consumed from both: ROCr and
 * rocprofiler-sdk are C++, and the extern "C" block only compiles as intended
 * in one of the two.
 *
 * The definition here is written exactly as src/dxg/abi.cpp writes it, so a
 * change to that signature has to be made in a place the compiler compares
 * against the header.
 */

#include <hsakmt/hsakmt.h>

extern "C" HSAKMT_STATUS HSAKMTAPI DxgAbiCheck(HsaStructureSizes* StructureSizes) {
  if (!StructureSizes ||
      StructureSizes->StructureSizes != static_cast<HSAuint16>(sizeof(HsaStructureSizes)))
    return HSAKMT_STATUS_INVALID_PARAMETER;

  return StructureSizes->SizeOfHsaNodeProperties ==
                 static_cast<HSAuint16>(sizeof(HsaNodeProperties))
             ? HSAKMT_STATUS_SUCCESS
             : HSAKMT_STATUS_DRIVER_MISMATCH;
}

/* The shape ROCr binds by dlsym(), copied from core/inc/thunk_loader.h. It is
 * a typedef of its own with nothing linking it to the thunk, so this is where
 * the two are made to agree: an incompatible declaration fails to convert.
 */
typedef HSAKMT_STATUS(DxgAbiCheckFunc)(HsaStructureSizes*);

int main() {
  DxgAbiCheckFunc* abi_check = DxgAbiCheck;

  /* Filled the way ThunkLoader::CheckThunkAbi() fills it. */
  HsaStructureSizes sizes = {};
  sizes.StructureSizes = static_cast<HSAuint16>(sizeof(HsaStructureSizes));
  sizes.SizeOfHsaNodeProperties = static_cast<HSAuint16>(sizeof(HsaNodeProperties));
  sizes.SizeOfHsaExternalHandleDesc = static_cast<HSAuint16>(sizeof(HsaHandleImportDesc));

  if (abi_check(&sizes) != HSAKMT_STATUS_SUCCESS) return 1;

  sizes.SizeOfHsaNodeProperties = static_cast<HSAuint16>(sizeof(HsaNodeProperties) - 1);
  return abi_check(&sizes) == HSAKMT_STATUS_DRIVER_MISMATCH ? 0 : 1;
}
