/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include <cstddef>

/* Structure-size handshake for the hsaKmt* entry points this library exports.
 *
 * They copy whole structures across the interface with no size parameter:
 * hsaKmtGetNodeProperties() writes sizeof(HsaNodeProperties) bytes as *this*
 * library knows the type, into storage the caller sized as *it* knows the type.
 * A caller built against a different hsakmt revision therefore either overruns
 * its own object or reads fields the thunk never wrote, and no individual call
 * can notice - the failure is silent memory corruption.
 *
 * HsaStructureSizes is the public hsakmt type that exists to negotiate exactly
 * this, so answering it adds nothing to the interface. ROCr asks once, in
 * ThunkLoader::CheckThunkAbi() - after it has opened the thunk, resolved the
 * API table and created the thunk instance, but before it exchanges any of
 * these structures - and treats an absent symbol as an older thunk.
 *
 * An earlier revision of this port dropped the handshake because the topology
 * entry points it shipped described their own records - each reply carried the
 * ABI version it was written to and how many bytes it wrote, which is a
 * stricter check than a size, and per record rather than per library. Those
 * entry points are gone, so this is what is left. It covers less than they
 * did: treating an absent symbol as compatible means the handshake only
 * constrains runtimes new enough to ask, and the runtimes most likely to
 * disagree - ROCm 7.2.x, where sizeof(HsaNodeProperties) is 376 - are exactly
 * the ones that never call it. Against those the guard is not here but in the
 * snapshot reference count, which refuses a release it never handed out.
 *
 * A matching size is necessary but not sufficient: HsaNodeProperties has also
 * been rearranged without changing sizeof (KFDGpuID and FamilyID swapped
 * offsets in be04fa8250041), and no size comparison can see that. A caller that
 * must be sure of individual fields needs a self-describing record, not this.
 */

namespace {

/* An advertised size of 0 means the caller never exchanges that structure, so
 * there is nothing to be incompatible about. Any other value has to match
 * exactly.
 */
bool abi_size_compatible(HSAuint16 caller_size, size_t callee_size) {
  return caller_size == 0 || caller_size == static_cast<HSAuint16>(callee_size);
}

}  // namespace

extern "C" HSAKMT_STATUS HSAKMTAPI DxgAbiCheck(HsaStructureSizes* StructureSizes) {
  if (!StructureSizes) return HSAKMT_STATUS_INVALID_PARAMETER;

  /* StructureSizes is itself versioned by its size, so it has to be validated
   * before any other member is read.
   */
  if (StructureSizes->StructureSizes != static_cast<HSAuint16>(sizeof(HsaStructureSizes))) {
    pr_err("DxgAbiCheck: caller HsaStructureSizes is %u bytes, thunk expects %zu\n",
           StructureSizes->StructureSizes, sizeof(HsaStructureSizes));
    return HSAKMT_STATUS_INVALID_PARAMETER;
  }

  if (!abi_size_compatible(StructureSizes->SizeOfHsaNodeProperties, sizeof(HsaNodeProperties))) {
    pr_err("DxgAbiCheck: caller HsaNodeProperties is %u bytes, thunk uses %zu\n",
           StructureSizes->SizeOfHsaNodeProperties, sizeof(HsaNodeProperties));
    return HSAKMT_STATUS_DRIVER_MISMATCH;
  }

  /* Historical field name; the type it sizes is HsaHandleImportDesc, which
   * hsaKmtHandleImport() also takes by pointer with no size parameter.
   */
  if (!abi_size_compatible(StructureSizes->SizeOfHsaExternalHandleDesc,
                           sizeof(HsaHandleImportDesc))) {
    pr_err("DxgAbiCheck: caller HsaHandleImportDesc is %u bytes, thunk uses %zu\n",
           StructureSizes->SizeOfHsaExternalHandleDesc, sizeof(HsaHandleImportDesc));
    return HSAKMT_STATUS_DRIVER_MISMATCH;
  }

  return HSAKMT_STATUS_SUCCESS;
}
