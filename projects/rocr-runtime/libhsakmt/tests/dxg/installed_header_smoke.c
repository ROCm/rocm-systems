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
 *
 * hsaKmtGetNodeProperties() is the entry point chosen for it because it is the
 * one that copies a whole HsaNodeProperties across the interface with no size
 * parameter, so the header each side is compiled against is the only thing
 * that says how large that record is.
 */

#include <hsakmt/hsakmt.h>

HSAKMT_STATUS HSAKMTAPI hsaKmtGetNodeProperties(HSAuint32 NodeId,
                                                HsaNodeProperties* NodeProperties) {
  if (!NodeProperties) return HSAKMT_STATUS_INVALID_PARAMETER;
  if (NodeId != 0) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  NodeProperties->NumCPUCores = 1;
  return HSAKMT_STATUS_SUCCESS;
}

int main(void) {
  HsaNodeProperties props = {0};

  if (hsaKmtGetNodeProperties(0, &props) != HSAKMT_STATUS_SUCCESS) return 1;
  if (props.NumCPUCores != 1) return 1;

  return hsaKmtGetNodeProperties(1, &props) == HSAKMT_STATUS_INVALID_NODE_UNIT ? 0 : 1;
}
