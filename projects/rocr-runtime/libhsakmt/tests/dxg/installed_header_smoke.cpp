/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/* The C++ half of the same check - see installed_header_smoke.c. Both
 * languages are covered because the header is consumed from both: ROCr and
 * rocprofiler-sdk are C++, and the extern "C" block only compiles as intended
 * in one of the two.
 */

#include <hsakmt/hsakmt.h>

extern "C" HSAKMT_STATUS HSAKMTAPI
hsaKmtGetNodeProperties(HSAuint32 NodeId, HsaNodeProperties* NodeProperties) {
  if (!NodeProperties) return HSAKMT_STATUS_INVALID_PARAMETER;
  if (NodeId != 0) return HSAKMT_STATUS_INVALID_NODE_UNIT;

  NodeProperties->NumCPUCores = 1;
  return HSAKMT_STATUS_SUCCESS;
}

/* The shape ROCr binds by dlsym(), copied from core/inc/thunk_loader.h, where
 * it is spelled through the HSAKMT_DEF() macro. It is a typedef of its own
 * with nothing linking it to the thunk, so this is where the two are made to
 * agree: an incompatible declaration fails to convert.
 */
typedef HSAKMT_STATUS(PFNhsaKmtGetNodeProperties)(HSAuint32, HsaNodeProperties*);

int main() {
  PFNhsaKmtGetNodeProperties* get_node_properties = hsaKmtGetNodeProperties;

  HsaNodeProperties props = {};

  if (get_node_properties(0, &props) != HSAKMT_STATUS_SUCCESS) return 1;
  if (props.NumCPUCores != 1) return 1;

  return get_node_properties(1, &props) == HSAKMT_STATUS_INVALID_NODE_UNIT ? 0 : 1;
}
