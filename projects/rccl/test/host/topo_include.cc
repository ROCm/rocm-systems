/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/graph/topo.cc for binaries that need it real but do not test it.
//
// It is pulled in with #include TOPO_CC_PATH rather than listed as a source
// file because the NETDEVS policy lives in topo.cc's file statics, and
// TopoSetNetDevsPolicy -- which TopoScene.h declares and the other src/graph
// tests call -- can only be defined from inside that TU.
//
// rccl-UnitTestsMicroTopo does NOT link this file: topo-test.cc is topo.cc's
// own test, so it includes the TU itself and carries the same definition.

#include "fakes/graph_fakes.h"
#include "fakes/param_redirect.h"

#include TOPO_CC_PATH

#include "TopoScene.h"

// The NETDEVS policy lives in this TU's statics; the other graph TUs set it through here.
void TopoSetNetDevsPolicy(enum netDevsPolicy policy, int num) {
  int unused;
  ncclTopoGetNetDevsPolicy(nullptr, &unused);  // consume the call_once first
  netDevsPolicy = policy;
  netDevsPolicyNum = num;
}
