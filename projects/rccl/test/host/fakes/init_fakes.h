/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Aggregating header for the `rccl-UnitTestsMicroInit` binary. Every seam is declared by the fakes file
// named after the PRODUCTION TU that owns the symbol; this header pulls those in and adds the few things
// that belong to src/init.cc itself. See test/host/MICROTEST_README.md.

#ifndef RCCL_TEST_HOST_INIT_FAKES_H_
#define RCCL_TEST_HOST_INIT_FAKES_H_

#include "amdsmi_fakes.h"        // src/misc/amdsmi_wrap.cc
#include "bootstrap_stubs.h"     // src/bootstrap.cc
#include "env_fakes.h"           // src/misc/param.cc + getenv interposition
#include "env_plugin_fakes.h"    // src/plugin/env.cc
#include "gin_fakes.h"           // src/plugin/gin.cc + src/gin/gin_host.cc
#include "group_fakes.h"         // src/group.cc
#include "hip_fakes.h"
#include "libc_interposers.h"    // gethostname / dladdr
#include "mem_manager_fakes.h"   // src/mem_manager.cc
#include "nccl_fakes.h"
#include "nccl_stubs.h"          // core/lifecycle stubs + data symbols
#include "os_fakes.h"            // src/os/linux.cc
#include "rccl_wrap_fakes.h"     // src/rccl_wrap.cc
#include "recorder_fakes.h"      // src/recorder.cc
#include "rocmwrap_fakes.h"      // src/misc/rocmwrap.cc
#include "strongstream_stubs.h"  // src/misc/strongstream.cc
#include "topo_stubs.h"          // src/graph/*.cc
#include "transport_stubs.h"     // src/transport/*.cc + src/plugin/net.cc
#include "tuning_fakes.h"        // src/graph/tuning.cc

struct ncclComm;

// ncclNetInit/ncclNetInitFromParent live in init-test.cc: they need the full ncclComm/ncclNet_t layout.
extern ncclResult_t g_ncclNetInitResult;

void InstallCommAllocSuccess();

void InstallDevCommSetupSuccess();

// Chains every per-TU Reset*, then clears what init.cc's own seams hold. A seam whose reset stops being
// called leaks state between tests, so anything added to a module here must be reset by that module.
void ResetInitFakes();

#endif  // RCCL_TEST_HOST_INIT_FAKES_H_
