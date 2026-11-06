/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// src/ras/ras.cc's communicator registry, for tests that read it without compiling ras.cc.
// fakes/ras_registry_test_support.h installs and resets the entries.

#include <mutex>

#include "comm.h"
#include "ras/ras_internal.h"

std::mutex ncclCommsMutex;
struct ncclComm** ncclComms = nullptr;
int nNcclComms = 0;
bool ncclCommsSorted = false;
