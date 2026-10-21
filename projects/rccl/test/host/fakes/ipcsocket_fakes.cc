/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Fakes for src/os/linux_ipcsocket.cc symbols, for binaries that don't compile it.

#include <unistd.h>

#include "ipcsocket.h"

// p2p.cc, dev_runtime.cc, proxy.cc and mem_manager.cc close imported fds through this instead of
// close(2). A real close, as production does, so suites that hand back a dup'd fd can still prove it
// was closed.
int ncclIpcFdClose(ncclIpcFd fd) { return ::close(fd); }
