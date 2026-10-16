/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

// Symbols owned by src/transport/net_ib/gin.cc, for the microtest binaries that do not compile it.
//
// A file of its own rather than a few more lines in transport_net_ib_fakes.cc: that file is named
// for connect.cc and is linked only into rccl-UnitTestsMicroDiagnostics, so folding gin.cc's params
// into it would either mis-name it or drag connect.cc's symbols into two more binaries. Same
// TU-per-file rule the rest of fakes/ follows.
//
// Routed through g_loadParam (fakes/nccl_fakes.cc) rather than a bespoke global, so a test drives
// this the same way it drives every other NCCL_PARAM -- see fakes/param_redirect.h. The default
// mirrors production's, and test/host/CMakeLists.txt pins both halves of that claim with
// rccl_assert_source_line.

#include <cstdint>

#include "fakes/nccl_fakes.h"

int64_t ncclParamGinType() { return g_loadParam("GIN_TYPE", -1); }
