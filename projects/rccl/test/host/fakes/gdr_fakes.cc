/*************************************************************************
 * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
 *
 * See LICENSE.txt for license information
 ************************************************************************/

#include "gdrwrap.h"

// Host-only tests do not load gdrcopy. Keep the production global disabled so
// allocMemCPUAccessible takes its ordinary CPU-accessible allocation path.
gdr_t ncclGdrCopy = nullptr;
