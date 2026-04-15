/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "stats.h"

#include <gmock/gmock.h>

namespace hipFile {
class MStatsServer : public StatsServer {
    ContextOverride<StatsServer> co;

public:
    MStatsServer() : co{this}
    {
    }
    MOCK_METHOD(Stats *, getStats, (), (override));
};
}
