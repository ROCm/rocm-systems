/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "context.h"
#include "stats.h"

#include <gmock/gmock.h>

namespace hipFile {
class MStatsServer : public IStatsServer {
    ContextOverride<IStatsServer> co;

public:
    MStatsServer() : co{this}
    {
    }
    MOCK_METHOD(Stats *, getStats, (), (noexcept, override));
};

class MStatsCollection : public StatsCollection {
    ContextOverride<StatsCollection> co;

public:
    MStatsCollection() : co{this}
    {
    }
    MOCK_METHOD(void, addIo, (IoType ioType, StatsBackend backend, uint64_t bytes, uint64_t timeUs),
                (const, noexcept, override));
    MOCK_METHOD(void, error, (IoType ioType, StatsBackend backend, uint64_t bytes),
                (const, noexcept, override));
};
}
