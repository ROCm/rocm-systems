/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "configuration.h"
#include "context.h"

#include <gmock/gmock.h>

namespace hipFile {

struct MConfiguration : Configuration {
    ContextOverride<Configuration> co;
    MConfiguration() : co{this}
    {
    }
    MOCK_METHOD(bool, fastpath, (), (const, noexcept, override));
    MOCK_METHOD(void, fastpath, (bool), (noexcept, override));
    MOCK_METHOD(bool, fallback, (), (const, noexcept, override));
    MOCK_METHOD(void, fallback, (bool), (noexcept, override));
    MOCK_METHOD(unsigned int, statsLevel, (), (const, noexcept, override));
    MOCK_METHOD(bool, unsupportedFileSystems, (), (const, noexcept, override));
};

}
