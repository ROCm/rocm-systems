/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#include "runtime.h"

#include "backend/fallback.h"
#include "backend/fastpath.h"
#include "hipfile-warnings.h"

#include <memory>
#include <mutex>
#include <vector>

namespace hipFile {

Runtime::Runtime() : state_{std::make_unique<DriverState>()}
{
}

Runtime::~Runtime() = default;

Runtime &
Runtime::instance()
{
    HIPFILE_WARN_NO_EXIT_DTOR_OFF
    static Runtime runtime{};
    HIPFILE_WARN_NO_EXIT_DTOR_ON
    return runtime;
}

const std::vector<std::shared_ptr<Backend>> &
Runtime::backends() const
{
    std::call_once(backends_once_, [this]() {
        std::shared_ptr<Fallback> fallback_backend = std::make_shared<Fallback>();
        std::shared_ptr<Fastpath> fastpath_backend = std::make_shared<Fastpath>();
        fastpath_backend->register_fallback_backend(fallback_backend);
        backends_.push_back(fallback_backend);
        backends_.push_back(fastpath_backend);
    });

    return backends_;
}

}
