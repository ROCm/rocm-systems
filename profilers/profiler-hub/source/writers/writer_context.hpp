// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "data_storage/backends/sqlite_backend.hpp"
#include "entity_registry.hpp"
#include "insert_validator.hpp"
#include "primary_key_providers.hpp"

#include <memory>
#include <string>

namespace profiler_hub
{

struct writer_context
{
    std::shared_ptr<data_storage::sqlite_backend> backend;
    std::shared_ptr<entity_registry>              registry;
    std::shared_ptr<primary_key_providers>        key_providers;
    std::shared_ptr<insert_validator>             validator;
    std::string                                   uuid;
};

}  // namespace profiler_hub
