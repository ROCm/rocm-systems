// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
#pragma once

#include <nlohmann/json.hpp>

namespace addc
{

// JSON types of the public C++ API. Consumers should name these rather than
// nlohmann:: types, so the underlying type can change here in one place.
using json_value = nlohmann::json;
// Keeps object keys in insertion order.
using ordered_json_value = nlohmann::ordered_json;

} // namespace addc
