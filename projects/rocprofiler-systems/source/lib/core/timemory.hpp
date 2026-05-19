// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common.hpp"
#include "common/defines.h"
#include "components/fwd.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components.hpp>
#include <timemory/config.hpp>
#include <timemory/environment.hpp>
#include <timemory/manager.hpp>
#include <timemory/mpl.hpp>
#include <timemory/operations.hpp>
#include <timemory/runtime.hpp>
#include <timemory/settings.hpp>
#include <timemory/storage.hpp>
#include <timemory/utility.hpp>
#include <timemory/utility/signals.hpp>
#include <timemory/variadic.hpp>

namespace rocprofsys
{
namespace audit     = ::tim::audit;      // NOLINT
namespace comp      = ::tim::component;  // NOLINT
namespace dmp       = ::tim::dmp;        // NOLINT
namespace operation = ::tim::operation;  // NOLINT
namespace quirk     = ::tim::quirk;      // NOLINT
namespace units     = ::tim::units;      // NOLINT

using settings = ::tim::settings;  // NOLINT

using ::tim::get_env;  // NOLINT
using ::tim::set_env;  // NOLINT
}  // namespace rocprofsys
