// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "common/defines.h"
#include "core/categories.hpp"
#include "core/concepts.hpp"

#include <timemory/api.hpp>
#include <timemory/api/macros.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/environment/types.hpp>
#include <timemory/mpl/types.hpp>
#include <timemory/utility/demangle.hpp>
#include <timemory/utility/filepath.hpp>
#include <timemory/utility/locking.hpp>

#include <cassert>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <unistd.h>
#include <utility>
#include <vector>

#define ROCPROFSYS_DECLARE_COMPONENT(NAME)                                               \
    namespace rocprofsys                                                                 \
    {                                                                                    \
    namespace component                                                                  \
    {                                                                                    \
    struct NAME;                                                                         \
    }                                                                                    \
    }                                                                                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct is_component<rocprofsys::component::NAME> : true_type                         \
    {};                                                                                  \
    }                                                                                    \
    }                                                                                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace component                                                                  \
    {                                                                                    \
    using ::rocprofsys::component::NAME;                                                 \
    }                                                                                    \
    }

#define ROCPROFSYS_COMPONENT_ALIAS(NAME, ...)                                            \
    namespace rocprofsys                                                                 \
    {                                                                                    \
    namespace component                                                                  \
    {                                                                                    \
    using NAME = __VA_ARGS__;                                                            \
    }                                                                                    \
    }                                                                                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace component                                                                  \
    {                                                                                    \
    using ::rocprofsys::component::NAME;                                                 \
    }                                                                                    \
    }

#define ROCPROFSYS_DEFINE_CONCRETE_TRAIT(TRAIT, TYPE, VALUE)                             \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct TRAIT<::rocprofsys::TYPE> : VALUE                                             \
    {};                                                                                  \
    }                                                                                    \
    }

namespace rocprofsys
{
namespace api       = ::tim::api;        // NOLINT
namespace category  = ::tim::category;   // NOLINT
namespace filepath  = ::tim::filepath;   // NOLINT
namespace project   = ::tim::project;    // NOLINT
namespace process   = ::tim::process;    // NOLINT
namespace threading = ::tim::threading;  // NOLINT
namespace scope     = ::tim::scope;      // NOLINT
namespace policy    = ::tim::policy;     // NOLINT
namespace trait     = ::tim::trait;      // NOLINT
namespace cereal    = ::tim::cereal;     // NOLINT

using ::tim::auto_lock_t;  // NOLINT
using ::tim::get_env;      // NOLINT
using ::tim::set_env;      // NOLINT
using ::tim::type_mutex;   // NOLINT

struct construct_on_thread
{
    int64_t index = threading::get_id();
};
}  // namespace rocprofsys
