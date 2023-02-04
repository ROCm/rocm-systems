// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "categories.hpp"
#include "common/join.hpp"
#include "concepts.hpp"
#include "defines.hpp"

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

#define OMNITRACE_DECLARE_COMPONENT(NAME)                                                \
    namespace omnitrace                                                                  \
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
    struct is_component<omnitrace::component::NAME> : true_type                          \
    {};                                                                                  \
    }                                                                                    \
    }                                                                                    \
    namespace tim                                                                        \
    {                                                                                    \
    namespace component                                                                  \
    {                                                                                    \
    using ::omnitrace::component::NAME;                                                  \
    }                                                                                    \
    }

#define OMNITRACE_COMPONENT_ALIAS(NAME, ...)                                             \
    namespace omnitrace                                                                  \
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
    using ::omnitrace::component::NAME;                                                  \
    }                                                                                    \
    }

#define OMNITRACE_DEFINE_CONCRETE_TRAIT(TRAIT, TYPE, VALUE)                              \
    namespace tim                                                                        \
    {                                                                                    \
    namespace trait                                                                      \
    {                                                                                    \
    template <>                                                                          \
    struct TRAIT<::omnitrace::TYPE> : VALUE                                              \
    {};                                                                                  \
    }                                                                                    \
    }

namespace omnitrace
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

using ::tim::auto_lock_t;   // NOLINT
using ::tim::demangle;      // NOLINT
using ::tim::get_env;       // NOLINT
using ::tim::set_env;       // NOLINT
using ::tim::try_demangle;  // NOLINT
using ::tim::type_mutex;    // NOLINT

struct construct_on_thread
{
    int64_t index = threading::get_id();
};
}  // namespace omnitrace

// same sort of functionality as python's " ".join([...])
#if !defined(JOIN)
#    define JOIN(...) ::omnitrace::common::join(__VA_ARGS__)
#endif
