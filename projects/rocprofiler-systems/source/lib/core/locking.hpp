// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <timemory/utility/locking.hpp>

#include <atomic>
#include <mutex>

namespace rocprofsys
{
namespace locking
{
using atomic_mutex = ::tim::locking::spin_mutex;
using atomic_lock  = ::tim::locking::spin_lock;
}  // namespace locking
}  // namespace rocprofsys
