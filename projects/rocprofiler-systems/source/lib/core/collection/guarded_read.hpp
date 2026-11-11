// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "logger/debug.hpp"

#include <cstddef>
#include <exception>
#include <string_view>
#include <utility>

namespace rocprofsys::collection
{
/** Reads one device unless it has failed before. On an exception: logs once (source,
 * device index, what) and sets `failed`, so the device is skipped from then on.
 * @return true when `read` ran and succeeded */
template <typename ReadFn>
[[nodiscard]] bool
guarded_read(bool& failed, std::string_view source, std::size_t device_index,
             ReadFn&& read)
{
    if(failed)
    {
        return false;
    }
    try
    {
        std::forward<ReadFn>(read)();
        return true;
    } catch(const std::exception& error)
    {
        LOG_ERROR("collection: {} device {} disabled: {}", source, device_index,
                  error.what());
    } catch(...)
    {
        LOG_ERROR("collection: {} device {} disabled: unknown exception", source,
                  device_index);
    }
    failed = true;
    return false;
}
}  // namespace rocprofsys::collection
