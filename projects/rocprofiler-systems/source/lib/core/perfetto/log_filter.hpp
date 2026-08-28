// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys::core::log_filter
{

void
register_with_perfetto_logger();

void
unregister_from_perfetto_logger();

}  // namespace rocprofsys::core::log_filter
