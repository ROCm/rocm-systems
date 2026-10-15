// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/control/session.hpp"

namespace rocprofsys::rocprofiler_sdk
{

void
setup();

void
shutdown();

void
start();

void
stop();

void
pause();

void
resume();

void
set_session(std::shared_ptr<control::session> sess);

void
create_roctx_trigger();

void
reset_sdk_session_guards();

}  // namespace rocprofsys::rocprofiler_sdk
