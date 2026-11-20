// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/control/session.hpp"

namespace rocprofsys::rocprofiler_sdk
{

void
setup();

/// What shutdown() does with this tool's rocprofiler-sdk client registration.
enum class client_registration
{
    release,  ///< The process is ending: ask the SDK to finalize the client.
    keep,     ///< An attach session is ending: stay registered for the next attach.
};

void
shutdown(client_registration registration);

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
