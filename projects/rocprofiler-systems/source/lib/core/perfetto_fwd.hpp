// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace tim
{
class manager;
}

namespace rocprofsys
{
namespace perfetto
{
void
setup();

void
start();

void
stop();

void
post_process(tim::manager*, bool&);
}  // namespace perfetto
}  // namespace rocprofsys
