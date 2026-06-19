// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace tim
{
class manager;
}

namespace rocprofsys
{

class output_file_registry;

namespace perfetto
{
void
setup();

void
start();

void
stop();

void
post_process(tim::manager*, bool&, output_file_registry&);
}  // namespace perfetto
}  // namespace rocprofsys
