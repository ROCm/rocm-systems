// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

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
post_process(bool&, output_file_registry&);
}  // namespace perfetto
}  // namespace rocprofsys
