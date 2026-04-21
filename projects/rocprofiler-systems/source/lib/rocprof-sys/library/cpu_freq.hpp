// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

namespace rocprofsys
{
namespace cpu_freq
{
void
setup();

void
config();

void
sample();

void
shutdown();

void
pause();

void
post_process();
}  // namespace cpu_freq
}  // namespace rocprofsys
