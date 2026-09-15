// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Pluggable logging sink for rocm/sha2, kept separate from sha256.h so a
// project only wiring up logging does not need to pull in the hashing API.

#pragma once

namespace rocm {
namespace sha2 {

// Reports a diagnostic message. Default implementation writes to stderr;
// install a different one with set_log_handler() to route into a project's
// own logging instead. This keeps the shared implementation free of any
// dependency on a particular logging library or format.
using log_handler = void (*)(const char *message);

// Installs `handler` as the log callback for this process; passing nullptr
// restores the default stderr handler. Intended to be called once during
// startup.
void set_log_handler(log_handler handler);

} // namespace sha2
} // namespace rocm
