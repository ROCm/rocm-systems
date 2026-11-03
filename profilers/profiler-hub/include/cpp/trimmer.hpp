// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/**
 * @file trimmer.hpp
 * @brief Public C++ entry for the trimmer API.
 *
 * Trimming is not part of reader_t or writer_t. Open a trace with the reader
 * API, then write a trimmed copy through ph_trim_trace().
 */

#pragma once

#include <profiler-hub/c/profiler_hub_trim.h>
