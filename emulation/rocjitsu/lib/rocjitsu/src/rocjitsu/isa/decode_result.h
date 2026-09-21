// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/result.h"
#include "util/diagnostic.h"

#include <memory>

namespace rocjitsu {

class Instruction;

using DecodeResult = FailureOr<std::unique_ptr<Instruction>>;
using DecodeErrorEmitter = util::DiagnosticEmitter;

} // namespace rocjitsu
