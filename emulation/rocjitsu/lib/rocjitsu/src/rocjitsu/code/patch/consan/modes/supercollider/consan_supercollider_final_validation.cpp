// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_final_validation_internal.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_perturbation_policy.h"
#include "rocjitsu/code/patch/consan/consan_sync_event_index.h"
#include "rocjitsu/code/patch/rdna4_instrumentation_builder.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <unordered_set>

namespace rocjitsu::consan_validation_detail {

#include "consan_supercollider_final_validation.inc"

} // namespace rocjitsu::consan_validation_detail
