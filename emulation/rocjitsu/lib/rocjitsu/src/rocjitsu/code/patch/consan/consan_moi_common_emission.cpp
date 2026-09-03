// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_common_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_probe_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_prologue.h"
#include "rocjitsu/code/patch/consan/consan_moi_record_planning.h"
#include "rocjitsu/code/patch/consan/consan_moi_relocation.h"
#include "rocjitsu/code/patch/consan/consan_moi_shared_lowering.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <utility>

namespace rocjitsu {

using consan_detail::MoiSpecialStateSgprs;
using consan_moi_detail::append_moi_scc_preserving_indirect_jump;
using consan_moi_detail::append_words_bytes;

namespace consan_moi_impl {

#include "rocjitsu/code/patch/consan/consan_moi_common_emission.inc"

} // namespace consan_moi_impl
} // namespace rocjitsu
