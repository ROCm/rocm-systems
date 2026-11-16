// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/analysis/analyze.h"

namespace rocjitsu::consan {

std::string render_analysis(const LdsRaceResult &result) {
  return render_lds_race_analysis(result);
}

} // namespace rocjitsu::consan
