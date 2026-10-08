// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_COMMON_H_
#define ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_COMMON_H_

#include "rocjitsu/isa/arch/amdgpu/shared/accvgpr_layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/fp_mode.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx11_dot2.h"
#include "rocjitsu/isa/arch/amdgpu/shared/gfx12_dot.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/arithmetic.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/extract.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/layout.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/register_access.h"
#include "rocjitsu/isa/arch/amdgpu/shared/mma/staging_limits.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/register_access.h"
#include "util/data_types.h"
#include "util/except.h"
#include "util/meta_programming.h"
#include "util/simd.h"

#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#endif // ROCJITSU_ISA_ARCH_AMDGPU_SHARED_MMA_COMMON_H_
