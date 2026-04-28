// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Single-stop include for code that calls services::sampling() or
// services::causal_sampling(). Makes default_sampling_service a complete type
// so its methods can be called.
//
// Policy definitions are NOT included here. The Meyers singleton in
// library/services_accessor.cpp includes library/sampling_production_policies.hpp
// (main-lib TU only). Test binaries supply their own policy stubs independently.

#include "sampling/sampling_service.hpp"
