# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Cycle-accurate model library. Built as part of the rocjitsu tree alongside the
# other plugin libs (added from emulation/rocjitsu/CMakeLists.txt, next to
# race-detector). Free of rocjitsu *types* — the rocjitsu-side adapter
# (emulation/rocjitsu/plugins/cycle_model_plugin.*) links against this and is the
# only place that touches rocjitsu types.
#
# nlohmann/json is vendored as a single header under third_party/ and included
# directly, so the lib needs no external json target (rocjitsu's own config path
# is FlatBuffers, not nlohmann json).

add_library(cycle_model
  src/uarch_config.cpp     # JSON load + validate.
  src/arch_model.cpp       # scheduler bodies (tick_cycle, scheduler_step).
  src/memory_system.cpp    # memory submodels (L1/L2/MSHR/HBM/LDS).
  src/shared_mem_model.cpp # shared L2 + HBM timing (owned by MemSysCycleModel, one per XCD).
  src/coalescer.cpp        # per-lane address stream -> cache-line transactions.
)
target_include_directories(cycle_model
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/third_party)   # vendored <nlohmann/json.hpp>
target_compile_features(cycle_model PUBLIC cxx_std_20)
# TimedTagCache (CacheModels.h) layers a fill-readiness overlay on top of
# simdojo's RuntimeSetAssocTags. Link simdojo for its PUBLIC include path only;
# the lib stays free of rocjitsu types (the adapter is the only coupling).
target_link_libraries(cycle_model PUBLIC simdojo)

# Header-only state structs are exercised purely through the library translation
# units; no rocjitsu link dependency exists here by design.

# Tests build with the tree's gtest (BUILD_TESTING is set by the parent build).
if(BUILD_TESTING)
  add_subdirectory(tests)
endif()
