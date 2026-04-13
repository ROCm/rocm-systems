# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Race detection library, copied from the race-emulator project
# (rocm-libraries/shared/race-emulator). The authoritative source is
# race-emulator; changes should be made there first, then copied here.
# When rocjitsu's design suggests a better interface, we reshape
# race-emulator to match rather than adding compatibility shims here.

add_library(race_detector OBJECT
    src/RaceDetector.cpp
    src/WaveRaceState.cpp
)
target_include_directories(race_detector PUBLIC include)
target_compile_features(race_detector PUBLIC cxx_std_20)
