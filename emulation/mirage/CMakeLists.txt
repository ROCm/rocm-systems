# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# TRANSITIONAL. This directory holds no project.
#
# The CLI that used to live here is now built by the rocjitsu subproject,
# from `emulation/rocjitsu/cli`, and installs as `bin/rocjitsu` with a
# `bin/mirage` symlink beside it. This file exists only because TheRock
# still declares a `mirage` subproject rooted here, and ninja needs the
# `CMakeLists.txt` its configure stamp depends on. Delete this directory
# once the TheRock pin moves past the revision that drops that
# declaration.
#
# Nothing is installed. The `mirage` artifact stages from
# `emulation/mirage/stage`, which is a different root from the rocjitsu
# artifact's, so a relative symlink to `rocjitsu` would dangle inside it
# and a wrapper that execs `rocjitsu` would find nothing on PATH.
# An empty artifact is the honest answer while the CLI ships from
# rocjitsu.
cmake_minimum_required(VERSION 3.20)
project(mirage NONE)

# TheRock passes this. Accepted so the configure step does not warn about
# an unused cache entry; the profile it names is the rocjitsu subproject's
# to act on.
set(MIRAGE_CARGO_PROFILE
    "release"
    CACHE STRING
    "Accepted for TheRock compatibility; this project builds nothing"
)

# `therock_cmake_subproject_build_test` runs ctest with `--no-tests=error`.
# The Linux package build never reaches that target, but a job that builds
# `therock-build-tests` would, and an empty suite is an error there.
enable_testing()
add_test(NAME mirage_shim_ok COMMAND ${CMAKE_COMMAND} -E true)
