# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# ----------------------------------------------------------------------------------------#
#
# SQLite3 - built from the pre-generated amalgamation stored in this repository
#
# The amalgamation (sqlite3.c + sqlite3.h) is upstream's published release
# artifact, kept under `profilers/profiler-hub/external/sqlite3/`. Configure
# reads it straight off disk, so building SQLite3 needs no network access, no
# git clone, and - crucially - no `tclsh`. SQLite's own `make sqlite3.c` target
# runs `tool/mksqlite3c.tcl`, which `./configure --disable-tcl` does not lift.
#
# `external/sqlite3/README.md` is the maintenance entry point: it records the
# exact upstream release and checksums, and documents how to regenerate,
# verify and re-publish the amalgamation when SQLite is bumped.
#
# ----------------------------------------------------------------------------------------#

set(SQLITE3_AMALGAMATION_VERSION
    "3.45.3"
    CACHE STRING
    "Upstream SQLite3 release that the bundled amalgamation was taken from"
)

get_filename_component(
    _sqlite3_default_dir
    "${CMAKE_CURRENT_LIST_DIR}/../external/sqlite3"
    ABSOLUTE
)
set(SQLITE3_AMALGAMATION_DIR
    "${_sqlite3_default_dir}"
    CACHE PATH
    "Directory holding the pre-generated SQLite3 amalgamation (sqlite3.c and sqlite3.h)"
)
unset(_sqlite3_default_dir)

set(SQLITE3_AMALGAMATION_SHA256_C
    "9ca336fbcbff9f1d78b4f45b6a19583fcc097192310dd2f5f6cd43b9a33d7d69"
    CACHE STRING
    "Expected SHA256 of sqlite3.c; set to an empty string to skip verification"
)
set(SQLITE3_AMALGAMATION_SHA256_H
    "882ad3c0448d0324fb3a6b1a85333a9173d539ac669c9972ae1f03722ff86282"
    CACHE STRING
    "Expected SHA256 of sqlite3.h; set to an empty string to skip verification"
)

set(SQLITE3_SOURCE_DIR "${SQLITE3_AMALGAMATION_DIR}")
set(SQLITE3_AMALG_C "${SQLITE3_SOURCE_DIR}/sqlite3.c")
set(SQLITE3_AMALG_H "${SQLITE3_SOURCE_DIR}/sqlite3.h")

set(_sqlite3_readme "profilers/profiler-hub/external/sqlite3/README.md")

# ----------------------------------------------------------------------------------------#
# Presence and integrity checks
#
# Presence: the amalgamation is committed to git, so a normal checkout always
# has it. The two ways it can legitimately be absent are a partial/sparse
# checkout that excludes `external/`, and a tree where these sources have been
# moved to DVC (see the README). Name both, rather than letting the compiler
# report a missing input file.
#
# Integrity: guards against a truncated checkout, an accidental in-place edit,
# and against a stale artifact if these sources are ever served from a content
# store rather than from git.
# ----------------------------------------------------------------------------------------#

foreach(_sqlite3_kind IN ITEMS C H)
    set(_sqlite3_file "${SQLITE3_AMALG_${_sqlite3_kind}}")
    set(_sqlite3_sha_var "SQLITE3_AMALGAMATION_SHA256_${_sqlite3_kind}")

    if(NOT EXISTS "${_sqlite3_file}")
        if(EXISTS "${_sqlite3_file}.dvc")
            message(
                FATAL_ERROR
                "[profiler-hub] SQLite3 amalgamation is tracked by DVC but has not been "
                "materialised:\n"
                "    ${_sqlite3_file}\n"
                "Fetch it from the repository's DVC remote before configuring:\n"
                "    pip install 'dvc[s3]'\n"
                "    dvc pull ${_sqlite3_file}.dvc\n"
                "`dvc pull` must run from a checkout that contains the top-level "
                "`.dvc/` directory. See ${_sqlite3_readme}."
            )
        endif()
        message(
            FATAL_ERROR
            "[profiler-hub] SQLite3 amalgamation not found:\n"
            "    ${_sqlite3_file}\n"
            "This file is committed to the repository, so it is normally present. If "
            "this is a partial or sparse checkout, include "
            "`profilers/profiler-hub/external/sqlite3`. To build against an "
            "amalgamation held elsewhere, pass -DSQLITE3_AMALGAMATION_DIR=<dir> (that "
            "directory must contain both sqlite3.c and sqlite3.h). See "
            "${_sqlite3_readme}."
        )
    endif()

    if(${_sqlite3_sha_var})
        file(SHA256 "${_sqlite3_file}" _sqlite3_actual_sha256)
        if(NOT _sqlite3_actual_sha256 STREQUAL "${${_sqlite3_sha_var}}")
            message(
                FATAL_ERROR
                "[profiler-hub] SQLite3 amalgamation checksum mismatch:\n"
                "    file:     ${_sqlite3_file}\n"
                "    expected: ${${_sqlite3_sha_var}}\n"
                "    actual:   ${_sqlite3_actual_sha256}\n"
                "If you are deliberately supplying a different amalgamation via "
                "-DSQLITE3_AMALGAMATION_DIR, disable this check with "
                "-D${_sqlite3_sha_var}= . If you are bumping the bundled SQLite "
                "release, update ${_sqlite3_readme} and the checksums in "
                "profilers/profiler-hub/cmake/sqlite3.cmake together."
            )
        endif()
    endif()
endforeach()

message(
    STATUS
    "[profiler-hub] Using bundled SQLite3 ${SQLITE3_AMALGAMATION_VERSION} amalgamation: ${SQLITE3_SOURCE_DIR}"
)

add_library(profiler-hub-sqlite3-static STATIC ${SQLITE3_AMALG_C})

target_include_directories(
    profiler-hub-sqlite3-static
    PUBLIC $<BUILD_INTERFACE:${SQLITE3_SOURCE_DIR}>
)

target_compile_definitions(
    profiler-hub-sqlite3-static
    PRIVATE
        SQLITE_DEFAULT_MEMSTATUS=0
        SQLITE_THREADSAFE=1
        SQLITE_DEFAULT_WAL_SYNCHRONOUS=1
        SQLITE_LIKE_DOESNT_MATCH_BLOBS=1
        SQLITE_OMIT_DEPRECATED=1
        SQLITE_OMIT_PROGRESS_CALLBACK=1
        SQLITE_OMIT_SHARED_CACHE=1
)

# Seal the bundled SQLite symbols so they are not exported from
# libprofiler-hub.{so,a} and cannot collide with (or be interposed by)
# other sqlite3 versions bundled by sibling components on TheRock.
target_compile_options(
    profiler-hub-sqlite3-static
    PRIVATE -O2 -fPIC -fvisibility=hidden
)

set_target_properties(
    profiler-hub-sqlite3-static
    PROPERTIES POSITION_INDEPENDENT_CODE ON C_STANDARD 11
)

add_library(profiler-hub-sqlite3 INTERFACE)
target_link_libraries(
    profiler-hub-sqlite3
    INTERFACE profiler-hub-sqlite3-static ${CMAKE_DL_LIBS}
)
