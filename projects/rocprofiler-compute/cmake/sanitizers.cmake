## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

include_guard(GLOBAL)

# Fold THEROCK_SANITIZER and legacy ENABLE_ADDRESS_SANITIZER into the canonical
# ENABLE_SANITIZER, precedence THEROCK_SANITIZER > ENABLE_SANITIZER >
# ENABLE_ADDRESS_SANITIZER, so every downstream site reads one variable.
function(resolve_sanitizer)
    set(sanitizer_valid
        ""
        "OFF"
        "ASAN"
        "HOST_ASAN"
        "TSAN"
    )

    if(DEFINED THEROCK_SANITIZER AND NOT THEROCK_SANITIZER IN_LIST sanitizer_valid)
        message(
            FATAL_ERROR
            "THEROCK_SANITIZER='${THEROCK_SANITIZER}' is not one of: OFF, ASAN, HOST_ASAN, TSAN"
        )
    endif()

    set(sanitizer_provenance "-DENABLE_SANITIZER")
    if(DEFINED THEROCK_SANITIZER AND NOT THEROCK_SANITIZER STREQUAL "")
        set(ENABLE_SANITIZER
            "${THEROCK_SANITIZER}"
            CACHE STRING
            "Sanitizer for the native tool library (driven by THEROCK_SANITIZER)"
            FORCE
        )
        set(sanitizer_provenance "THEROCK_SANITIZER")
    elseif(
        (ENABLE_SANITIZER STREQUAL "" OR ENABLE_SANITIZER STREQUAL "OFF")
        AND ENABLE_ADDRESS_SANITIZER
    )
        # Promote ENABLE_ADDRESS_SANITIZER to ASAN so downstream only reads ENABLE_SANITIZER.
        set(ENABLE_SANITIZER
            "ASAN"
            CACHE STRING
            "Sanitizer for the native tool library (driven by ENABLE_ADDRESS_SANITIZER)"
            FORCE
        )
        set(sanitizer_provenance "ENABLE_ADDRESS_SANITIZER")
    endif()

    # Normalize OFF -> "" so downstream code only tests for emptiness.
    if(ENABLE_SANITIZER STREQUAL "OFF")
        set(ENABLE_SANITIZER
            ""
            CACHE STRING
            "Sanitizer for the native tool library: OFF, ASAN, HOST_ASAN, or TSAN"
            FORCE
        )
    endif()

    if(NOT ENABLE_SANITIZER IN_LIST sanitizer_valid)
        message(
            FATAL_ERROR
            "ENABLE_SANITIZER='${ENABLE_SANITIZER}' is not one of: OFF, ASAN, HOST_ASAN, TSAN"
        )
    endif()

    # Nuitka onefile is incompatible with sanitizers (it execs a stripped binary
    # from a temp dir; the sanitizer runtime cannot be located).
    if(ENABLE_SANITIZER AND STANDALONEBINARY)
        message(
            FATAL_ERROR
            "ENABLE_SANITIZER=${ENABLE_SANITIZER} cannot be combined with STANDALONEBINARY=ON"
        )
    endif()

    if(ENABLE_SANITIZER)
        message(STATUS "Sanitizer: ${ENABLE_SANITIZER} (from ${sanitizer_provenance})")
    else()
        message(STATUS "Sanitizer: OFF")
    endif()

    set(ENABLE_SANITIZER "${ENABLE_SANITIZER}" PARENT_SCOPE)
endfunction()

# Apply -fsanitize=... compile flags and link options to the current scope.
# Compile-flag injection is skipped when TheRock already populated -fsanitize= via
# CMAKE_CXX_FLAGS_INIT (avoid double-instrumentation); link options are emitted
# unconditionally and are idempotent, so duplication is benign.
function(enable_sanitizer)
    if(NOT ENABLE_SANITIZER)
        return()
    endif()

    if(ENABLE_SANITIZER STREQUAL "ASAN" OR ENABLE_SANITIZER STREQUAL "HOST_ASAN")
        set(_flag "address")
    elseif(ENABLE_SANITIZER STREQUAL "TSAN")
        set(_flag "thread")
    endif()

    if(CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=")
        message(
            STATUS
            "enable_sanitizer(): -fsanitize= already in CMAKE_CXX_FLAGS_INIT; skipping local compile-flag injection"
        )
    else()
        set(_extra "-fsanitize=${_flag} -fno-omit-frame-pointer -g")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_extra}" PARENT_SCOPE)
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_extra}" PARENT_SCOPE)
    endif()

    # clang defaults to static sanitizer linkage; gcc defaults to shared.
    # Force shared on clang only.
    add_link_options(
        $<$<LINK_LANGUAGE:C,CXX>:-fsanitize=${_flag}>
        $<$<AND:$<LINK_LANGUAGE:C,CXX>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>
    )
endfunction()

# Rewrite GPU_TARGETS for full ASAN and TSAN modes (gfx942/gfx950 -> :xnack+).
# No-op for HOST_ASAN or when TheRock has already rewritten the targets upstream.
function(enable_sanitizer_gpu_target_munging)
    if(NOT (ENABLE_SANITIZER STREQUAL "ASAN" OR ENABLE_SANITIZER STREQUAL "TSAN"))
        return()
    endif()
    if(NOT DEFINED GPU_TARGETS)
        message(
            WARNING
            "${ENABLE_SANITIZER}: GPU_TARGETS is not set; skipping the gfx942/gfx950 -> :xnack+ rewrite. Pass -DGPU_TARGETS=... for device-side sanitizer instrumentation."
        )
        return()
    endif()
    # The gfx942/gfx950 arch list mirrors TheRock's regex, tracked upstream as
    # TheRock TODO #3444 (ASAN variants may need xnack-suffix expansion). Keep it in
    # sync with upstream rather than widening it independently.
    list(TRANSFORM GPU_TARGETS REPLACE "^(gfx942|gfx950)$" "\\1:xnack+")
    set(GPU_TARGETS "${GPU_TARGETS}" PARENT_SCOPE)
    set(AMDGPU_TARGETS "${GPU_TARGETS}" PARENT_SCOPE)
    message(STATUS "${ENABLE_SANITIZER}: GPU_TARGETS rewritten -> ${GPU_TARGETS}")
endfunction()

# Wrap the ctest python command with THEROCK_SANITIZER_LAUNCHER plus env that
# quiets known false positives.
function(enable_sanitizer_python_launcher out_var)
    set(_launcher ${THEROCK_SANITIZER_LAUNCHER} ${${out_var}})
    if(ENABLE_SANITIZER STREQUAL "ASAN" OR ENABLE_SANITIZER STREQUAL "HOST_ASAN")
        list(
            PREPEND
            _launcher
            "${CMAKE_COMMAND}"
            -E
            env
            "ASAN_OPTIONS=detect_leaks=0"
            --
        )
    elseif(ENABLE_SANITIZER STREQUAL "TSAN")
        list(
            PREPEND
            _launcher
            "${CMAKE_COMMAND}"
            -E
            env
            "TSAN_OPTIONS=second_deadlock_stack=1"
            --
        )
    endif()
    set(${out_var} "${_launcher}" PARENT_SCOPE)
endfunction()

set(ENABLE_SANITIZER
    "OFF"
    CACHE STRING
    "Sanitizer for the native tool library: OFF, ASAN, HOST_ASAN, or TSAN"
)
set_property(
    CACHE ENABLE_SANITIZER
    PROPERTY STRINGS OFF ASAN HOST_ASAN TSAN
)

# Drive all three side effects at include time: resolve the selection, rewrite GPU
# targets, inject compile/link flags. CMAKE_CXX_FLAGS propagates into src/lib via
# subdir inheritance, so subdirs need no sanitizer awareness. The runtime JIT build
# of src/lib does not include this module, so it stays sanitizer-free. The python
# launcher is wired separately once PYTHON_TEST_COMMAND is defined.
resolve_sanitizer()
enable_sanitizer_gpu_target_munging()
enable_sanitizer()
