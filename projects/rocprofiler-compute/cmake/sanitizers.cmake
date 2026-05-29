include_guard(GLOBAL)

# Resolve the sanitizer selection in place (normalize, validate, write back via
# PARENT_SCOPE), mirroring the PYTHON_TEST_COMMAND pattern. THEROCK_SANITIZER is
# promoted over the passed variable so TheRock-driven builds have a single source
# of truth.
function(resolve_sanitizer out_var)
    set(_san_valid
        ""
        "OFF"
        "ASAN"
        "HOST_ASAN"
        "TSAN"
    )

    # Validated before promotion so the error names THEROCK_SANITIZER, not ${out_var}.
    if(DEFINED THEROCK_SANITIZER AND NOT THEROCK_SANITIZER IN_LIST _san_valid)
        message(
            FATAL_ERROR
            "THEROCK_SANITIZER='${THEROCK_SANITIZER}' is not one of: OFF, ASAN, HOST_ASAN, TSAN"
        )
    endif()

    set(_san_provenance "${out_var}")
    if(DEFINED THEROCK_SANITIZER AND NOT THEROCK_SANITIZER STREQUAL "")
        set(${out_var}
            "${THEROCK_SANITIZER}"
            CACHE STRING
            "Sanitizer for the native tool library (driven by THEROCK_SANITIZER)"
            FORCE
        )
        set(_san_provenance "THEROCK_SANITIZER")
    endif()

    # Normalize OFF -> "" so downstream code only tests for emptiness.
    if(${out_var} STREQUAL "OFF")
        set(${out_var} "" CACHE STRING "" FORCE)
    endif()

    if(NOT ${out_var} IN_LIST _san_valid)
        message(
            FATAL_ERROR
            "${out_var}='${${out_var}}' is not one of: OFF, ASAN, HOST_ASAN, TSAN"
        )
    endif()

    # Nuitka onefile is incompatible with sanitizers (it execs a stripped binary
    # from a temp dir; the sanitizer runtime cannot be located).
    if(${out_var} AND STANDALONEBINARY)
        message(
            FATAL_ERROR
            "${out_var}=${${out_var}} cannot be combined with STANDALONEBINARY=ON"
        )
    endif()

    if(${out_var})
        message(STATUS "Sanitizer: ${${out_var}} (from ${_san_provenance})")
    else()
        message(STATUS "Sanitizer: OFF")
    endif()

    set(${out_var} "${${out_var}}" PARENT_SCOPE)
endfunction()

# Apply -fsanitize=... flags and link options to the current scope (call from
# src/lib/CMakeLists.txt). No-op when off, or when TheRock already injected
# -fsanitize= via CMAKE_CXX_FLAGS_INIT (avoid double-instrumentation).
function(enable_sanitizer)
    if(NOT ENABLE_SANITIZER)
        return()
    endif()

    if(CMAKE_CXX_FLAGS_INIT MATCHES "-fsanitize=")
        message(
            STATUS
            "enable_sanitizer(): -fsanitize= already in CMAKE_CXX_FLAGS_INIT; skipping local injection"
        )
        return()
    endif()

    if(ENABLE_SANITIZER STREQUAL "ASAN" OR ENABLE_SANITIZER STREQUAL "HOST_ASAN")
        set(_flag "address")
    elseif(ENABLE_SANITIZER STREQUAL "TSAN")
        set(_flag "thread")
    endif()

    set(_extra "-fsanitize=${_flag} -fno-omit-frame-pointer -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${_extra}" PARENT_SCOPE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${_extra}" PARENT_SCOPE)

    # clang defaults to static sanitizer linkage; gcc defaults to shared.
    # Force shared on clang only.
    add_link_options(
        $<$<LINK_LANGUAGE:C,CXX>:-fsanitize=${_flag}>
        $<$<AND:$<LINK_LANGUAGE:C,CXX>,$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>>>:-shared-libsan>
    )
endfunction()

# Rewrite GPU_TARGETS for full ASAN and TSAN modes (gfx942/gfx950 -> :xnack+).
# Mirrors TheRock/cmake/therock_sanitizers.cmake. No-op for HOST_ASAN or when
# TheRock has already rewritten the targets upstream.
function(enable_sanitizer_gpu_target_munging)
    if(NOT (ENABLE_SANITIZER STREQUAL "ASAN" OR ENABLE_SANITIZER STREQUAL "TSAN"))
        return()
    endif()
    if(NOT DEFINED GPU_TARGETS)
        return()
    endif()
    list(TRANSFORM GPU_TARGETS REPLACE "^(gfx942|gfx950)$" "\\1:xnack+")
    set(GPU_TARGETS "${GPU_TARGETS}" PARENT_SCOPE)
    set(AMDGPU_TARGETS "${GPU_TARGETS}" PARENT_SCOPE)
    message(STATUS "${ENABLE_SANITIZER}: GPU_TARGETS rewritten -> ${GPU_TARGETS}")
endfunction()

# Wrap the ctest python command with THEROCK_SANITIZER_LAUNCHER plus env that
# quiets known false positives (python intentionally leaks on exit -> detect_leaks=0).
function(enable_sanitizer_python_launcher out_var)
    if(NOT DEFINED THEROCK_SANITIZER_LAUNCHER)
        set(THEROCK_SANITIZER_LAUNCHER)
    endif()
    set(_launcher ${THEROCK_SANITIZER_LAUNCHER} ${PYTHON_TEST_COMMAND})
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
resolve_sanitizer(ENABLE_SANITIZER)
