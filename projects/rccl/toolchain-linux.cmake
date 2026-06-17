# toolchain-linux.cmake — Default toolchain for RCCL on Linux.
#
# Sets ROCM_PATH and AMD clang++/clang as the CXX and C compiler.
# Also sets CXX/C compiler flags for Release/Debug/RelWithDebInfo builds.
#
# Usage:
#   cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-linux.cmake ..
#
# The toolchain is auto-loaded by CMakeLists.txt if no toolchain file is specified.

# -----------------------------------------------------------------------------
# macro: rccl_detect_compilers
#
# Detects the ROCm compiler bin directory and sets CMAKE_CXX_COMPILER /
# CMAKE_C_COMPILER from the same location.
#
# Priority: -DCMAKE_CXX_COMPILER / $CXX > ROCm bin/amdclang++ > llvm/bin/amdclang++ > llvm/bin/clang++
# The C compiler is derived from the same directory (amdclang++→amdclang, clang++→clang).
# NOTE: Once written to cache, compilers are not re-detected on re-runs.
# To change, pass -DCMAKE_CXX_COMPILER / -DCMAKE_C_COMPILER or wipe the build directory.
# -----------------------------------------------------------------------------
macro(rccl_detect_compilers rocm_path)
    if(NOT CMAKE_CXX_COMPILER)
        if(DEFINED ENV{CXX} AND NOT "$ENV{CXX}" STREQUAL "")
            set(CMAKE_CXX_COMPILER "$ENV{CXX}" CACHE PATH "Path to C++ compiler")
        else()
            if(EXISTS "${rocm_path}/bin/amdclang++")
                set(_cxx "amdclang++")
                set(_cc  "amdclang")
                set(_bin "${rocm_path}/bin")
            elseif(EXISTS "${rocm_path}/llvm/bin/amdclang++")
                set(_cxx "amdclang++")
                set(_cc  "amdclang")
                set(_bin "${rocm_path}/llvm/bin")
            elseif(EXISTS "${rocm_path}/llvm/bin/clang++")
                set(_cxx "clang++")
                set(_cc  "clang")
                set(_bin "${rocm_path}/llvm/bin")
            else()
                message(FATAL_ERROR
                    "Cannot find amdclang++/clang++ under ${rocm_path}/bin or ${rocm_path}/llvm/bin.")
            endif()
            set(CMAKE_CXX_COMPILER "${_bin}/${_cxx}" CACHE PATH "Path to C++ compiler")
        endif()
    endif()

    if(NOT CMAKE_C_COMPILER)
        if(DEFINED ENV{CC} AND NOT "$ENV{CC}" STREQUAL "")
            set(CMAKE_C_COMPILER "$ENV{CC}" CACHE PATH "Path to C compiler")
        elseif(DEFINED _bin)
            set(CMAKE_C_COMPILER "${_bin}/${_cc}" CACHE PATH "Path to C compiler")
        else()
            if(EXISTS "${rocm_path}/bin/amdclang")
                set(CMAKE_C_COMPILER "${rocm_path}/bin/amdclang" CACHE PATH "Path to C compiler")
            elseif(EXISTS "${rocm_path}/llvm/bin/amdclang")
                set(CMAKE_C_COMPILER "${rocm_path}/llvm/bin/amdclang" CACHE PATH "Path to C compiler")
            elseif(EXISTS "${rocm_path}/llvm/bin/clang")
                set(CMAKE_C_COMPILER "${rocm_path}/llvm/bin/clang" CACHE PATH "Path to C compiler")
            else()
                message(FATAL_ERROR
                    "Cannot find amdclang/clang under ${rocm_path}/bin or ${rocm_path}/llvm/bin.")
            endif()
        endif()
    endif()

    unset(_cxx)
    unset(_cc)
    unset(_bin)
endmacro()

# -----------------------------------------------------------------------------
# macro: rccl_set_build_flags
#
# Sets default per-build-type flags for both CXX and C from a single definition.
# Skipped per-language if the user has set $CXXFLAGS/$CFLAGS or the per-type
# CMake variable explicitly (e.g. -DCMAKE_CXX_FLAGS_DEBUG=...).
#
# -O1 is used for debug builds as -O0 exceeds GPU scratch space for the full RCCL build.
# -O0 can be used for testing specific pain points if used in conjunction with ONLY_FUNCS.
# DebugFast drops -ggdb3 to speed up debug builds (shorter compile and link times).
# -----------------------------------------------------------------------------
macro(rccl_set_build_flags)
    if(CMAKE_BUILD_SUBTYPE MATCHES "DebugFast")
        set(_debug "-O1 -g")
    else()
        set(_debug "-O1 -g -ggdb3")
    endif()
    set(_release        "-O3")
    set(_relwithdebinfo "-O3 -g")

    if(NOT (DEFINED ENV{CXXFLAGS} AND NOT "$ENV{CXXFLAGS}" STREQUAL ""))
        if(NOT DEFINED CMAKE_CXX_FLAGS_DEBUG)
            set(CMAKE_CXX_FLAGS_DEBUG          "${_debug}")
        endif()
        if(NOT DEFINED CMAKE_CXX_FLAGS_RELEASE)
            set(CMAKE_CXX_FLAGS_RELEASE        "${_release}")
        endif()
        if(NOT DEFINED CMAKE_CXX_FLAGS_RELWITHDEBINFO)
            set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${_relwithdebinfo}")
        endif()
    endif()

    if(NOT (DEFINED ENV{CFLAGS} AND NOT "$ENV{CFLAGS}" STREQUAL ""))
        if(NOT DEFINED CMAKE_C_FLAGS_DEBUG)
            set(CMAKE_C_FLAGS_DEBUG            "${_debug}")
        endif()
        if(NOT DEFINED CMAKE_C_FLAGS_RELEASE)
            set(CMAKE_C_FLAGS_RELEASE          "${_release}")
        endif()
        if(NOT DEFINED CMAKE_C_FLAGS_RELWITHDEBINFO)
            set(CMAKE_C_FLAGS_RELWITHDEBINFO   "${_relwithdebinfo}")
        endif()
    endif()

    unset(_debug)
    unset(_release)
    unset(_relwithdebinfo)
endmacro()

# -----------------------------------------------------------------------------
# Detect ROCm installation.
# Priority: -DROCM_PATH > $ROCM_PATH env > PATH (via amdclang++/clang++) > /opt/rocm.
# NOTE: ROCM_PATH is written to the CMake cache on first configure. If you change the
# ROCm installation, pass -DROCM_PATH=<new_path> or wipe the build directory.
# -----------------------------------------------------------------------------

# 1. -DROCM_PATH or $ROCM_PATH env var.
if(NOT ROCM_PATH)
    if(DEFINED ENV{ROCM_PATH} AND NOT "$ENV{ROCM_PATH}" STREQUAL "")
        set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "Path to ROCm installation.")
    endif()
endif()

# 2. Derive from PATH: find amdclang++ or clang++ and walk up to the ROCm root.
#    Handles both ${ROCM_PATH}/bin/ and ${ROCM_PATH}/llvm/bin/ layouts.
if(NOT ROCM_PATH)
    find_program(_rocm_bin_hint NAMES amdclang++ clang++)
    if(_rocm_bin_hint)
        get_filename_component(_bin_dir "${_rocm_bin_hint}" DIRECTORY)
        get_filename_component(_parent  "${_bin_dir}"       DIRECTORY)
        if(EXISTS "${_parent}/lib/libamdhip64.so")
            set(ROCM_PATH "${_parent}" CACHE PATH "Path to ROCm installation (auto-detected from PATH).")
            message(STATUS "ROCM_PATH auto-detected from PATH: ${ROCM_PATH}")
        else()
            # llvm/bin layout: go one level higher
            get_filename_component(_grandparent "${_parent}" DIRECTORY)
            if(EXISTS "${_grandparent}/lib/libamdhip64.so")
                set(ROCM_PATH "${_grandparent}" CACHE PATH "Path to ROCm installation (auto-detected from PATH).")
                message(STATUS "ROCM_PATH auto-detected from PATH: ${ROCM_PATH}")
            endif()
        endif()
    endif()
    unset(_rocm_bin_hint CACHE)
    unset(_bin_dir)
    unset(_parent)
    unset(_grandparent)
endif()

# 3. Fall back to /opt/rocm.
if(NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm" CACHE PATH "Path to ROCm installation.")
endif()

# Finally, does ROCm exist?
if(NOT EXISTS "${ROCM_PATH}")
    message(FATAL_ERROR "ROCM_PATH=${ROCM_PATH} does not exist")
endif()

# Found ROCm, let's set compiler paths
rccl_detect_compilers("${ROCM_PATH}")

# Found ROCm and compilers, let's set compiler build flags
rccl_set_build_flags()
