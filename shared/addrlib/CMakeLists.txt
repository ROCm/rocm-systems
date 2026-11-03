# Copyright Advanced Micro Devices, Inc. SPDX-License-Identifier: MIT

# AMD address library (addrlib): surface addressing, tiling and swizzle math.
#
# Consumed with add_subdirectory() rather than find_package(), for the same reason
# as shared/sha256: it depends on nothing but the standard library, so it can be
# pulled in from either direction without creating a cycle.
#
# This tree is a vendored upstream drop. Keep it that way: do not fix warnings or
# style here, or every future re-sync has to re-apply the patches. Warnings are
# suppressed below instead, which is how CRUX already handles the vendored Khronos
# headers (projects/crux/CMakeLists.txt).

cmake_minimum_required(VERSION 3.16)
project(addrlib VERSION 1.0.0 LANGUAGES CXX)

# Namespace the library is compiled into.
#
# Consumers call addrlib with unqualified names from inside their own namespace,
# so the library has to land in whichever namespace the caller lives in. ROCr and
# rocaddr3 use the default; CRUX configures amd::crux::hsa. Verified to produce
# byte-identical code either way.
set(ADDRLIB_NAMESPACE "rocr" CACHE STRING
    "Namespace enclosing addrlib (e.g. rocr, amd::crux::hsa)")

set(ADDRLIB_SOURCES
    src/addrinterface.cpp
    src/core/coord.cpp
    src/core/addrlib.cpp
    src/core/addrlib1.cpp
    src/core/addrlib2.cpp
    src/core/addrlib3.cpp
    src/core/addrobject.cpp
    src/core/addrelemlib.cpp
    src/gfx9/gfx9addrlib.cpp
    src/gfx10/gfx10addrlib.cpp
    src/gfx11/gfx11addrlib.cpp
    src/gfx12/gfx12addrlib.cpp)

add_library(addrlib-objects OBJECT ${ADDRLIB_SOURCES})

# Public: consumers that include addrinterface.h must agree with the library on
# the namespace, or the declarations they see will not match the definitions.
target_compile_definitions(addrlib-objects PUBLIC
    ADDRLIB_NAMESPACE=${ADDRLIB_NAMESPACE})

# Required by src/chip/*/\*_gb_reg.h, which #errors without an endianness macro.
target_compile_definitions(addrlib-objects PRIVATE LITTLEENDIAN_CPU)

# All of these are PUBLIC, including the internal src/ directories, because
# consumers legitimately reach past the public header: ROCr's image manager
# includes src/core/addrlib.h and src/amdgpu_asic_addr.h directly. The parent
# directory is exported too, so existing "addrlib/..."-prefixed includes keep
# resolving without touching consumer sources.
#
# SYSTEM so consumers compiling with -Werror are not broken by upstream warnings.
target_include_directories(addrlib-objects SYSTEM PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/inc>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/core>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/r800>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gfx9>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gfx10>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gfx11>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/gfx12>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/chip/r800>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/chip/gfx9>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/chip/gfx10>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/chip/gfx11>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/chip/gfx12>)

# Vendored code: silence warnings on its own sources rather than patching it.
# These come after any directory-level -Wall/-Werror, so they win.
if(MSVC)
    target_compile_options(addrlib-objects PRIVATE /w)
else()
    target_compile_options(addrlib-objects PRIVATE -w)
endif()

target_compile_features(addrlib-objects PUBLIC cxx_std_17)

# Linked into shared libraries (ROCr's hsa-runtime, CRUX's libamdhip64).
set_target_properties(addrlib-objects PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON)

add_library(addrlib STATIC)
add_library(rocm::addrlib ALIAS addrlib)

target_link_libraries(addrlib PUBLIC addrlib-objects)
