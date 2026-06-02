# Hotswap.cmake — build HotSwap from source as part of the mirage build.
#
# Gated behind MIRAGE_BUILD_HOTSWAP (OFF by default; this is a full LLVM +
# COMGR + ROCR build and takes a long time). Included from the top-level
# CMakeLists.txt only when the flag is ON.
#
# HotSwap is NOT a single `libhsa-hotswap.so`. It is the set of artifacts the
# reference Docker recipe (huggingface_ut_hotswap.ubuntu.amd.Dockerfile)
# produces, staged into a single tree that mirage discovers and wires into the
# workload via the HotSwap env contract:
#
#   ${MIRAGE_HOTSWAP_STAGE}/
#     lib/        libamd_comgr.so*        (COMGR transpiler)
#                 libhsa-runtime64.so*    (HotSwap-patched ROCR)
#                 libhotswap_intercept.so (HIP intercept, LD_PRELOADed)
#     llvm-tools/ lld ld.lld llc llvm-mc  (the transpiler shells out to these)
#     runtime/hotswap_py/                 (python adapter runtime)
#
# Three independent-ish build stages mirror the Dockerfile:
#   1. hotswap_comgr  — llvm-project fork: amd_comgr + clang/lld/llc/llvm-mc.
#   2. hotswap_rocr   — rocm-systems fork: libhsa-runtime64 (needs stage 1).
#   3. hotswap_intercept — testing repo: the HIP intercept + python runtime.
#
# Private forks: clone over HTTPS using whatever credentials git is configured
# with (e.g. a token in the URL, a credential helper, or an SSH rewrite). To
# avoid re-cloning the huge llvm-project, the in-tree `llvm-project-hotswap`
# checkout is used as the COMGR source by default.

include(ExternalProject)

# The top-level project is declared `project(mirage NONE)`; the intercept stage
# needs a C++ compiler, so enable CXX here (only reached when the flag is ON).
enable_language(CXX)

# ---- knobs ------------------------------------------------------------------
# Stage into the workspace Rust target dir so the lib lands in `target/lib`,
# which mirage discovers automatically (it searches `../lib` relative to the
# `target/<profile>/mirage` binary). The LLVM tools and python runtime land in
# the sibling `target/llvm-tools` and `target/runtime/hotswap_py`.
set(MIRAGE_HOTSWAP_STAGE "${CMAKE_CURRENT_SOURCE_DIR}/target"
    CACHE PATH "Directory the built HotSwap artifacts are staged into")

# Stage 1: COMGR transpiler (llvm-project fork). Default to the in-tree
# checkout so we don't re-download ~GBs of llvm-project; fall back to a clone.
set(MIRAGE_HOTSWAP_LLVM_SRC "${CMAKE_CURRENT_SOURCE_DIR}/llvm-project-hotswap"
    CACHE PATH "Existing llvm-project (HotSwap fork) checkout to build COMGR from")
set(MIRAGE_HOTSWAP_LLVM_REPO "https://github.com/martin-luecke/llvm-project"
    CACHE STRING "llvm-project (HotSwap fork) git URL, used when the checkout is absent")
set(MIRAGE_HOTSWAP_LLVM_REF "users/mluecke/hotswap-landing-parity"
    CACHE STRING "llvm-project (HotSwap fork) git ref")

# Stage 2: HotSwap-patched ROCR (rocm-systems fork).
set(MIRAGE_HOTSWAP_ROCR_REPO "https://github.com/martin-luecke/rocm-systems"
    CACHE STRING "rocm-systems (HotSwap fork) git URL")
set(MIRAGE_HOTSWAP_ROCR_REF "users/mluecke/hotswap-compatibility"
    CACHE STRING "rocm-systems (HotSwap fork) git ref")

# Stage 3: HIP intercept + python runtime (testing repo).
set(MIRAGE_HOTSWAP_TESTING_REPO "https://github.com/harsh-amd/rocm-hotswap-testing"
    CACHE STRING "HotSwap testing repo git URL (HIP intercept + python runtime)")
set(MIRAGE_HOTSWAP_TESTING_REF "mluecke/hotswap-env-contract"
    CACHE STRING "HotSwap testing repo git ref")

# ROCm install used to satisfy ROCR's find_package() calls (hsakmt, libdrm...).
set(MIRAGE_HOTSWAP_ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH "ROCm install prefix for the ROCR build")
if(NOT MIRAGE_HOTSWAP_ROCM_PATH)
  set(MIRAGE_HOTSWAP_ROCM_PATH "/opt/rocm")
endif()

include(ProcessorCount)
ProcessorCount(_hs_nproc)
if(_hs_nproc EQUAL 0)
  set(_hs_nproc 4)
endif()
set(MIRAGE_HOTSWAP_JOBS "${_hs_nproc}" CACHE STRING "Parallel jobs for the HotSwap source build")

set(_hs_stage   "${MIRAGE_HOTSWAP_STAGE}")
set(_hs_libdir  "${_hs_stage}/lib")
set(_hs_tooldir "${_hs_stage}/llvm-tools")
set(_hs_pydir   "${_hs_stage}/runtime/hotswap_py")
set(_hs_copy    "${CMAKE_CURRENT_LIST_DIR}/hotswap_copy_glob.cmake")

message(STATUS "mirage: HotSwap source build ENABLED")
message(STATUS "mirage:   stage dir   ${_hs_stage}")
message(STATUS "mirage:   build jobs  ${MIRAGE_HOTSWAP_JOBS}")
message(STATUS "mirage:   ROCm path   ${MIRAGE_HOTSWAP_ROCM_PATH}")

# ---- stage 1: COMGR transpiler + LLVM tools ---------------------------------
# Build amd_comgr from the llvm-project fork with the HotSwap transpiler folded
# in, plus the LLVM tools the transpiler shells out to at runtime. The COMGR
# build is told the transpiler will find those tools under ${_hs_tooldir}, which
# is where we stage them below.
set(_hs_comgr_build "${CMAKE_CURRENT_BINARY_DIR}/hotswap-comgr-build")

set(_hs_comgr_src_args)
if(EXISTS "${MIRAGE_HOTSWAP_LLVM_SRC}/llvm/CMakeLists.txt")
  message(STATUS "mirage:   COMGR source (in-tree) ${MIRAGE_HOTSWAP_LLVM_SRC}")
  set(_hs_llvm_src "${MIRAGE_HOTSWAP_LLVM_SRC}")
  set(_hs_comgr_src_args DOWNLOAD_COMMAND "")
else()
  message(STATUS "mirage:   COMGR source (clone)   ${MIRAGE_HOTSWAP_LLVM_REPO}@${MIRAGE_HOTSWAP_LLVM_REF}")
  set(_hs_llvm_src "${CMAKE_CURRENT_BINARY_DIR}/hotswap-llvm-src")
  set(_hs_comgr_src_args
    GIT_REPOSITORY "${MIRAGE_HOTSWAP_LLVM_REPO}"
    GIT_TAG        "${MIRAGE_HOTSWAP_LLVM_REF}"
    GIT_SHALLOW    TRUE)
endif()

ExternalProject_Add(hotswap_comgr
  ${_hs_comgr_src_args}
  SOURCE_DIR        "${_hs_llvm_src}"
  SOURCE_SUBDIR     llvm
  BINARY_DIR        "${_hs_comgr_build}"
  CMAKE_GENERATOR   Ninja
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DLLVM_ENABLE_PROJECTS=clang$<SEMICOLON>lld
    -DLLVM_TARGETS_TO_BUILD=X86$<SEMICOLON>AMDGPU
    -DLLVM_EXTERNAL_PROJECTS=device-libs$<SEMICOLON>comgr
    -DLLVM_EXTERNAL_DEVICE_LIBS_SOURCE_DIR=${_hs_llvm_src}/amd/device-libs
    -DLLVM_EXTERNAL_COMGR_SOURCE_DIR=${_hs_llvm_src}/amd/comgr
    -DCOMGR_ENABLE_HOTSWAP_TRANSPILE=ON
    -DHOTSWAP_TRANSPILER_LLVM_TOOLS_DIR=${_hs_tooldir}
    -DLLVM_USE_LINKER=lld
    -DBUILD_TESTING=OFF
    -DLLVM_APPEND_VC_REV=OFF
  # Generate the embedded device-libs headers first, else a parallel race:
  # comgr-device-libs.cpp compiles before the oclc_*.inc are generated.
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${MIRAGE_HOTSWAP_JOBS}
      --target libraries_sha_header opencl-c-base.inc_target
  COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${MIRAGE_HOTSWAP_JOBS}
      --target amd_comgr
  # clang/llvm-objcopy: ROCR trap handler. llc/llvm-mc: the transpiler shells
  # out to them at runtime. lld so the matching ld.lld is available.
  COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${MIRAGE_HOTSWAP_JOBS}
      --target clang llvm-objcopy lld llc llvm-mc
  # Stage the COMGR runtime library and the LLVM tools into the HotSwap tree.
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -DHS_ROOT=<BINARY_DIR>/lib -DHS_GLOB=libamd_comgr.so*
      -DHS_DST=${_hs_libdir} -P ${_hs_copy}
  COMMAND
    ${CMAKE_COMMAND} -DHS_ROOT=<BINARY_DIR>/bin "-DHS_GLOB=lld$<SEMICOLON>ld.lld$<SEMICOLON>llc$<SEMICOLON>llvm-mc"
      -DHS_DST=${_hs_tooldir} -P ${_hs_copy}
  USES_TERMINAL_BUILD   TRUE
  USES_TERMINAL_INSTALL TRUE
)

# ---- stage 2: HotSwap-patched ROCR ------------------------------------------
# Links the COMGR adapter, exports rocr_hotswap_patch_elf, and produces the
# patched libhsa-runtime64. amd_comgr_DIR points find_package(amd_comgr) at the
# stage-1 build tree; CMAKE_PREFIX_PATH still supplies ROCm + the LLVM build
# (clang/llvm-objcopy for the trap handler).
set(_hs_rocr_build "${CMAKE_CURRENT_BINARY_DIR}/hotswap-rocr-build")

ExternalProject_Add(hotswap_rocr
  DEPENDS           hotswap_comgr
  GIT_REPOSITORY    "${MIRAGE_HOTSWAP_ROCR_REPO}"
  GIT_TAG           "${MIRAGE_HOTSWAP_ROCR_REF}"
  GIT_SHALLOW       TRUE
  SOURCE_DIR        "${CMAKE_CURRENT_BINARY_DIR}/hotswap-rocr-src"
  SOURCE_SUBDIR     projects/rocr-runtime
  BINARY_DIR        "${_hs_rocr_build}"
  CMAKE_GENERATOR   Ninja
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${MIRAGE_HOTSWAP_ROCM_PATH}
    "-DCMAKE_PREFIX_PATH=${MIRAGE_HOTSWAP_ROCM_PATH}$<SEMICOLON>${_hs_comgr_build}"
    -Damd_comgr_DIR=${_hs_comgr_build}/tools/comgr/lib/cmake/amd_comgr
    -DBUILD_SHARED_LIBS=ON
    -DIMAGE_SUPPORT=ON
    -DROCR_ENABLE_HOTSWAP_COMGR_ADAPTER=ON
  BUILD_COMMAND
    ${CMAKE_COMMAND} --build <BINARY_DIR> --parallel ${MIRAGE_HOTSWAP_JOBS}
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -DHS_ROOT=<BINARY_DIR> -DHS_RECURSE_ANCHOR=libhsa-runtime64.so.1
      -DHS_GLOB=libhsa-runtime64.so* -DHS_DST=${_hs_libdir} -P ${_hs_copy}
  USES_TERMINAL_BUILD   TRUE
  USES_TERMINAL_INSTALL TRUE
)

# ---- stage 3: HIP intercept + python runtime --------------------------------
# A single translation unit built with the system g++; the python runtime is
# copied verbatim. Independent of stages 1/2.
set(_hs_testing_src "${CMAKE_CURRENT_BINARY_DIR}/hotswap-testing-src")

ExternalProject_Add(hotswap_intercept
  GIT_REPOSITORY    "${MIRAGE_HOTSWAP_TESTING_REPO}"
  GIT_TAG           "${MIRAGE_HOTSWAP_TESTING_REF}"
  GIT_SHALLOW       TRUE
  SOURCE_DIR        "${_hs_testing_src}"
  CONFIGURE_COMMAND ${CMAKE_COMMAND} -E make_directory ${_hs_libdir}
  BUILD_COMMAND
    ${CMAKE_CXX_COMPILER} -shared -fPIC -std=c++17 -O2
      -o ${_hs_libdir}/libhotswap_intercept.so
      <SOURCE_DIR>/runtime/tools/hotswap_intercept.cpp -ldl
  INSTALL_COMMAND
    ${CMAKE_COMMAND} -E copy_directory <SOURCE_DIR>/runtime/hotswap_py ${_hs_pydir}
  USES_TERMINAL_BUILD TRUE
)

# Convenience aggregate so `cmake --build build --target hotswap` builds all of
# it, and so it participates in the default ALL build when the flag is ON.
add_custom_target(hotswap ALL
  DEPENDS hotswap_comgr hotswap_rocr hotswap_intercept
  COMMENT "HotSwap built and staged under ${_hs_stage}")
