include_guard(GLOBAL)

include("${CMAKE_CURRENT_LIST_DIR}/rocprofiler_default_gpu_targets.cmake")

if(NOT GPU_TARGETS)
    set(GPU_TARGETS
        "${ROCPROFILER_DEFAULT_GPU_TARGETS}"
        CACHE STRING "GPU targets to compile for" FORCE)
endif()

set(AMDGPU_TARGETS
    "${GPU_TARGETS}"
    CACHE STRING
          "GPU targets to compile for AMDGPUs (update GPU_TARGETS, not this variable)"
          FORCE)

# Every sample/test that declares its own HIP kernels does so via a nested project(...
# LANGUAGES ... HIP) call (CMake's native HIP language support), rather than the
# GPU_TARGETS/AMDGPU_TARGETS-driven hipcc invocation the main library uses. That language
# support reads CMAKE_HIP_ARCHITECTURES, which is otherwise never pointed at GPU_TARGETS,
# so it falls back to CMake's own default detection. That default can silently pick an
# architecture the running agent doesn't match (observed: it landed on gfx1200 on a
# gfx1250-strict system), producing a code object the HIP runtime refuses to load
# (hipErrorInvalidImage / "device kernel image is invalid"). Force it here, before any of
# those nested project() calls run, so every one of them inherits the same target; a plain
# `if(NOT CMAKE_HIP_ARCHITECTURES)` guard isn't enough because CMake's own HIP language
# support already writes its wrong default into the cache the first time it runs.
set(CMAKE_HIP_ARCHITECTURES
    "${GPU_TARGETS}"
    CACHE STRING "GPU architectures for CMake's native HIP language support" FORCE)
