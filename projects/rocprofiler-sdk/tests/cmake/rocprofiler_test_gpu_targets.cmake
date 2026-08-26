include_guard(GLOBAL)

set(ROCPROFILER_DEFAULT_GPU_TARGETS
    "gfx900"
    "gfx906"
    "gfx908"
    "gfx90a"
    "gfx942"
    "gfx950"
    "gfx1030"
    "gfx1010"
    "gfx1100"
    "gfx1101"
    "gfx1102"
    "gfx1151"
    "gfx1152"
    "gfx1250")

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
