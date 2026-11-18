include_guard(GLOBAL)

# The GPUs the SDK builds device code for when GPU_TARGETS is not given. Unlike
# rocprofiler_test_gpu_targets.cmake, including this file sets nothing else, so code
# outside the tests can use it. It lives here because the tests are also built from an
# installed copy of this directory.
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
    "gfx1170"
    "gfx1171"
    "gfx1250")
