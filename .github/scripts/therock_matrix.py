"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""

subtree_to_project_map = {
    "emulation/rocjitsu": "emulation",
    "emulation/mirage": "emulation",
    "projects/amdsmi": "core",
    "projects/aqlprofile": "profiler",
    "projects/clr": "runtimes",
    "projects/cuid": "rdc",
    "projects/hip": "runtimes",
    "projects/hip-tests": "runtimes",
    "projects/hipother": "runtimes",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools-dbgapi",
    # "projects/rocdecode": "media-libs",
    # "projects/rocjpeg": "media-libs",
    "projects/rocm-core": "core",
    "projects/rocminfo": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocprofiler": "profiler",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocr-debug-agent": "debug_tools-debug-agent",
    "projects/hotswap": "runtimes",
    "projects/rocr-runtime": "runtimes",
    "projects/rocshmem": "rocshmem",
    "projects/roctracer": "profiler",
    "shared/amdgpu-windows-interop": "runtimes",
}
# Below is the comprehensive list which the rock-ci runs for bump PRs. TODO - fetch this list programmatically using TheRock's build_tools/github_actions/fetch_test_configurations.py
therock_projects_to_test = "aqlprofile, hip-tests, hipblas, hipblaslt, hipcub, hipfft, hiprand, hipsolver, hipsparse, hipsparselt, miopen, miopenprovider, rocblas, rocfft, rocgdb, rocprim, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems, rocr-debug-agent, rocrand, rocroller, rocrtst, rocsolver, rocsparse, rocthrust, rocwmma"
project_map = {
    "core": {
        "cmake_options": ["-DTHEROCK_ENABLE_CORE=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "projects_to_test": "",  # will run sanity test to cover rocminfo and amdsmi
    },
    "emulation": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_EMULATION=ON"],
        "projects_to_test": "",
    },
    "dc_tools": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_DC_TOOLS=ON"],
        "projects_to_test": "",  # rdc-tests is not built by TheRock build system - TBD
    },
    # dbgapi changes need to exercise both ROCgdb and debug agent.
    "debug_tools-dbgapi": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ],
        "projects_to_test": "rocr-debug-agent, rocgdb",
    },
    # debug agent changes don't have to exercise ROCgdb.
    "debug_tools-debug-agent": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ],
        "projects_to_test": "rocr-debug-agent",
    },
    # media libs to be enabled in following PR
    # "media-libs": {
    #     "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_MEDIA_LIBS=ON"],
    #     "projects_to_test": "", # "rocdecode-tests, rocjpeg-tests",
    # },
    "profiler": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"],
        "projects_to_test": "aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems",
    },
    "rocshmem": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_ROCSHMEM=ON"],
        "projects_to_test": "",  # rocshmem testing to be enabled in a future PR
    },
    # Also test rocr-debug-agent and rocgdb since those depend on runtimes.
    # Mathlib tests are included because runtime changes (hip, rocr, clr) can affect
    # the full math library stack. This matches nightly test coverage for gfx94x.
    "runtimes": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"],
        #TODO - Enable miopen, miopenprovider and rocwmma once they start passing in ci-nightly
        "projects_to_test": "hip-tests, hipblas, hipblaslt, hipcub, hipfft, hiprand, hipsolver, hipsparse, hipsparselt, rocblas, rocfft, rocgdb, rocprim, rocprofiler-sdk, rocr-debug-agent, rocrand, rocroller, rocrtst, rocsolver, rocsparse, rocthrust",
    },
    "all": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"],
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems, rocr-debug-agent, rocgdb",
    },
    # Same test coverage as TheRock submodule-bump PRs (rocm-systems scope).
    # Nightly (schedule) uses this entry explicitly for alignment with runtimes coverage.
    "nightly": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"],
        "projects_to_test": therock_projects_to_test,
    },
}

# Subtrees that should only trigger Windows CI, not Linux CI.
# Note: Linux-only subtrees (e.g. projects/rocshmem) have no explicit list —
# any subtree absent from trigger_windows_ci_for_subtrees_paths will
# automatically skip Windows CI.
windows_only_subtrees = {
    "shared/amdgpu-windows-interop",
}

# Paths matching any of these patterns will trigger Windows CI.
# Subtrees not represented here are treated as Linux-only.
trigger_windows_ci_for_subtrees_paths = [
    "projects/clr/*",
    "projects/hip/*",
    "projects/hip-tests/*",
    "projects/rocr-runtime/*",
    "shared/amdgpu-windows-interop/**",
    ".github/*/therock*",
]
