"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/amdsmi": "core",
    "projects/aqlprofile": "profiler",
    "projects/clr": "runtimes",
    "projects/cuid": "rdc",
    "projects/hip": "runtimes",
    "projects/hip-tests": "runtimes",
    "projects/hipother": "runtimes",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools",
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
    "projects/rocprofiler": "profiler",
    "projects/rocr-debug-agent": "debug_tools",
    "projects/rocr-runtime": "runtimes",
    "projects/rocshmem": "rocshmem",
    "projects/roctracer": "profiler",
    "shared/amdgpu-windows-interop": "runtimes",
}

project_map = {
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "",  # will run sanity test to cover rocminfo and amdsmi
    },
    "dc_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DC_TOOLS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "",  # rdc-tests is not built by TheRock build system - TBD
    },
    "debug_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "rocr-debug-agent, rocgdb",
    },
    # media libs to be enabled in following PR
    # "media-libs": {
    #     "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_MEDIA_LIBS=ON",
    #     "projects_to_test": "", # "rocdecode-tests, rocjpeg-tests",
    # },
    "profiler": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "aqlprofile, rocprofiler-compute, rocprofiler-systems",
    },
    "rocshmem": {
        "cmake_options": "-DTHEROCK_ENABLE_ROCSHMEM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "",  # rocshmem testing to be enabled in a future PR
    },
    "runtimes": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocrtst",
    },
    "all": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler-systems, rocr-debug-agent, rocgdb",
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
