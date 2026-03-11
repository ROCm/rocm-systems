"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/amdsmi": "core",
    "projects/aqlprofile": "profiler",
    "projects/clr": "core",
    "projects/cuid": "rdc",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/hipother": "core",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools",
    # "projects/rocdecode": "media-libs",
    # "projects/rocjpeg": "media-libs",
    "projects/rocm-core": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocminfo": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocprofiler": "profiler",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocprofiler": "profiler",
    "projects/rocr-debug-agent": "debug_tools",
    "projects/rocr-runtime": "core",
    "projects/roctracer": "profiler",
}

project_map = {
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocrtst",
    },
    "profiler": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "aqlprofile, rocprofiler-compute, rocprofiler_systems",
    },
    # media libs to be enabled in following PR
    # "media-libs": {
    #     "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_MEDIA_LIBS=ON",
    #     "projects_to_test": "", # "rocdecode-tests, rocjpeg-tests",
    # },
    "dc_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DC_TOOLS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "",  # rdc-tests is not built by TheRock build system - TBD
    },
    "debug_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "rocr-debug-agent",  # rocgdb testing requires custom container support in rocm-systems, to be enabled in a future PR
    },
    "all": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler_systems, rocr-debug-agent",
    },
}

trigger_windows_ci_for_subtrees_paths = [
    "projects/clr/*",
    "projects/hip/*",
    "projects/hip-tests/*",
    "projects/rocr-runtime/*",
    "shared/amdgpu-windows-interop/**",
    ".github/*/therock*"
]
