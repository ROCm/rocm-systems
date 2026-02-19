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
    "projects/rdc": "rdc",
    "projects/rccl": "comm-libs",
    "projects/rccl-tests": "comm-libs",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools",
    "projects/rocdecode": "media-libs",
    "projects/rocjpeg": "media-libs",
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
    "projects/rocshmem": "comm-libs",
    "projects/roctracer": "profiler",
}

project_map = {
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests",
    },
    "profiler": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "rocprofiler-tests",
    },
    # # This needs to be enabled as part of PR 3358 , currently not built by TheRock build system.
    # "media-libs": {
    #     "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_MEDIA_LIBS=OFF",
    #     "projects_to_test": "rocdecode-tests, rocjpeg-tests",
    # },
    "dc_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DC_TOOLS=ON",
        "projects_to_test": "", # rdc-tests is not built by TheRock build system - TBD
    },
    "debug_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON -DTHEROCK_ENABLE_ROCGDB=ON",
        "projects_to_test": "", # rocdbgapi-tests is not built by TheRock build system - TBD
    },
    "comm_libs": {
        "cmake_options": "-DTHEROCK_ENABLE_COMM_LIBS=ON -DTHEROCK_ENABLE_RCCL=ON -DTHEROCK_ENABLE_PROFILER=ON",
        "projects_to_test": "rccl-tests",
    },
    "all": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocprofiler-tests",
    },
}
