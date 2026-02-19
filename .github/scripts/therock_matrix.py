"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests.
"""

subtree_to_project_map = {
    "projects/amdsmi": "core",
    "projects/aqlprofile": "profiler",
    "projects/clr": "core",
    "projects/cuid": "rdc",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/hipother": "core",
    "projects/rccl": "comm-libs",
    "projects/rccl-tests": "comm-libs",
    "projects/rdc": "dc_tools",
    "projects/rocdbgapi": "debug_tools",
    "projects/rocdecode": "decoders",
    "projects/rocjpeg": "decoders",
    "projects/rocm-core": "core",
    "projects/rocminfo": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocprofiler": "profiler",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocr-debug-agent": "debug_tools",
    "projects/rocr-runtime": "core",
    "projects/rocshmem": "decoders",
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
    # This needs to be fixed , looks like rocdecode and rocjpeg are not built by TheRock build system.
    "decoders": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "rocdecode-tests, rocjpeg-tests",
    },
    "dc_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DC_TOOLS=ON",
        "projects_to_test": "rdc-tests",
    },
    "debug_tools": {
        "cmake_options": "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON -DTHEROCK_ENABLE_ROCGDB=ON",
        "projects_to_test": "rocdbgapi-tests",
    },
    "comm_libs": {
        "cmake_options": "-DTHEROCK_ENABLE_COMM_LIBS=ON -DTHEROCK_ENABLE_PROFILER=ON",
        "projects_to_test": "rccl-tests",
    },
    "all": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_ALL=OFF",
        "projects_to_test": "hip-tests, rocprofiler-tests",
    },
}
