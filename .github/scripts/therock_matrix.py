"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/aqlprofile": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/roctracer": "profiler",
    "projects/rocm_smi_lib": "base",
    "projects/rocm-core": "base",
    "projects/rocprofiler-register": "base",
    "projects/clr": "core",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/rocminfo": "core",
    "projects/rocr-runtime": "core",
}

project_map = {
    "profiler": {
        "cmake_options": "-DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "",
        "subtree_checkout": "projects/aqlprofile\nprojects/rocprofiler-sdk\nprojects/roctracer\nprojects/rocprofiler-register\nprojects/clr\nprojects/hip\nprojects/hip-tests\nprojects/rocminfo\nprojects/rocr-runtime\nprojects/rocm_smi_lib\nprojects/rocm-core\nprojects/rocprofiler-register",
    },
    "base": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "",
        "subtree_checkout": "projects/rocm_smi_lib\nprojects/rocm-core\nprojects/rocprofiler-register",
    },
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "",
        "subtree_checkout": "projects/clr\nprojects/hip\nprojects/hip-tests\nprojects/rocminfo\nprojects/rocr-runtime",
    },
}
