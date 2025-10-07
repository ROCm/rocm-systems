"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    'projects/aqlprofile': 'profiler', 
    'projects/clr': 'core', 
    'projects/hip': 'core', 
    'projects/hip-tests': 'core', 
    'projects/rocminfo': 'core', 
    'projects/rocprofiler': 'profiler', 
    'projects/rocprofiler-compute': 'profiler', 
    'projects/rocprofiler-register': 'profiler', 
    'projects/rocprofiler-sdk': 'profiler', 
    'projects/rocprofiler-systems': 'profiler', 
    'projects/rocr-runtime': 'core', 
    'projects/roctracer': 'profiler'
}

project_map = {
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_HIP_RUNTIME=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "hip-tests",
        "subtree_checkout": "projects/clr\nprojects/hip\nprojects/hip-tests\nprojects/rocminfo\nprojects/rocr-runtime",
    },
    "profiler": {
        "cmake_options": "-DTHEROCK_ENABLE_PROFILER=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocprofiler-tests",
        "subtree_checkout": "projects/aqlprofile\nprojects/rocprofiler\nprojects/rocprofiler-compute\nprojects/rocprofiler-register\nprojects/rocprofiler-sdk\nprojects/rocprofiler-systems\nprojects/roctracer",
    },
}
