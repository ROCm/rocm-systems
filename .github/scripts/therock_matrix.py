"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/rocprim": "prim",
    "projects/rocthrust": "prim",
    "projects/hipcub": "prim",
    "projects/rocrand": "rand",
    "projects/hiprand": "rand",
    "shared/": "clr",
}

project_map = {
    "prim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocprim, rocthrust, hipcub",
        "subtree_checkout": "projects/rocprim\nprojects/hipcub\nprojects/rocthrust",
    },
    "rand": {
        "cmake_options": "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocrand, hiprand",
        "subtree_checkout": "projects/rocrand\nprojects/hiprand",
    },
    "clr": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=OFF -DTHEROCK_ENABLE_HIP_RUNTIME=ON '-DTHEROCK_AMDGPU_FAMILIES=gfx115X-all;gfx120X-all'",
        "project_to_test": "clr, hip, hip-tests",
        "subtree_checkout": "projects/clr\nprojects/hip\nprojects/hip-tests",
    }
}
