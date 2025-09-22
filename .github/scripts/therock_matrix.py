"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/clr": "core",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/rocminfo": "core",
    "projects/rocr-runtime": "core",
}

project_map = {
    "core": {
        "cmake_options": "-DTHEROCK_ENABLE_CORE=ON -DTHEROCK_ENABLE_HIP_RUNTIME=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "hip-tests",
        "subtree_checkout": "projects/clr\nprojects/hip\nprojects/hip-tests\nprojects/rocminfo\nprojects/rocr-runtime",
    },
}
