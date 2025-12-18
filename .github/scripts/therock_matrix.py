"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/aqlprofile": "profiler",
    "projects/clr": "core",
    "projects/hip": "core",
    "projects/hip-tests": "core",
    "projects/hipother": "core",
    "projects/rdc": "rdc",
    "projects/rocm-core": "core",
    "projects/rocm-smi-lib": "core",
    "projects/rocminfo": "core",
    "projects/rocprofiler-compute": "profiler",
    "projects/rocprofiler-register": "profiler",
    "projects/rocprofiler-sdk": "profiler",
    "projects/rocprofiler-systems": "profiler",
    "projects/rocprofiler": "profiler",
    "projects/rocr-runtime": "core",
    "projects/roctracer": "profiler",
}

project_map = {
    "core": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_HIP_RUNTIME=ON",
            "-DTHEROCK_ENABLE_ALL=OFF",
        ],
        "project_to_test": "hip-tests",
    },
    "profiler": {
        "cmake_options": ["-DTHEROCK_ENABLE_PROFILER=ON", "-DTHEROCK_ENABLE_ALL=OFF"],
        "project_to_test": "rocprofiler-tests",
    },
    "all": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_PROFILER=ON",
            "-DTHEROCK_ENABLE_ALL=OFF",
        ],
        "project_to_test": "hip-tests, rocprofiler-tests",
    },
}


def collect_projects_to_run(subtrees, enable_rocm_libraries):
    projects = set()
    # collect the associated subtree to project
    for subtree in subtrees:
        if subtree in subtree_to_project_map:
            projects.add(subtree_to_project_map.get(subtree))

    # retrieve the subtrees to checkout, cmake options to build, and projects to test
    project_to_run = []
    # Currently as we have no tests, we just build all packages available if an applicable change is made.
    # As we start to get an idea of test times, we can divide test jobs.
    if projects:
        for project in ["all"]:
            if project in project_map:
                # If the "enable_rocm_libraries" label is set, we allow the ROCm Libraries to be used and we build everything
                if enable_rocm_libraries:
                    project_map[project]["cmake_options"] = [
                        "-DTHEROCK_ROCM_LIBRARIES_SOURCE_DIR=../rocm-libraries", "-DTHEROCK_USE_EXTERNAL_COMPOSABLE_KERNEL=ON", "-DTHEROCK_COMPOSABLE_KERNEL_SOURCE_DIR=../composable_kernel"
                    ]

                cmake_flag_options = " ".join(project_map[project]["cmake_options"])
                project_map[project]["cmake_options"] = cmake_flag_options
                project_to_run.append(project_map.get(project))

    return project_to_run
