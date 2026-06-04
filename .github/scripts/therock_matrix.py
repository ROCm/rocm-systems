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
    # NOTE: This intentionally does NOT enable math/ML libraries (rocBLAS, rocFFT,
    # MIOpen, composable_kernel, ...). Building those is by far the most expensive
    # part of a ROCm build and the profiler tests below do not exercise them. They
    # are auto-enabled only when a selected test needs them (see
    # feature_groups_for_tests) and are always built by the nightly job, which
    # provides downstream-library build coverage.
    "profiler": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_PROFILER=ON",
        ],
        "projects_to_test": "aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems",
    },
    "rocshmem": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=OFF", "-DTHEROCK_ENABLE_ROCSHMEM=ON"],
        "projects_to_test": "",  # rocshmem testing to be enabled in a future PR
    },
    # Also test rocr-debug-agent and rocgdb since those depend on runtimes.
    # See the note on "profiler" above: math/ML libraries are not enabled here and
    # are auto-detected from projects_to_test / covered by the nightly job.
    "runtimes": {
        "cmake_options": [
            "-DTHEROCK_ENABLE_ALL=OFF",
            "-DTHEROCK_ENABLE_CORE=ON",
            "-DTHEROCK_ENABLE_PROFILER=ON",
            "-DTHEROCK_ENABLE_DEBUG_TOOLS=ON",
        ],
        "projects_to_test": "hip-tests, rocrtst, rocprofiler-sdk, rocr-debug-agent, rocgdb",
    },
    "all": {
        "cmake_options": ["-DTHEROCK_ENABLE_ALL=ON"],
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems, rocr-debug-agent, rocgdb",
    },
    # Same test coverage as TheRock submodule-bump PRs (rocm-systems scope).
    # Nightly (schedule) uses this entry explicitly for alignment.
    # additional mathlib to test for nightly: rocprim, rocthrust, rocrand, hiprand, hipblaslt, rocblas, hipblas, rocroller, miopen, miopenprovider, hipfft, rocfft, rocsparse, hipsparse, hipsparselt, rocsolver, hipsolver, rocwmma
    # instead of above blanket addition of all tests, we can add logic to determine which mathlibs to test, based on file changes from last nightly run. Can be handled once the tests scripts move to component/monorepo src
    "nightly": {
        "cmake_options": "-DTHEROCK_ENABLE_ALL=ON",
        "projects_to_test": "hip-tests, rocrtst, aqlprofile, rocprofiler-compute, rocprofiler-sdk, rocprofiler-systems, rocr-debug-agent, rocgdb, rocprim, rocthrust, rocrand, hiprand, hipblaslt, rocblas, hipblas, rocroller, miopen, miopenprovider, hipfft, rocfft, rocsparse, hipsparse, hipsparselt, rocsolver, hipsolver, rocwmma",
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

# ---------------------------------------------------------------------------
# Math/ML library detection
#
# Math libraries (rocBLAS, rocFFT, rocSPARSE, ...) and ML libraries (MIOpen,
# composable_kernel, ...) are the most expensive part of a ROCm build. To keep
# presubmit fast, feature groups MATH_LIBS / ML_LIBS are left OFF by default and
# only turned ON when a test that is actually being run needs the corresponding
# artifacts.
#
# The sets below mirror the per-test `fetch_artifact_args` declared in TheRock's
# build_tools/github_actions/fetch_test_configurations.py:
#   * tests fetching --blas/--rand/--fft/--prim/--rocwmma/--libhipcxx -> MATH_LIBS
#   * tests fetching --miopen/--hipdnn*/--*provider                    -> ML_LIBS
# ML libraries depend on math libraries, so ML_LIBS implies MATH_LIBS.
#
# Keep these in sync with fetch_test_configurations.py when a test's artifact
# requirements change.
# ---------------------------------------------------------------------------
tests_requiring_math_libs = {
    "rocblas",
    "rocroller",
    "tensilelite",
    "hipblas",
    "hipblaslt",
    "hipsolver",
    "rocsolver",
    "rocprim",
    "hipcub",
    "rocthrust",
    "hipsparse",
    "rocsparse",
    "hipsparselt",
    "rocrand",
    "hiprand",
    "rocfft",
    "hipfft",
    "rocwmma",
    "libhipcxx_hipcc",
    "libhipcxx_hiprtc",
}

tests_requiring_ml_libs = {
    "miopen",
    "hipdnn",
    "hipdnn_install",
    "hipdnn-integration-tests",
    "hipdnn-samples",
    "miopenprovider",
    "hipblasltprovider",
    "hipkernelprovider",
}


def feature_groups_for_tests(tests):
    """Returns the optional feature groups required by the given tests.

    Args:
        tests: Iterable of test/component names (e.g. "rocblas", "miopen").

    Returns:
        A subset of {"MATH_LIBS", "ML_LIBS"}. ML_LIBS implies MATH_LIBS because
        the ML libraries are built on top of the math libraries.
    """
    requested = {t.strip() for t in tests if t and t.strip()}
    features = set()
    if requested & tests_requiring_math_libs:
        features.add("MATH_LIBS")
    if requested & tests_requiring_ml_libs:
        features.add("ML_LIBS")
        features.add("MATH_LIBS")
    return features
