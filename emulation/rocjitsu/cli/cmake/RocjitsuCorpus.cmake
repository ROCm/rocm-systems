# RocjitsuCorpus.cmake — run the rocjitsu-corpus gfx1250 regression through
# rocjitsu + an emulator (HotSwap by default) as part of `ctest`.
#
# The corpus (https://github.com/kuhar/rocjitsu-corpus) packages gfx1250 VMFB
# and TensileLite kernels plus a runner. This module:
#
#   1. makes sure a corpus checkout is available (cloning it if needed), and
#   2. registers ctest tests that drive that runner via
#      tests/corpus/run_corpus.sh, which routes every IREE tool invocation
#      through `rocjitsu run --profile <profile> -- <tool> ...` so the corpus
#      exercises the emulator instead of the bare runtime.
#
# It is opt-in (ROCJITSU_RUN_CORPUS, OFF by default) since the corpus is a
# separate repository and the tests need IREE tooling and a working emulator.
# When prerequisites are missing the tests are reported as SKIPPED (the runner
# exits 77), never as spurious failures.
#
# Expects the including scope to define `_rocjitsu_bin` (the built rocjitsu binary).

option(
    ROCJITSU_RUN_CORPUS
    "Register the rocjitsu-corpus regression as a ctest"
    OFF
)

set(ROCJITSU_CORPUS_REPO
    "git@github.com:kuhar/rocjitsu-corpus.git"
    CACHE STRING
    "rocjitsu-corpus git URL"
)
set(ROCJITSU_CORPUS_REF
    ""
    CACHE STRING
    "rocjitsu-corpus git ref (branch/tag/sha); empty = default branch"
)
set(ROCJITSU_CORPUS_SRC
    "${CMAKE_BINARY_DIR}/rocjitsu-corpus"
    CACHE PATH
    "rocjitsu-corpus checkout (cloned here if absent)"
)
set(ROCJITSU_CORPUS_EMULATOR
    "hotswap"
    CACHE STRING
    "Emulator the corpus runs under (hotswap or rocjitsu)"
)
set(ROCJITSU_CORPUS_PROFILE
    ""
    CACHE STRING
    "rocjitsu profile name to run the corpus under (default: corpus-<emulator>)"
)

# Container image carrying the IREE test tools (see tests/corpus/Dockerfile).
# When set, the corpus runs inside a containerised rocjitsu profile built around
# this image instead of relying on host-installed IREE tools.
set(ROCJITSU_CORPUS_IMAGE
    "rocjitsu/iree-corpus:gfx1250"
    CACHE STRING
    "Container image (with the IREE tools) the corpus runs in; empty = host mode"
)
option(
    ROCJITSU_CORPUS_BUILD_IMAGE
    "Add a `corpus_image` target that builds ROCJITSU_CORPUS_IMAGE from tests/corpus/Dockerfile"
    OFF
)
set(ROCJITSU_CORPUS_IMAGE_BASE
    "docker.io/rocm/dev-ubuntu-24.04:7.1.1-complete"
    CACHE STRING
    "Base image the corpus image is built FROM"
)
set(ROCJITSU_CORPUS_IREE_REF
    "iree-3.12.0rc20260604"
    CACHE STRING
    "IREE git ref the corpus image builds the IREE tools from"
)
set(ROCJITSU_CONTAINER_PROVIDER
    ""
    CACHE STRING
    "Container provider to build the image with (auto: podman then docker)"
)

if(NOT ROCJITSU_RUN_CORPUS)
    return()
endif()

message(
    STATUS
    "rocjitsu: rocjitsu-corpus regression ENABLED (emulator=${ROCJITSU_CORPUS_EMULATOR})"
)

# Clone the corpus at configure time if it is not already present. A failure
# here is non-fatal: the ctest simply SKIPs until a checkout exists.
if(NOT EXISTS "${ROCJITSU_CORPUS_SRC}/scripts/run_gfx1250_regression.sh")
    find_program(GIT_EXECUTABLE git)
    if(GIT_EXECUTABLE)
        message(
            STATUS
            "rocjitsu:   cloning ${ROCJITSU_CORPUS_REPO} -> ${ROCJITSU_CORPUS_SRC}"
        )
        set(_corpus_clone "${GIT_EXECUTABLE}" clone --depth 1)
        if(ROCJITSU_CORPUS_REF)
            list(APPEND _corpus_clone --branch "${ROCJITSU_CORPUS_REF}")
        endif()
        list(
            APPEND _corpus_clone
            "${ROCJITSU_CORPUS_REPO}"
            "${ROCJITSU_CORPUS_SRC}"
        )
        execute_process(
            COMMAND ${_corpus_clone}
            RESULT_VARIABLE _corpus_clone_rc
        )
        if(NOT _corpus_clone_rc EQUAL 0)
            message(
                WARNING
                "rocjitsu: failed to clone rocjitsu-corpus (rc=${_corpus_clone_rc}); the "
                "corpus ctest(s) will SKIP until ${ROCJITSU_CORPUS_SRC} is populated."
            )
        endif()
    else()
        message(
            WARNING
            "rocjitsu: git not found; cannot clone rocjitsu-corpus. The corpus ctest(s) "
            "will SKIP until ${ROCJITSU_CORPUS_SRC} is populated."
        )
    endif()
else()
    message(
        STATUS
        "rocjitsu:   using existing corpus checkout ${ROCJITSU_CORPUS_SRC}"
    )
endif()

set(_corpus_profile "${ROCJITSU_CORPUS_PROFILE}")
if(NOT _corpus_profile)
    set(_corpus_profile "corpus-${ROCJITSU_CORPUS_EMULATOR}")
    if(ROCJITSU_CORPUS_IMAGE)
        set(_corpus_profile "${_corpus_profile}-img")
    endif()
endif()

set(_corpus_runner "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/run_corpus.sh")
set(_corpus_dockerfile "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/Dockerfile")

# Pick a container provider for building the image (auto: podman then docker).
set(_corpus_provider "${ROCJITSU_CONTAINER_PROVIDER}")
if(NOT _corpus_provider)
    find_program(_corpus_podman podman)
    find_program(_corpus_docker docker)
    if(_corpus_podman)
        set(_corpus_provider "${_corpus_podman}")
    elseif(_corpus_docker)
        set(_corpus_provider "${_corpus_docker}")
    endif()
endif()

# Optional target that builds the IREE-tools image from tests/corpus/Dockerfile.
# Off by default since it is a from-source IREE build; enable with
# -DROCJITSU_CORPUS_BUILD_IMAGE=ON and run `cmake --build <build> --target corpus_image`.
if(ROCJITSU_CORPUS_BUILD_IMAGE AND ROCJITSU_CORPUS_IMAGE)
    if(_corpus_provider)
        add_custom_target(
            corpus_image
            COMMAND
                ${_corpus_provider} build --build-arg
                "BASE_IMAGE=${ROCJITSU_CORPUS_IMAGE_BASE}" --build-arg
                "IREE_REF=${ROCJITSU_CORPUS_IREE_REF}" -t
                "${ROCJITSU_CORPUS_IMAGE}" -f "${_corpus_dockerfile}"
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus"
            COMMENT
                "Building IREE corpus image ${ROCJITSU_CORPUS_IMAGE} (base=${ROCJITSU_CORPUS_IMAGE_BASE}, iree=${ROCJITSU_CORPUS_IREE_REF})"
            VERBATIM
            USES_TERMINAL
        )
    else()
        message(
            WARNING
            "rocjitsu: ROCJITSU_CORPUS_BUILD_IMAGE=ON but no container provider found; "
            "set ROCJITSU_CONTAINER_PROVIDER to enable the corpus_image target."
        )
    endif()
endif()

# Register one ctest per corpus kind so failures localize cleanly. Each forwards
# the configured emulator/profile/image/corpus to the bridge runner and treats
# exit code 77 as "skipped" (missing rocjitsu binary, IREE tools/image, provider,
# or emulator install).
function(_rocjitsu_add_corpus_test test_name only)
    add_test(
        NAME ${test_name}
        COMMAND
            ${CMAKE_COMMAND} -E env "ROCJITSU_BIN=${_rocjitsu_bin}"
            "CORPUS_ROOT=${ROCJITSU_CORPUS_SRC}"
            "ROCJITSU_CORPUS_EMULATOR=${ROCJITSU_CORPUS_EMULATOR}"
            "ROCJITSU_CORPUS_PROFILE=${_corpus_profile}"
            "ROCJITSU_CORPUS_IMAGE=${ROCJITSU_CORPUS_IMAGE}"
            "ROCJITSU_CORPUS_ONLY=${only}"
            "ROCJITSU_CORPUS_OUT_DIR=${CMAKE_BINARY_DIR}/corpus-results/${only}"
            bash "${_corpus_runner}"
        WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    )
    set_tests_properties(
        ${test_name}
        PROPERTIES SKIP_RETURN_CODE 77 TIMEOUT 3600
    )
endfunction()

_rocjitsu_add_corpus_test(corpus_e2e e2e)
_rocjitsu_add_corpus_test(corpus_matmul matmul)
