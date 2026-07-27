# RocjitsuCorpus.cmake — run the rocjitsu-corpus gfx1250 regression through
# mirage + an emulator (HotSwap by default) as part of `ctest`.
#
# The corpus (https://github.com/kuhar/rocjitsu-corpus) packages gfx1250 VMFB
# and TensileLite kernels plus a runner. This module:
#
#   1. makes sure a corpus checkout is available (cloning it if needed), and
#   2. registers ctest tests that drive that runner via
#      tests/corpus/run_corpus.sh, which routes every IREE tool invocation
#      through `mirage run --profile <profile> -- <tool> ...` so the corpus
#      exercises the emulator instead of the bare runtime.
#
# It is opt-in (MIRAGE_RUN_CORPUS, OFF by default) since the corpus is a
# separate repository and the tests need IREE tooling and a working emulator.
# When prerequisites are missing the tests are reported as SKIPPED (the runner
# exits 77), never as spurious failures.
#
# Expects the including scope to define `_mirage_bin` (the built mirage binary).

option(MIRAGE_RUN_CORPUS "Register the rocjitsu-corpus regression as a ctest" OFF)

set(MIRAGE_CORPUS_REPO "git@github.com:kuhar/rocjitsu-corpus.git"
    CACHE STRING "rocjitsu-corpus git URL")
set(MIRAGE_CORPUS_REF "" CACHE STRING "rocjitsu-corpus git ref (branch/tag/sha); empty = default branch")
set(MIRAGE_CORPUS_SRC "${CMAKE_BINARY_DIR}/rocjitsu-corpus"
    CACHE PATH "rocjitsu-corpus checkout (cloned here if absent)")
set(MIRAGE_CORPUS_EMULATOR "hotswap"
    CACHE STRING "Emulator the corpus runs under (hotswap or rocjitsu)")
set(MIRAGE_CORPUS_PROFILE ""
    CACHE STRING "mirage profile name to run the corpus under (default: corpus-<emulator>)")

# Container image carrying the IREE test tools (see tests/corpus/Dockerfile).
# When set, the corpus runs inside a containerised mirage profile built around
# this image instead of relying on host-installed IREE tools.
set(MIRAGE_CORPUS_IMAGE "mirage/iree-corpus:gfx1250"
    CACHE STRING "Container image (with the IREE tools) the corpus runs in; empty = host mode")
option(MIRAGE_CORPUS_BUILD_IMAGE "Add a `corpus_image` target that builds MIRAGE_CORPUS_IMAGE from tests/corpus/Dockerfile" OFF)
set(MIRAGE_CORPUS_IMAGE_BASE "docker.io/rocm/dev-ubuntu-24.04:7.1.1-complete"
    CACHE STRING "Base image the corpus image is built FROM")
set(MIRAGE_CORPUS_IREE_REF "iree-3.12.0rc20260604"
    CACHE STRING "IREE git ref the corpus image builds the IREE tools from")
set(MIRAGE_CONTAINER_PROVIDER "" CACHE STRING "Container provider to build the image with (auto: podman then docker)")

if(NOT MIRAGE_RUN_CORPUS)
  return()
endif()

message(STATUS "mirage: rocjitsu-corpus regression ENABLED (emulator=${MIRAGE_CORPUS_EMULATOR})")

# Clone the corpus at configure time if it is not already present. A failure
# here is non-fatal: the ctest simply SKIPs until a checkout exists.
if(NOT EXISTS "${MIRAGE_CORPUS_SRC}/scripts/run_gfx1250_regression.sh")
  find_program(GIT_EXECUTABLE git)
  if(GIT_EXECUTABLE)
    message(STATUS "mirage:   cloning ${MIRAGE_CORPUS_REPO} -> ${MIRAGE_CORPUS_SRC}")
    set(_corpus_clone "${GIT_EXECUTABLE}" clone --depth 1)
    if(MIRAGE_CORPUS_REF)
      list(APPEND _corpus_clone --branch "${MIRAGE_CORPUS_REF}")
    endif()
    list(APPEND _corpus_clone "${MIRAGE_CORPUS_REPO}" "${MIRAGE_CORPUS_SRC}")
    execute_process(COMMAND ${_corpus_clone} RESULT_VARIABLE _corpus_clone_rc)
    if(NOT _corpus_clone_rc EQUAL 0)
      message(WARNING
        "mirage: failed to clone rocjitsu-corpus (rc=${_corpus_clone_rc}); the "
        "corpus ctest(s) will SKIP until ${MIRAGE_CORPUS_SRC} is populated.")
    endif()
  else()
    message(WARNING
      "mirage: git not found; cannot clone rocjitsu-corpus. The corpus ctest(s) "
      "will SKIP until ${MIRAGE_CORPUS_SRC} is populated.")
  endif()
else()
  message(STATUS "mirage:   using existing corpus checkout ${MIRAGE_CORPUS_SRC}")
endif()

set(_corpus_profile "${MIRAGE_CORPUS_PROFILE}")
if(NOT _corpus_profile)
  set(_corpus_profile "corpus-${MIRAGE_CORPUS_EMULATOR}")
  if(MIRAGE_CORPUS_IMAGE)
    set(_corpus_profile "${_corpus_profile}-img")
  endif()
endif()

set(_corpus_runner "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/run_corpus.sh")
set(_corpus_dockerfile "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus/Dockerfile")

# Pick a container provider for building the image (auto: podman then docker).
set(_corpus_provider "${MIRAGE_CONTAINER_PROVIDER}")
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
# -DMIRAGE_CORPUS_BUILD_IMAGE=ON and run `cmake --build <build> --target corpus_image`.
if(MIRAGE_CORPUS_BUILD_IMAGE AND MIRAGE_CORPUS_IMAGE)
  if(_corpus_provider)
    add_custom_target(corpus_image
      COMMAND ${_corpus_provider} build
              --build-arg "BASE_IMAGE=${MIRAGE_CORPUS_IMAGE_BASE}"
              --build-arg "IREE_REF=${MIRAGE_CORPUS_IREE_REF}"
              -t "${MIRAGE_CORPUS_IMAGE}"
              -f "${_corpus_dockerfile}"
              "${CMAKE_CURRENT_SOURCE_DIR}/tests/corpus"
      COMMENT "Building IREE corpus image ${MIRAGE_CORPUS_IMAGE} (base=${MIRAGE_CORPUS_IMAGE_BASE}, iree=${MIRAGE_CORPUS_IREE_REF})"
      VERBATIM USES_TERMINAL)
  else()
    message(WARNING
      "mirage: MIRAGE_CORPUS_BUILD_IMAGE=ON but no container provider found; "
      "set MIRAGE_CONTAINER_PROVIDER to enable the corpus_image target.")
  endif()
endif()

# Register one ctest per corpus kind so failures localize cleanly. Each forwards
# the configured emulator/profile/image/corpus to the bridge runner and treats
# exit code 77 as "skipped" (missing mirage binary, IREE tools/image, provider,
# or emulator install).
function(_mirage_add_corpus_test test_name only)
  add_test(NAME ${test_name}
    COMMAND ${CMAKE_COMMAND} -E env
            "MIRAGE_BIN=${_mirage_bin}"
            "CORPUS_ROOT=${MIRAGE_CORPUS_SRC}"
            "MIRAGE_CORPUS_EMULATOR=${MIRAGE_CORPUS_EMULATOR}"
            "MIRAGE_CORPUS_PROFILE=${_corpus_profile}"
            "MIRAGE_CORPUS_IMAGE=${MIRAGE_CORPUS_IMAGE}"
            "MIRAGE_CORPUS_ONLY=${only}"
            "MIRAGE_CORPUS_OUT_DIR=${CMAKE_BINARY_DIR}/corpus-results/${only}"
            bash "${_corpus_runner}"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
  set_tests_properties(${test_name} PROPERTIES
    SKIP_RETURN_CODE 77
    TIMEOUT 3600)
endfunction()

_mirage_add_corpus_test(corpus_e2e e2e)
_mirage_add_corpus_test(corpus_matmul matmul)

