# Opt-in guard: compares amd-smi's vendored librocdxg ABI against upstream.
# Off by default so the normal build never needs rocr-runtime's source tree.
#
#   cmake -DENABLE_WSL_BACKEND=ON -DVERIFY_ROCDXG_ABI=ON \
#         -DROCDXG_UPSTREAM_INCLUDE_DIR=<path-to>/libhsakmt/include

find_path(ROCDXG_UPSTREAM_INCLUDE_DIR hsakmt/rocdxg_smi.h HINTS ${PROJECT_SOURCE_DIR}/../rocr-runtime/libhsakmt/include)

if(NOT ROCDXG_UPSTREAM_INCLUDE_DIR)
    message(
        FATAL_ERROR
        "VERIFY_ROCDXG_ABI=ON but hsakmt/rocdxg_smi.h was not found. "
        "Set ROCDXG_UPSTREAM_INCLUDE_DIR to a libhsakmt include tree."
    )
endif()

add_executable(rocdxg_abi_check rocdxg_abi_check.cc)
target_include_directories(rocdxg_abi_check PRIVATE ${PROJECT_SOURCE_DIR}/include ${ROCDXG_UPSTREAM_INCLUDE_DIR})
# Not registered with ctest: every check is a static_assert, so building the
# target is what runs them.
