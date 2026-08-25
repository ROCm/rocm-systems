# Pure-logic tests for the DXG (WSL) thunk. No GPU, DXCore adapter or open
# thunk is involved, so these run on any machine that can build the library.
#
# libhsakmt has no test framework; ../../../tests/unit_test_harness.h is the
# whole harness. These targets are build-only artifacts and are deliberately
# not installed.

add_executable(dxg-snapshot-refcount-test snapshot_refcount_test.cpp)
target_include_directories(
    dxg-snapshot-refcount-test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../../../tests
            ${CMAKE_CURRENT_SOURCE_DIR}/../../src/dxg
)
set_target_properties(
    dxg-snapshot-refcount-test
    PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)

add_test(NAME dxg-snapshot-refcount COMMAND dxg-snapshot-refcount-test)

# ---------------------------------------------------------------------------
# Installed public headers: is the dev package self-contained?
#
# Stages ROCDXG_PUBLIC_HEADERS - the exact list install() ships, so the two
# cannot drift - into a prefix of its own, then compiles a C and a C++ consumer
# that see that prefix and nothing else. No path into include/hsakmt is on the
# include line, so a header the install rule forgot is a build failure here
# instead of a bug report from whoever tried to use the package. That matters
# to more than packaging tidiness: rocprofiler-sdk reads the WSL topology
# through <hsakmt/hsakmttypes.h>, and a dev package without it leaves the SDK
# hand-copying HsaNodeProperties.
#
# Header-only on purpose. The production link needs libwkmi.a, which is not in
# this checkout, so these define the entry point themselves rather than
# pretending librocdxg linked. That checks the declaration harder anyway: a
# definition has to agree with the header.
# ---------------------------------------------------------------------------

set(ROCDXG_STAGED_PREFIX ${CMAKE_CURRENT_BINARY_DIR}/staged-prefix)
set(ROCDXG_STAGED_INCLUDE ${ROCDXG_STAGED_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR})

set(staged_headers "")
foreach(header IN LISTS ROCDXG_PUBLIC_HEADERS)
    get_filename_component(header_name ${header} NAME)
    set(staged ${ROCDXG_STAGED_INCLUDE}/hsakmt/${header_name})
    add_custom_command(
        OUTPUT ${staged}
        COMMAND ${CMAKE_COMMAND} -E copy_if_different ${header} ${staged}
        DEPENDS ${header}
        COMMENT "Staging ${header_name} into the test install prefix"
    )
    list(APPEND staged_headers ${staged})
endforeach()

add_custom_target(dxg-staged-public-headers DEPENDS ${staged_headers})

add_executable(dxg-installed-header-c-test installed_header_smoke.c)
add_dependencies(dxg-installed-header-c-test dxg-staged-public-headers)
target_include_directories(
    dxg-installed-header-c-test PRIVATE ${ROCDXG_STAGED_INCLUDE}
)
set_target_properties(
    dxg-installed-header-c-test
    PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON
)

add_executable(dxg-installed-header-cxx-test installed_header_smoke.cpp)
add_dependencies(dxg-installed-header-cxx-test dxg-staged-public-headers)
target_include_directories(
    dxg-installed-header-cxx-test PRIVATE ${ROCDXG_STAGED_INCLUDE}
)
set_target_properties(
    dxg-installed-header-cxx-test
    PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED ON
)

add_test(NAME dxg-installed-header-c COMMAND dxg-installed-header-c-test)
add_test(NAME dxg-installed-header-cxx COMMAND dxg-installed-header-cxx-test)

# What the built artifact's dynamic symbol table actually offers. Nothing else
# here links or loads librocdxg - the refcount test compiles a header on its
# own and the two smoke tests define their entry point rather than binding it -
# so this is the only test that speaks for what ROCr's dlsym() will find. A
# symbol can be lost to mangling or to local visibility while the header and
# the version script both still read correctly, and nothing at link time would
# notice, because ROCr binds every one of these by name at run time.
#
# The names are given here rather than parsed out of the version script, so
# deleting an entry there fails here rather than quietly narrowing the
# five-call contract shared by ROCr and rocprofiler-sdk.
#
# Requires nm, which a cross-build might not have.
if(CMAKE_NM)
    add_test(
        NAME dxg-dynamic-exports
        COMMAND
            ${CMAKE_COMMAND} -DLIBRARY=$<TARGET_FILE:${ROCDXG_TARGET}>
            -DNM=${CMAKE_NM} -DEXPECTED_VERSION=HSAKMT_1
            -DEXPECTED_SYMBOLS=hsaKmtOpenKFD,hsaKmtCloseKFD,hsaKmtAcquireSystemProperties,hsaKmtReleaseSystemProperties,hsaKmtGetNodeProperties
            -P
            ${CMAKE_CURRENT_SOURCE_DIR}/check_dynamic_exports.cmake
    )
else()
    message(
        STATUS
        "librocdxg tests: no nm found, skipping the dynamic-export test. "
        "Set CMAKE_NM to run it."
    )
endif()
