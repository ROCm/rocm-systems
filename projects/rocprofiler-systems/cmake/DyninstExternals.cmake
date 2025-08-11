include(MacroUtilities)

# Map deprecated DYNINST_BUILD_* variables to new ROCPROFSYS_BUILD_* variables
foreach(dep BOOST TBB ELFUTILS LIBIBERTY)
    if(DYNINST_BUILD_${dep})
        message(
            WARNING
            "DYNINST_BUILD_${dep} is deprecated. Using ROCPROFSYS_BUILD_${dep} instead."
        )
        set(ROCPROFSYS_BUILD_${dep} ON)
    endif()
endforeach()

# Set BUILD_* to ON if ROCPROFSYS_BUILD_* is ON
foreach(dep BOOST TBB ELFUTILS LIBIBERTY)
    if(ROCPROFSYS_BUILD_${dep})
        if(dep STREQUAL "BOOST")
            rocprofiler_systems_add_option(BUILD_BOOST "Enable building Boost internally"
                                           ON
            )
        elseif(dep STREQUAL "TBB")
            rocprofiler_systems_add_option(BUILD_TBB "Enable building TBB internally" ON)
        elseif(dep STREQUAL "ELFUTILS")
            rocprofiler_systems_add_option(BUILD_ELFUTILS
                                           "Enable building elfutils internally" ON
            )
        elseif(dep STREQUAL "LIBIBERTY")
            rocprofiler_systems_add_option(BUILD_LIBIBERTY
                                           "Enable building libiberty internally" ON
            )
        endif()
    endif()
endforeach()

set(TPL_STAGING_PREFIX
    "${PROJECT_BINARY_DIR}/external"
    CACHE PATH
    "Third-party library build-tree install prefix"
)
file(MAKE_DIRECTORY "${TPL_STAGING_PREFIX}")
file(MAKE_DIRECTORY "${TPL_STAGING_PREFIX}/include")

add_custom_target(external-prebuild)

# Add external dependencies to be built
include(DyninstBoost)
if(TARGET rocprofiler-systems-boost-build)
    # Make Boost build serially
    set_target_properties(
        rocprofiler-systems-boost
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    # Create a prebuild target that depends on Boost
    add_dependencies(external-prebuild rocprofiler-systems-boost-build)
endif()

include(DyninstTBB)
if(TARGET rocprofiler-systems-tbb-build AND TARGET external-prebuild)
    # Make TBB build serially and wait for Boost
    set_target_properties(
        rocprofiler-systems-tbb-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    add_dependencies(external-prebuild rocprofiler-systems-tbb-build)
endif()

include(DyninstElfUtils)
if(TARGET rocprofiler-systems-elfutils-build AND TARGET external-prebuild)
    set_target_properties(
        rocprofiler-systems-elfutils-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    add_dependencies(external-prebuild rocprofiler-systems-elfutils-build)
endif()

include(DyninstLibIberty)
if(TARGET rocprofiler-systems-libiberty-build AND TARGET external-prebuild)
    set_target_properties(
        rocprofiler-systems-libiberty-build
        PROPERTIES JOB_POOL_COMPILE external_deps_pool JOB_POOL_LINK external_deps_pool
    )
    add_dependencies(external-prebuild rocprofiler-systems-libiberty-build)
endif()

# Final dependency check
if(NOT TARGET external-prebuild)
    message(WARNING "Not all dyninst external dependencies found. Build may fail.")
endif()

# Create a dummy target to ensure external dependencies are fully built
add_custom_target(external-deps-complete)
if(TARGET external-prebuild)
    add_dependencies(external-deps-complete external-prebuild)
endif()

if(NOT TARGET Dyninst::Boost AND TARGET rocprofiler-systems-boost)
    add_library(Dyninst::Boost INTERFACE IMPORTED)
    set_target_properties(
        Dyninst::Boost
        PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-boost
    )
    message(
        STATUS
        "Created imported target Dyninst::Boost linked to rocprofiler-systems-boost"
    )
endif()

if(NOT TARGET Dyninst::ElfUtils AND TARGET rocprofiler-systems-elfutils)
    add_library(Dyninst::ElfUtils INTERFACE IMPORTED)
    set_target_properties(
        Dyninst::ElfUtils
        PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-elfutils
    )
    message(STATUS "Created imported target Dyninst::ElfUtils linked to ElfUtils")
endif()

if(NOT TARGET Dyninst::TBB AND TARGET rocprofiler-systems-tbb)
    add_library(Dyninst::TBB INTERFACE IMPORTED)
    set_target_properties(
        Dyninst::TBB
        PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-tbb
    )
    message(
        STATUS
        "Created imported target Dyninst::TBB linked to rocprofiler-systems-tbb"
    )
endif()

if(NOT TARGET Dyninst::LibIberty AND TARGET rocprofiler-systems-libiberty)
    add_library(Dyninst::LibIberty INTERFACE IMPORTED)
    set_target_properties(
        Dyninst::LibIberty
        PROPERTIES INTERFACE_LINK_LIBRARIES rocprofiler-systems-libiberty
    )
    message(
        STATUS
        "Created imported target Dyninst::LibIberty linked to rocprofiler-systems-libiberty"
    )
endif()
