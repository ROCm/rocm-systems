# include guard
include_guard(DIRECTORY)

# Default: Simply add the subdirectory, this will use their default HEADER_ONLY build

# Give option to use lib that can be downloaded from package manager?

rocprofiler_systems_checkout_git_submodule(
    RELATIVE_PATH external/spdlog
    WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
    TEST_FILE INSTALL
    REPO_URL https://github.com/gabime/spdlog.git
    REPO_BRANCH "v1.16.0" # As of writing, this is thie highest version. Should this be master? Maybe...
)

# TODO: Figure this out
if(NOT DEFINED ROCPROFSYS_USE_SPDLOG)
    set(ROCPROFSYS_USE_SPDLOG TRUE)
    set(ROCPROFSYS_USE_SPDLOG_HEADER FALSE)
endif()

#TODO: It spawns the built lib in external/spdlog, is this where we want it?

if(ROCPROFSYS_USE_SPDLOG) # Recommended mode
    message(STATUS "Building spdlog as shared library from submodule")
    set(SPDLOG_COMPILED_LIB ON)
    set(SPDLOG_BUILD_SHARED ON)
    add_subdirectory(external/spdlog)
    target_link_libraries(rocprofiler-systems-spdlog INTERFACE spdlog::spdlog)

elseif(ROCPROFSYS_USE_SPDLOG_HEADER)
    message(STATUS "Using spdlog header-only from submodule")
    set(SPDLOG_COMPILED_LIB OFF)
    add_subdirectory(external/spdlog)
    target_link_libraries(rocprofiler-systems-spdlog INTERFACE spdlog::spdlog_header_only)

else()
    # TODO: Will this fail as I force the spdlog library to be an interface is source/lib/CMakeLists.txt and source/lib/core?

    message(STATUS "Using system spdlog library") #libspdlog.so
    find_package(spdlog REQUIRED)
    target_link_libraries(rocprofiler-systems-spdlog INTERFACE spdlog::spdlog)
endif()

