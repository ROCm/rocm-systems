include_guard(GLOBAL)

if(ROCPROFSYS_BUILD_NLOHMANN_JSON)
    message(STATUS "Building nlohmann/json from source")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3
        SOURCE_DIR
        ${PROJECT_BINARY_DIR}/external/nlohmann/src
        BINARY_DIR
        ${PROJECT_BINARY_DIR}/external/nlohmann/lib
        SUBBUILD_DIR
        ${PROJECT_BINARY_DIR}/external/nlohmann/subdir
    )
    FetchContent_MakeAvailable(nlohmann_json)

    target_include_directories(
        rocprofiler-systems-json
        SYSTEM
        INTERFACE $<TARGET_PROPERTY:nlohmann_json,INTERFACE_INCLUDE_DIRECTORIES>
    )
    target_link_libraries(rocprofiler-systems-json INTERFACE nlohmann_json)
else()
    message(STATUS "Using system nlohmann/json library")
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(rocprofiler-systems-json INTERFACE nlohmann_json::nlohmann_json)
endif()
