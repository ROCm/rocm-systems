include_guard(GLOBAL)

if(ROCPROFILER_BUILD_NLOHMANN_JSON)
    message(STATUS "Building nlohmann/json from source!")
    execute_process(
        COMMAND
            ${CMAKE_COMMAND} -E make_directory
            ${PROJECT_BINARY_DIR}/external/nlohmann_json
    )
    # checkout submodule if not already checked out or clone repo if no .gitmodules file
    rocprofiler_systems_checkout_git_submodule(
        RELATIVE_PATH external/nlohmann_json
        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR}
        TEST_FILE CMakeLists.txt
        REPO_URL https://github.com/nlohmann/json.git
        REPO_BRANCH "v3.11.3"
    )

    include(ExternalProject)
    ExternalProject_Add(
        rocprofiler-systems-nlohmann-json-build
        PREFIX ${PROJECT_BINARY_DIR}/external/nlohmann_json/build
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/external/nlohmann_json
        BUILD_IN_SOURCE 0
        CMAKE_ARGS
            -DCMAKE_INSTALL_PREFIX=${PROJECT_BINARY_DIR}/external/nlohmann_json/install
            -DCMAKE_BUILD_TYPE=Release -DJSON_BuildTests=OFF -DJSON_Install=ON
        BUILD_COMMAND ${CMAKE_COMMAND} --build <BINARY_DIR> --config Release
        INSTALL_COMMAND
            ${CMAKE_COMMAND} --build <BINARY_DIR> --target install --config Release
    )

    target_include_directories(
        rocprofiler-systems-nlohmann-json
        SYSTEM
        INTERFACE
            $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/external/nlohmann_json/install/include>
    )
    add_dependencies(
        rocprofiler-systems-nlohmann-json
        rocprofiler-systems-nlohmann-json-build
    )
else()
    message(STATUS "Using system nlohmann/json library")
    find_package(nlohmann_json REQUIRED)
    target_link_libraries(
        rocprofiler-systems-nlohmann-json
        INTERFACE nlohmann_json::nlohmann_json
    )
endif()
