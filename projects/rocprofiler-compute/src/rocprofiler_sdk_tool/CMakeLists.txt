if(NOT CMAKE_HIP_COMPILER)
    find_program(
        amdclangpp_EXECUTABLE
        NAMES amdclang++
        HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATHS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm
        PATH_SUFFIXES bin llvm/bin NO_CACHE)
    mark_as_advanced(amdclangpp_EXECUTABLE)

    if(amdclangpp_EXECUTABLE)
        set(CMAKE_HIP_COMPILER "${amdclangpp_EXECUTABLE}")
    endif()
endif()

find_package(rocprofiler-sdk REQUIRED
    HINTS ${ROCM_PATH} ENV ROCM_PATH /opt/rocm)
find_package(fmt REQUIRED)

add_library(rocprofiler-sdk-tool-for-compute SHARED)
target_sources(rocprofiler-sdk-tool-for-compute PRIVATE counter_collection.cpp counter_collection.hpp tmp_file.cpp tmp_file.hpp)
target_link_libraries(rocprofiler-sdk-tool-for-compute PRIVATE rocprofiler-sdk::rocprofiler-sdk)
target_link_libraries(rocprofiler-sdk-tool-for-compute INTERFACE fmt::fmt)