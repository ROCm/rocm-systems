if(WIN32)
    set(CMAKE_AR "${ROCM_PATH}/../lc/bin/llvm-ar" CACHE FILEPATH "Archiver for Clang" FORCE)
endif()
message("**********THEROCK_BUILD_MODE - ${THEROCK_BUILD_MODE}*****")   
if(NOT THEROCK_BUILD_MODE)
    # Setup catch2 manually
    set(CATCH2_DOWNLOAD_URL "https://rocm-third-party-deps.s3.us-east-2.amazonaws.com/Catch2-3.8.1.tar.gz")
    include(FetchContent)
    FetchContent_Declare(
        Catch2
        URL ${CATCH2_DOWNLOAD_URL}
        #GIT_REPOSITORY https://github.com/catchorg/Catch2.git
        #GIT_TAG        v3.8.1
        DOWNLOAD_EXTRACT_TIMESTAMP OFF
        URL_HASH SHA256=18b3f70ac80fccc340d8c6ff0f339b2ae64944782f8d2fca2bd705cf47cadb79
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}
        FIND_PACKAGE_ARGS
    )
    if (NOT Catch2_POPULATED)
        FetchContent_MakeAvailable(Catch2)
    else()
        message("**********catch2 downloaded already******")
    endif()

    if(WIN32)
        target_link_options(Catch2 PRIVATE LINKER:--rsp-quoting=windows)
    endif()
    list(APPEND CMAKE_MODULE_PATH ${Catch2_SOURCE_DIR}/extras)
    include(Catch)

else()
    message("**********Find CATCH2 ******")
    list(APPEND CMAKE_PREFIX_PATH 
         "C:\\workspace\\staging_10_14\\drivers\\drivers\\compute\\build\\native\\Release\\x64\\hip\\catch_off\\external\\Catch2\\extras")
    
    # TheRock has already setup Catch2 for us
    find_package(Catch2 REQUIRED)
endif()
