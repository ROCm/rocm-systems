include(FetchContent)

if(ENABLE_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG
            52eb8108c5bdec04579160ae17225d66034bd723 # 1.17.0
    )
    FetchContent_MakeAvailable(googletest)
endif()

FetchContent_Declare(
    fmt
    GIT_REPOSITORY https://github.com/fmtlib/fmt.git
    GIT_TAG
        407c905e45ad75fc29bf0f9bb7c5c2fd3475976f # 12.1.0
)

FetchContent_MakeAvailable(fmt)
