cmake_minimum_required(VERSION 3.10)
project(amdcuid)

set(CMAKE_CXX_STANDARD 14)

find_package(OpenSSL REQUIRED)

file(GLOB CLI_SOURCES *.cc)

add_executable(amdcuid ${CLI_SOURCES})

target_include_directories(amdcuid PRIVATE ${CMAKE_SOURCE_DIR}/../lib)
target_link_libraries(amdcuid PRIVATE cuid_static OpenSSL::Crypto pthread)
