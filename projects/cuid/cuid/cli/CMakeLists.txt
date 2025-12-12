cmake_minimum_required(VERSION 3.10)
project(amdcuid_tool)

set(CMAKE_CXX_STANDARD 14)

# Find OpenSSL
find_package(OpenSSL REQUIRED)

# Add the tool executable
add_executable(amdcuid_tool amdcuid_tool.cc)

# Link against the CUID library
target_link_libraries(amdcuid_tool cuid_static OpenSSL::Crypto)

# Include directories
target_include_directories(amdcuid_tool PRIVATE ${CMAKE_SOURCE_DIR}/lib)

# Install the tool
install(TARGETS amdcuid_tool DESTINATION bin)
