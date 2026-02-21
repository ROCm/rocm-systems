# CMake generated Testfile for 
# Source directory: D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client
# Build directory: D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[hip_remote_basic]=] "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_basic.exe")
set_tests_properties([=[hip_remote_basic]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;142;add_test;D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_extended]=] "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_extended.exe")
set_tests_properties([=[hip_remote_extended]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;159;add_test;D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_phase2]=] "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_phase2.exe")
set_tests_properties([=[hip_remote_phase2]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;175;add_test;D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
add_test([=[hip_remote_graphs]=] "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/build/hip_remote_test_graphs.exe")
set_tests_properties([=[hip_remote_graphs]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;191;add_test;D:/jam/temp/TheRock/rocm-systems/projects/hip-remote-client/CMakeLists.txt;0;")
