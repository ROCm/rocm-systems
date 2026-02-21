# CMake generated Testfile for 
# Source directory: D:/jam/temp/TheRock/rocm-systems/projects/smi-remote-client
# Build directory: D:/jam/temp/TheRock/rocm-systems/projects/smi-remote-client/build
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test([=[smi_remote_basic]=] "D:/jam/temp/TheRock/rocm-systems/projects/smi-remote-client/build/test_smi_basic.exe")
set_tests_properties([=[smi_remote_basic]=] PROPERTIES  ENVIRONMENT "TF_DEBUG=1" LABELS "integration" _BACKTRACE_TRIPLES "D:/jam/temp/TheRock/rocm-systems/projects/smi-remote-client/CMakeLists.txt;127;add_test;D:/jam/temp/TheRock/rocm-systems/projects/smi-remote-client/CMakeLists.txt;0;")
