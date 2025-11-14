
===================> Standalone 

rocm-systems/projects/hip-tests/v1_StandaloneApp
sh buildAll.sh


===================> Standalone with cmake 

rocm-systems/projects/hip-tests/v2_StandaloneApp_cmake

rm -rf build
mkdir build
cd build
cmake ../
make 

./app


===================> Standalone with cmake and ctest

rocm-systems/projects/hip-tests/v3_StandaloneApp_ctest_binary

rm -rf build
mkdir build
cd build
cmake ../
make 

ctest -R SampleAppTest --verbose


===================> Standalone with ctest & gtest

rocm-systems/projects/hip-tests/v4_StandaloneApp_ctest_Gtest

rm -rf build
mkdir build
cd build
cmake ../
make

ctest -R LibraryManagementTests --verbose

ctest LibraryManagementTests.MultipleLibs

=================> in the hip-test samples

rocm-systems/projects/hip-tests

rm -rf build
mkdir build
cd build
cmake ../samples/
make build_samples 
or make build_libMangment


cd 3_Library_Management/
./libMangnt



