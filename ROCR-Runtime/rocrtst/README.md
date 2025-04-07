# Building rocrtst

## Library dependencies
rocrtst needs hwloc and libnuma to build and run. On Debian systems, for example, you would need to get them like so:
```sh
sudo apt-get install libhwloc-dev libnuma-dev
```
## CMake option values
When building rocrtst, several cmake command line options are available--some mandatory, some optional. These are described here:
  * TARGET_DEVICES=<string>
    * Optional
    * semi-colon separated list of gpus to build kernels for; e.g. "gfx908;gfx900;...".
    * Default: the list of devices that is used is specified in the CMakeLists.txt file, and includes the all the currently supported targets.
  * ROCRTST_BLD_TYPE=<debug|release>
    * Optional
    * Build a debug or release build
    * Default: Build the debug version
  * CMAKE_PREFIX_PATH=<"ROCR root path; LLVM root path">
    * Required
    * Where to find ROCr and LLVM. The ROCr root path is typically something like /opt/rocm. The LLVM directory is typically something like /opt/rocm/llvm
  * CMAKE_INSTALL_PREFIX="<Root path where rocrtst should be installed>"
    * Optional
    * Where to install rocrtst
  * CPACK_PACKAGING_INSTALL_PREFIX="<path where to install>"
    * Optional
    * Where to install rocrtst within DEB/RPM packages
  * CPACK_GENERATOR=<list of package generators>
    * Optional
    * List of CPack build generators to use; e.g. "DEB;RPM"
  * ROCM_PATCH_VERSION=<string>
    * Optional
    * ROCm patch version used in package name
  * ROCM_DIR=<ROCm path>
    * Required
    * ROCm root directory
  * LLVM_DIR="<clang location>"
    * Required
    * Location of clang executable
  * OPENCL_DIR=<location of OpenCL root>
    * Required
    * Location where OpenCL root resides
  * EMULATOR_BUILD=<true|false>
    * Optional
    * If EMULATOR_BUILD is defined, rocrtst will avoid tests that typically run too long on an HW emulator, or use a scaled-down version of the test.

## Steps to build
```sh
mkdir build
cd build
# See description of these options above.
# The values for these options are examples. They should be tailored
# for your system.
cmake -DTARGET_DEVICES=$GPU_LIST \
  -DROCRTST_BLD_TYPE=$ROCRTST_BUILD_TYPE \
  -DCMAKE_PREFIX_PATH="$PACKAGE_ROOT;$PACKAGE_ROOT/llvm" \
  -DCMAKE_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
  -DCPACK_PACKAGING_INSTALL_PREFIX="$ROCM_INSTALL_PATH" \
  -DCPACK_GENERATOR="DEB;RPM" \
  -DROCM_PATCH_VERSION=$ROCM_LIBPATCH_VERSION \
  -DROCM_DIR=$PACKAGE_ROOT \
  -DLLVM_DIR="$PACKAGE_ROOT/llvm/bin" \
  -DOPENCL_DIR=$PACKAGE_ROOT \
  -DEMULATOR_BUILD=$EMULATOR_BUILD \
      ..
# Build rocrtst executable
make
# Build rocrtst kernels
make rocrtst_kernels
```
## Running rocrtst
rocrtst needs to be able to find the ROCr library. This can be through ldconfig method or by setting LD_LIBRARY_PATH to have the ROCr library directory.
When rocrtst is built, there is one rocrtst executable, and several symlinks pointing to that executable, one from each asic sub-directory. For example, for gfx900, we would see the following:
```sh
cd <rocrtst bin root>/gfx900
ls -l rocrtst
lrwxrwxrwx 1 user user 12 Sep 28 17:23 rocrtst64 -> ../rocrtst64
```
To run rocrtst, we should call the ASIC specific symlink. This allows the asic-specific kernels to be found.

rocrtst is a Google Test ("gtest") based program and accepts gtest options. Additionally, there are some rocrtst specfic options. All of these options can be seen by using the "-h" option:
```sh
$ <rocrtst bin>/gfx900 $ ./rocrtst64 -h
<GTest option descrption>
Optional RocRTst Arguments:
--iterations, -i <number of iterations to execute>; override default, which varies for each test
--rocrtst_help, -r print this help message
--verbosity, -v <verbosity level>
  Verbosity levels:
   0    -- minimal; just summary information
   1    -- intermediate; show intermediate values such as intermediate perf. data
   2    -- progress; show progress displays
   >= 3 -- more debug output
--monitor_verbosity, -m <monitor verbosity level>
  Monitor Verbosity levels:
   0    -- don't read or print out any GPU monitor information;
   1    -- print out all available monitor information before the first test and after each test
   >= 2 -- print out even more monitor information (test specific)

```


