# Building docker image
Populate the <username> variable in `docker/docker-compose.therock.tarball.yml`.
Populate the <tarball_name> variable in `docker/Dockerfile.therock.tarball` (see the comment there for the naming convention and nightly index).

To quickly get the environment (bash shell) for building and testing, run the following commands:
* `cd docker`
* If the docker image is not available on the machine, then build the image, otherwise skip this step: `docker compose -f docker-compose.therock.tarball.yml build`
* Launch the container, and check the name of the container: `docker compose -f docker-compose.therock.tarball.yml up --force-recreate -d `
* Run bash shell on the launched container: `docker exec -it <container_name> bash`
* If testing is done, kill the container: `docker container kill <container_name>`

Inside the docker container, clean, build, then install the project with tests enabled:
```
rm -rf build install && cmake -B build -D CMAKE_INSTALL_PREFIX=install -D ENABLE_TESTS=ON -D INSTALL_TESTS=ON -D ENABLE_COVERAGE=ON -S . && cmake --build build --target install --parallel 8
```

Common CMake options:
- `-D ENABLE_TESTS=ON` - Enable building test executables
- `-D INSTALL_TESTS=ON` - Install test files and test suite
- `-D ENABLE_COVERAGE=ON` - Enable code coverage reporting
- `-D TEST_FROM_INSTALL=ON` - Enable testing from installation directory instead of build directory
- `-D SKIP_NATIVE_TOOL_BUILD=ON` - Skip building the native profiling tool (enables runtime compilation instead), useful when rocprofiler-sdk is not available during build time
- `-D TORCH_TRACE_PYTHON=/path/to/python3` - Select the Python interpreter for the `roctx_recordfn` build when `ENABLE_TESTS=ON`
- `-D ENABLE_SANITIZER=ASAN|HOST_ASAN|TSAN` - Build with sanitizer instrumentation for development (default OFF); cannot be combined with `STANDALONEBINARY=ON`

Note that per the above command, build assets will be stored under `build` directory and installed assets will be stored under `install` directory.

Then, to run the automated test suite, run the following commands:
```
cd build
ctest
```

For manual testing, you can find the executable at `install/bin/rocprof-compute`