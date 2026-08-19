# Building standalone binary
## Create standalone binary using docker container

This method uses the cmake target inside a RHEL 8 docker container with Python3.11 installed.

To create a standalone binary, run the following commands:
* `cd docker`
* Optionally, provide `--build-arg STANDALONEBINARY_EXTRACT_DIR=/<path>` option in build container command to change the absolute path where standalone binary will extract its contents. This option should be specified after the `build` keyword. Default is `/tmp`.
* `docker compose -f docker-compose.standalone.yml build` (build container command)
* `docker compose -f docker-compose.standalone.yml up --force-recreate -d && docker attach docker-standalone-1` (run container and attach to see its output)

## Create standalone binary using cmake target locally without docker

**NOTE: Python3.11 should be installed on the system to build the standalone binary**

To create a standalone binary, run the following commands:

* Optionally, provide `-D STANDALONEBINARY_EXTRACT_DIR=/<path>` option in cmake config. command to change the absolute path where standalone binary will extract its contents. Default is `/tmp`.
* `cmake -B build -D CMAKE_INSTALL_PREFIX=install -D STANDALONEBINARY=ON -S .` (cmake config. command)
* `cmake --build build --target install --parallel 8` (run cmake install target)

## Standalone binary creation methodology

To build the binary we follow these steps:
* Use RHEL 8.10 docker image as the base image (only in docker method)
* Install python3.11
* Install python dependencies
* Call the install cmake target with STANDALONEBINARY=ON cmake args. which will use Nuitka to build the standalone binary

You should find the rocprof-compute.bin standalone binary inside the `install/libexec/rocprofiler-compute/rocprof-compute.bin` folder in the root directory of the project.

## Things to note about standalone binary

* [Nuitka](https://nuitka.net/user-documentation/) is used for compiling the python interpreter, python dependencies and source code into C and then to a executable. The whole process takes about 30 minutes. The self-extracting standalone binary itself is approximately 150 MB in size, however, the total size of the extracted compiled artifacts is approximately 650 MB.

* By default, standalone binary extracts its contents to a directory `rocprof_compute_standalonebinary_<pid>` under `/tmp` parent directory upon execution, however, the parent directory can be configured as explained in standalone binary creation section.

* When using docker method, since RHEL 8 ships with glibc version 2.28, this standalone binary can only be run on environment with glibc version greater than or equal to 2.28. glibc version can be checked using `ldd --version` command.

* If not using docker, the minimum glibc version is determined by the OS where cmake is run.

* When using docker, native counter collection tool is not compiled due to unavailability of rocprofiler-sdk. Instead, native counter collection tool will be runtime compiled based on the environment where the binary is running.

## Test standalone binary

Create standalone binary with tests enabled, then run the tests:

* `cmake -B build -D CMAKE_INSTALL_PREFIX=install -D ENABLE_TESTS=ON -D INSTALL_TESTS=ON -D STANDALONEBINARY=ON -S .`
* `cmake --build build --target install --parallel 8`
* `cd install/libexec/rocprofiler-compute`
* `ctest`
