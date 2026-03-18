# ROCm Compute Profiler

## General

ROCm Compute Profiler is a system performance profiling tool for machine
learning/HPC workloads running on AMD MI GPUs. The tool presently
targets usage on MI100, MI200, MI300, and MI350 series accelerators.

* For more information on available features, installation steps, and
workload profiling and analysis, please refer to the online
[documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

* ROCm Compute Profiler is an AMD open source tool that is part of the ROCm software stack. We welcome contributions and
feedback from the community. Please see the
[CONTRIBUTING.md](CONTRIBUTING.md) file for additional details on our
contribution process.

* Licensing information can be found in the [LICENSE](LICENSE.md) file.

## Development

ROCm Compute Profiler is now included in the rocm-systems super-repo. The latest sources are in the `develop` branch. You can find particular releases in the `release/rocm-rel-X.Y` branch for the particular release you're looking for.

### Pulling the source using sparse-checkout

Being in the super-repo, if you only want to pull the source for a particular project, do a sparse checkout:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-compute
git checkout develop

cd projects/rocprofiler-compute
python3 -m pip install -r requirements.txt
```

## Testing

Populate the <usename> variable in `docker/docker-compose.customrocmtest.yml`.
Populate the <rocm_build_image> variable in `docker/Dockerfile.customrocmtest` based on latest ROCm CI build information.

To quickly get the environment (bash shell) for building and testing, run the following commands:
* `cd docker`
* If the docker image is not available on the machine, then build the image, otherwise skip this step: `docker compose -f docker-compose.customrocmtest.yml build`
* Launch the container, and check the name of the container: `docker compose -f docker-compose.customrocmtest.yml up --force-recreate -d `
* Run bash shell on the launched container: `docker exec -it <container_name> bash`
* If testing is done, kill the container: `docker container kill <container_name>`

Inside the docker container, clean, build, then install the project with tests enabled:
```
rm -rf build install && cmake -B build -D CMAKE_INSTALL_PREFIX=install -D ENABLE_TESTS=ON -D INSTALL_TESTS=ON -DENABLE_COVERAGE=ON -S . && cmake --build build --target install --parallel 8
```

Note that per the above command, build assets will be stored under `build` directory and installed assets will be stored under `install` directory.

Then, to run the automated test suite, run the following commands:
```
cd build
ctest
```

For manual testing, you can find the executable at `install/bin/rocprof-compute`

## Standalone binary

### Create standalone binary using docker container

This method uses the cmake target inside a RHEL 8 docker container with Python3.11 installed.

To create a standalone binary, run the following commands:
* `cd docker`
* Optionally, provide `--build-arg STANDALONEBINARY_EXTRACT_DIR=/<path>` option in build container command to change the absolute path where standalone binary will extract its contents. This option should be specified after the `build` keyword. Default is `/tmp`.
* `docker compose -f docker-compose.standalone.yml build` (build container command)
* `docker compose -f docker-compose.standalone.yml up --force-recreate -d && docker attach docker-standalone-1` (run container and attach to see its output)

### Create standalone binary using cmake target locally without docker

**NOTE: Python3.11 should be installed on the system to build the standalone binary**

To create a standalone binary, run the following commands:

* Optionally, provide `-D STANDALONEBINARY_EXTRACT_DIR=/<path>` option in cmake config. command to change the absolute path where standalone binary will extract its contents. Default is `/tmp`.
* `cmake -B build -D CMAKE_INSTALL_PREFIX=install -D STANDALONEBINARY=ON -S .` (cmake config. command)
* `cmake --build build --target install --parallel 8` (run cmake install target)

### Standalone binary creation methodology

To build the binary we follow these steps:
* Use RHEL 8.10 docker image as the base image (only in docker method)
* Install python3.11
* Install python dependencies
* Call the install cmake target with STANDALONEBINARY=ON cmake args. which will use Nuitka to build the standalone binary

You should find the rocprof-compute.bin standalone binary inside the `install/libexec/rocprofiler-compute/rocprof-compute.bin` folder in the root directory of the project.

### Things to note about standalone binary

* [Nuitka](https://nuitka.net/user-documentation/) is used for compiling the python interpreter, python dependencies and source code into C and then to a executable. The whole process takes about 30 minutes. The self-extracting standalone binary itself is approximately 150 MB in size, however, the total size of the extracted compiled artifacts is approximately 650 MB.

* By default, standalone binary extracts its contents to a directory `rocprof_compute_standalonebinary_<pid>` under `/tmp` parent directory upon execution, however, the parent directory can be configured as explained in standalone binary creation section.

* When using docker method, since RHEL 8 ships with glibc version 2.28, this standalone binary can only be run on environment with glibc version greater than or equal to 2.28. glibc version can be checked using `ldd --version` command.

* If not using docker, the minimum glibc version is determined by the OS where cmake is run.

* When using docker, native counter collection tool is not compiled due to unavailability of rocprofiler-sdk. Instead, native counter collection tool will be runtime compiled based on the environment where the binary is running.

### Test standalone binary

Create standalone binary with tests enabled, then run the tests:

* `cmake -B build -D CMAKE_INSTALL_PREFIX=install -D ENABLE_TESTS=ON -D INSTALL_TESTS=ON -D STANDALONEBINARY=ON -S .`
* `cmake --build build --target install --parallel 8`
* `cd install/libexec/rocprofiler-compute`
* `ctest`

## How to Cite

This software can be cited using a Zenodo
[DOI](https://doi.org/10.5281/zenodo.7314631) reference. A BibTex
style reference is provided below for convenience:

```
@misc{xiaomin_lu_2022_7314631
  author       = {Xiaomin Lu and
                  Cole Ramos and
                  Fei Zheng and
                  Karl W. Schulz and
                  Jose Santos and
                  Keith Lowery and
                  Nicholas Curtis and
                  Cristian Di Pietrantonio},
  title        = {rocprofiler-compute},
  url          = {https://github.com/ROCm/rocm-systems/blob/develop/projects/rocprofiler-compute}
}
```
