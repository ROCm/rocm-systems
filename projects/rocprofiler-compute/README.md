# ROCm Compute Profiler

## General

ROCm Compute Profiler is a system performance profiling tool for machine
learning/HPC workloads running on AMD MI GPUs. The tool presently
targets usage on MI100, MI200, and MI300 accelerators.

* For more information on available features, installation steps, and
workload profiling and analysis, please refer to the online
[documentation](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/latest/).

* ROCm Compute Profiler is an AMD open source research project and is not supported
as part of the ROCm software stack. We welcome contributions and
feedback from the community. Please see the
[CONTRIBUTING.md](CONTRIBUTING.md) file for additional details on our
contribution process.

* Licensing information can be found in the [LICENSE](LICENSE.md) file.

## Development

ROCm Compute Profiler is now included in the rocm-systems super-repo. The latest sources are in the `develop` branch. You can find particular releases in the `release/rocm-rel-X.Y` branch for the paricular release you're looking for.

### Pulling the source using sparse-checkout

Being in the super-repo, if you only want to pull the source for a particular project, do a sparse checkout:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-compute
git checkout develop

cd rocprofiler-compute
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
mkdir build
ctest
```

For manual testing, you can find the executable at `install/bin/rocprof-compute`

## Standalone binary

To create a standalone binary, run the following commands:
* `cd docker`
* `docker compose -f docker-compose.standalone.yml build`
* `docker compose -f docker-compose.standalone.yml up --force-recreate -d && docker attach docker-standalone-1`

You should find the rocprof-compute.bin standalone binary inside the `build` folder in the root directory of the project.

To build the binary we follow these steps:
* Use RHEL 8.10 docker image as the base image
* Install python3.9
* Install runtime dependencies
* Install dependencies for building standalone binary
* Call the make target which uses Nuitka to build the standalone binary

NOTE: Since RHEL 8 ships with glibc version 2.28, this standalone binary can only be run on environment with glibc version greater than 2.28.
glibc version can be checked using `ldd --version` command.

NOTE: libnss3.so shared library is required when using --roof-only option which generates roofline data in PDF format

To test the standalone binary provide the `--call-binary` option to pytest.

## How to Cite

This software can be cited using a Zenodo
[DOI](https://doi.org/10.5281/zenodo.7314631) reference. A BibTex
style reference is provided below for convenience:

```
@software{xiaomin_lu_2022_7314631
  author       = {Xiaomin Lu and
                  Cole Ramos and
                  Fei Zheng and
                  Karl W. Schulz and
                  Jose Santos and
                  Keith Lowery and
                  Nicholas Curtis and
                  Cristian Di Pietrantonio},
  title        = {ROCm/rocprofiler-compute: v3.1.0 (12 February 2025)},
  month        = February,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {v3.1.0},
  doi          = {10.5281/zenodo.7314631},
  url          = {https://doi.org/10.5281/zenodo.7314631}
}
```
