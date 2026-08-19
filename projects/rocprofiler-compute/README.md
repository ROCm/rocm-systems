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

Being in the super-repo, if you only want to pull the source for a particular project, do a sparse checkout:

```bash
git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
cd rocm-systems
git sparse-checkout init --cone
git sparse-checkout set projects/rocprofiler-compute
git checkout develop

cd projects/rocprofiler-compute

# Initialize submodule dependencies (vendored Python deps and src/lib/external C++ libs)
git submodule update --init --recursive -- src/

python3 -m pip install -r requirements.txt
```

**Note**: When working from source, submodules live under `src/` (vendored Python dependencies like PyYAML in `src/vendored/`, and C++ libraries like googletest, fmt, and json in `src/lib/external/`). If you see import errors about missing vendored modules or missing C++ externals during a build, run `git submodule update --init --recursive -- src/`.

To install development tools (linter, pre-commit hooks, YAML utilities), run:

```bash
python3 -m pip install -r requirements-development.txt
```

## Testing
For testing purposes user one of the following ways:
- [Building Docker Container](./docs/dev-docs/building-docker-image.md) (preferred).
- [Buidling Standalone Binary](./docs/dev-docs/building-standalone-binary.md).

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
