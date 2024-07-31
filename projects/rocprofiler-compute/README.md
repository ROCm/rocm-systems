[![Ubuntu 22.04](https://github.com/ROCm/omniperf/actions/workflows/ubuntu-jammy.yml/badge.svg)](https://github.com/ROCm/omniperf/actions/workflows/ubuntu-jammy.yml)
[![RHEL 8](https://github.com/ROCm/omniperf/actions/workflows/rhel-8.yml/badge.svg)](https://github.com/ROCm/omniperf/actions/workflows/rhel-8.yml)
[![Instinct](https://github.com/ROCm/omniperf/actions/workflows/mi-rhel9.yml/badge.svg)](https://github.com/ROCm/omniperf/actions/workflows/mi-rhel9.yml)
[![Docs](https://github.com/ROCm/omniperf/actions/workflows/docs.yml/badge.svg)](https://rocm.github.io/omniperf/)
[![DOI](https://zenodo.org/badge/561919887.svg)](https://zenodo.org/badge/latestdoi/561919887)

# Omniperf

## General

Omniperf is a system performance profiling tool for machine
learning/HPC workloads running on AMD MI GPUs. The tool presently
targets usage on MI100, MI200, and MI300 accelerators.

* For more information on available features, installation steps, and
workload profiling and analysis, please refer to the online
[documentation](https://rocm.docs.amd.com/projects/omniperf/en/latest/).

* Omniperf is an AMD open source research project and is not supported
as part of the ROCm software stack. We welcome contributions and
feedback from the community. Please see the
[CONTRIBUTING.md](CONTRIBUTING.md) file for additional details on our
contribution process.

* Licensing information can be found in the [LICENSE](LICENSE) file.

## Development

Omniperf follows a
[main-dev](https://nvie.com/posts/a-successful-git-branching-model/)
branching model. As a result, our latest stable release is shipped
from the `main` branch, while new features are developed in our
`dev` branch.

Users may checkout `dev` to preview upcoming features.

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
  title        = {AMDResearch/omniperf: v2.0.1 (03 June 2024)},
  month        = june,
  year         = 2024,
  publisher    = {Zenodo},
  version      = {v2.0.1},
  doi          = {10.5281/zenodo.7314631},
  url          = {https://doi.org/10.5281/zenodo.7314631}
}
```
