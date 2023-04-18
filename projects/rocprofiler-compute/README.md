[![Ubuntu 20.04](https://github.com/AMDResearch/omniperf/actions/workflows/ubuntu-focal.yml/badge.svg)](https://github.com/AMDResearch/omniperf/actions/workflows/ubuntu-focal.yml)
[![RHEL 8](https://github.com/AMDResearch/omniperf/actions/workflows/opensuse.yml/badge.svg)](https://github.com/AMDResearch/omniperf/actions/workflows/opensuse.yml)
[![Docs](https://github.com/AMDResearch/omniperf/actions/workflows/docs.yml/badge.svg)](https://amdresearch.github.io/omniperf/)
[![DOI](https://zenodo.org/badge/561919887.svg)](https://zenodo.org/badge/latestdoi/561919887)


# Omniperf

## General
Omniperf is a system performance profiling tool for machine
learning/HPC workloads running on AMD MI GPUs. The tool presently
targets usage on MI100 and MI200 accelerators.

* For more information on available features, installation steps, and
workload profiling and analysis, please refer to the online
[documentation](https://amdresearch.github.io/omniperf).

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
@software{xiamin_lu_2022_7314631
  author       = {Xiaomin Lu and
                  Cole Ramos and
                  Fei Zheng and
                  Karl W. Schulz and
                  Jose Santos and
                  Keith Lowery},
  title        = {AMDResearch/omniperf: v1.0.8-PR2 (17 April 2023)},
  month        = apr,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {v1.0.8-PR1},
  doi          = {10.5281/zenodo.7314631},
  url          = {https://doi.org/10.5281/zenodo.7314631}
}
```

