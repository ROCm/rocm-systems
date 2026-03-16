---
myst:
  html_meta:
    "description lang=en": "AMD SMI documentation and API reference."
    "keywords": "amdsmi, lib, cli, system, management, interface, amdgpu, admin, sys"
---

# AMD SMI {{ AMDSMI_VERSION }} documentation

The AMD System Management Interface (AMD SMI) library offers a unified tool for
managing and monitoring GPUs, particularly in high-performance computing
environments. It provides a user-space interface that allows applications to
control GPU operations, monitor performance, and retrieve information about the
system's drivers and GPUs.

Find the source code at <https://github.com/ROCm/rocm-systems/projects/amdsmi>.

```{important}
This AMD SMI project supports Linux bare metal and Linux virtual machine guest
environments. For documentation regarding virtualization on SR-IOV Linux hosts,
refer to the [AMD SMI for Virtualization
documentation](https://instinct.docs.amd.com/projects/amd-smi-virt/en/latest/).
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} Install
* [Install the library and CLI tool](./install/install.md)
* [Build from source](./install/build.md)
:::

:::{grid-item-card} How to
* [Use the C/C++ library](./how-to/amdsmi-cpp-lib.md)
* [Use the Python library](./how-to/amdsmi-py-lib.md)
* [Use the Go library](./how-to/amdsmi-go-lib.md)
* [Use the amd-smi CLI](./how-to/amdsmi-cli-tool.md)
* [Use AMD SMI in a Docker container](./how-to/setup-docker-container.md)
:::

:::{grid-item-card} Reference
* [C/C++ API](./reference/amdsmi-cpp-api/index.md)
* [Python API](./reference/amdsmi-py-api.md)
* [Go API](./reference/amdsmi-go-api.md)
:::

:::{grid-item-card} Conceptual
* [Performance levels and determinism](./conceptual/perf-determinism.md)
* [Reliability, availability, serviceability](./conceptual/ras.md)
:::

:::{grid-item-card} Tutorials
* [AMD SMI examples (GitHub)](https://github.com/ROCm/rocm-systems/tree/develop/projects/amdsmi/example)
* [amd-smi CLI walkthrough](https://rocm.blogs.amd.com/software-tools-optimization/amd-smi-overview/README.html)
* [GPU partitioning (MI300X)](https://instinct.docs.amd.com/projects/amdgpu-docs/en/latest/gpu-partitioning/mi300x/quick-start-guide.html)
:::
::::

```{note}
AMD SMI is the successor to <https://github.com/ROCm/rocm_smi_lib>.
```

To learn about contributing to AMD SMI, see [Contributing to AMD
SMI](https://github.com/ROCm/rocm-systems/blob/develop/projects/amdsmi/.github/CONTRIBUTING.md).
To contribute to the documentation, see {doc}`Contributing to ROCm
documentation <rocm:contribute/contributing>`.

Find ROCm licensing information on the {doc}`Licensing <rocm:about/license>`
page.
