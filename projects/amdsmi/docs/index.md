---
myst:
  html_meta:
    "description lang=en": "AMD SMI documentation and API reference."
    "keywords": "amdsmi, lib, cli, system, management, interface, amdgpu, admin, sys"
---

# AMD SMI documentation

The AMD System Management Interface (AMD SMI) library offers a unified tool for
managing and monitoring GPUs, particularly in high-performance computing
environments. It provides a user-space interface that allows applications to
control GPU operations, monitor performance, and retrieve information about the
system's drivers and GPUs.

Find the source code at <https://github.com/ROCm/amdsmi>.

```{note}
AMD SMI is the successor to <https://github.com/ROCm/rocm_smi_lib>.
```

::::{grid} 2
:gutter: 3

:::{grid-item-card} Install
* [Library and CLI tool installation](./install/install.md)
* [Build from source](./install/build.md)
:::

:::{grid-item-card} How to
* [C++ library usage](./how-to/amdsmi-cpp-lib.md)
* [Python library usage](./how-to/amdsmi-py-lib.md)
* [Go library usage](./how-to/amdsmi-go-lib.md)
* [CLI tool usage](./how-to/amdsmi-cli-tool.md)
* [Use AMD SMI in a Docker container](./how-to/setup-docker-container.md)
:::

:::{grid-item-card} Reference
* [C++ API](./reference/amdsmi-cpp-api.md)
  * [Modules](../doxygen/docBin/html/topics)
  * [Files](../doxygen/docBin/html/files)
  * [Globals](../doxygen/docBin/html/globals)
  * [Data structures](../doxygen/docBin/html/annotated)
  * [Data fields](../doxygen/docBin/html/functions_data_fields)
* [Python API](./reference/amdsmi-py-api.md)
* [Go API](./reference/amdsmi-go-api.md)
:::

:::{grid-item-card} Conceptual
* [Reliability, availability, serviceability](./conceptual/ras.md)
:::

:::{grid-item-card} Tutorials
* [AMD SMI examples (GitHub)](https://github.com/ROCm/amdsmi/tree/amd-staging/example)
* [AMD SMI CLI walkthrough](https://rocm.blogs.amd.com/software-tools-optimization/amd-smi-overview/README.html)
:::
::::

To learn about contributing to AMD SMI, see [Contibuting to AMD
SMI](https://github.com/ROCm/amdsmi/blob/amd-mainline/.github/CONTRIBUTING.md).
To contribute to the documentation, see
{doc}`Contributing to ROCm documentation <rocm:contribute/contributing>`.

Find ROCm licensing information on the
{doc}`Licensing <rocm:about/license>` page.
