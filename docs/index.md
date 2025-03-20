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
  * [Modules](../doxygen/docBin/html/modules)
  * [Files](../doxygen/docBin/html/files)
  * [Globals](../doxygen/docBin/html/globals)
  * [Data structures](../doxygen/docBin/html/annotated)
  * [Data fields](../doxygen/docBin/html/functions_data_fields)
* [Python API](./reference/amdsmi-py-api.md)
* [Go API](./reference/amdsmi-go-api.md)
:::

:::{grid-item-card} Tutorials
* [AMD SMI examples (GitHub)](https://github.com/ROCm/amdsmi/tree/amd-staging/example)
* [ROCm SMI examples (GitHub)](https://github.com/ROCm/rocm_smi_lib/tree/amd-staging/example)
:::
::::

To contribute to the documentation, refer to
{doc}`Contributing to ROCm <rocm:contribute/contributing>`.

Find ROCm licensing information on the
{doc}`Licensing <rocm:about/license>` page.

<style>
#disclaimer {
    font-size: 0.8rem;
}
</style>

<div id="disclaimer">
The information contained herein is for informational purposes only, and is
subject to change without notice. While every precaution has been taken in the
preparation of this document, it may contain technical inaccuracies, omissions
and typographical errors, and AMD is under no obligation to update or otherwise
correct this information. Advanced Micro Devices, Inc. makes no representations
or warranties with respect to the accuracy or completeness of the contents of
this document, and assumes no liability of any kind, including the implied
warranties of noninfringement, merchantability or fitness for particular
purposes, with respect to the operation or use of AMD hardware, software or
other products described herein.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced
Micro Devices, Inc. Other product names used in this publication are for
identification purposes only and may be trademarks of their respective
companies.

Copyright (c) 2014-2024 Advanced Micro Devices, Inc. All rights reserved.
</div>
