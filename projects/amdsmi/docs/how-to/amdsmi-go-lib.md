---
myst:
  html_meta:
    "description lang=en": "Get started with the AMD SMI Go interface."
    "keywords": "api, smi, lib, go, golang, system, management, interface, ROCm"
---

# AMD SMI Go interface overview

The AMD SMI Go interface provides a convenient way to interact with AMD
hardware through a simple and accessible [API](../reference/amdsmi-go-api.md).
The API is compatible with Go 1.20 and higher and requires the AMD driver to
be loaded for initialization. Review the [prerequisites](#install_reqs).

```{seealso}
Refer to the [Go library API reference](../reference/amdsmi-go-api.md).
```

(go_prereqs)=
## Prerequisites

Before get started, make sure your environment satisfies the following prerequisites.
See the [requirements](#install_reqs) section for more information.

1. Ensure `amdgpu` drivers are installed properly for initialization.

2. Export `LD_LIBRARY_PATH` to the `amdsmi` installation directory.

   ```bash
   export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/rocm/lib:/opt/rocm/lib64:
   ```

3. Install Go 1.20+.

   Download Go from [https://go.dev/dl/](https://go.dev/dl/) and follow the
   official installation documentation at [Download and
   install](https://go.dev/doc/install).

   Alternatively, use a third-party utility like update-golang.

   ```bash
   git clone https://github.com/udhos/update-golang
   cd update-golang
   sudo ./update-golang.sh
   source /etc/profile.d/golang_path.sh
   go version
   ```

## Get started

```{note}
``hipcc`` and other compilers will not automatically link in the ``libamd_smi``
dynamic library. To compile code that uses the AMD SMI library API, ensure the
``libamd_smi.so`` can be located by setting the ``LD_LIBRARY_PATH`` environment
variable to the directory containing ``librocm_smi64.so`` (usually
``/opt/rocm/lib``) or by passing the ``-lamd_smi`` flag to the compiler.
```

A Go application using AMD SMI must call `goamdsmi.GO_gpu_init()` to initialize
the AMI SMI library before all other calls. This call initializes the internal
data structures required for subsequent AMD SMI operations.

`goamdsmi.GO_gpu_shutdown()` must be the last call to properly close connection to
driver and make sure that any resources held by AMD SMI are released.

## Usage

For an example on using the AMD SMI Go API, refer to this implementation
[https://github.com/amd/amd_smi_exporter/tree/master](https://github.com/amd/amd_smi_exporter/tree/master).

```{seealso}
Refer to the [Go library API reference](../reference/amdsmi-go-api.md).
```

### Add AMD SMI library to your project

To include the AMD SMI Go API in your project, update your Makefile or Go module configuration
to fetch the appropriate version of the AMD SMI library.

```shell
go get github.com/ROCm/amdsmi@amd-staging 
```

When using a Makefile, ensure you're fetching the latest AMD SMI repository
with Go API support. See
[https://github.com/amd/amd_smi_exporter/blob/master/src/Makefile](https://github.com/amd/amd_smi_exporter/blob/master/src/Makefile)
for an example implementation.
