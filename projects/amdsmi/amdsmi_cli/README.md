# AMD SMI CLI tool

A command line tool for manipulating and monitoring the `amdgpu` kernel;
`amd-smi` is intended to replace and deprecate the existing
[`rocm-smi`](https://github.com/ROCm/rocm-systems/tree/develop/projects/rocm-smi-lib) CLI tool.

When using the CLI tool, you should have at least one AMD GPU and the driver
installed.

>[!NOTE]
>The AMD SMI CLI tool is provided as an example code to aid the development of
>telemetry tools. The Python or C++ library is recommended as a robust data
>source.

Find the documentation in the `docs/` directory.

- [Install AMD SMI](../docs/install/install.md)
- [About the tool and how to get started](../docs/how-to/amdsmi-cli-tool.md)

## Argument syntax

Multi-value options accept space-separated values or comma-separated values with `=`.
For example, `amd-smi metric --gpu 0 1` and `amd-smi metric --gpu=0,1` select the same GPUs.
The short form `-g=0,1` also works. Do not put spaces around commas or mix trailing
space-separated values into an equals list. Repeating a device option keeps its last selection.
Scalar options retain their existing syntax, including commas in file paths.

## Online documentation

Explore the latest documentation on the [ROCm documentation
portal](https://rocm.docs.amd.com/projects/amdsmi/en/latest/index.html).

- [Install AMD SMI](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)

- [CLI tool usage](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html).

