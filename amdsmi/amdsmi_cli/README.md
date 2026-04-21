# AMD SMI CLI tool for WSL

`amd-smi` is the command-line interface for the WSL-focused AMD SMI build in
this repository. It provides GPU monitoring for supported AMD GPU setups
running on Windows Subsystem for Linux.

This CLI is part of the [AMD SMI WSL component](../README.md).

> [!NOTE]
> This repository contains a WSL adaptation of AMD SMI. The CLI behavior and
> feature availability can differ from upstream Linux deployments depending on
> what is supported by the WSL stack and Windows driver.

## Prerequisites

- A supported AMD GPU exposed to WSL
- AMD ROCm installed in WSL
- The matching AMD Windows graphics driver installed on the host
- Python 3.6.8 or newer to run the CLI from source or from the installed Python package

## Build and install

Build this CLI from the `amdsmi` directory as part of the `amdsmi` component.
Use the build instructions in [../README.md](../README.md#building).

After install, the `amd-smi` executable is available in the installed ROCm
environment.

If you install with `make install`, the Python module may also need to be
registered manually as described in [../README.md](../README.md).

## Usage

Basic examples:

```bash
amd-smi
amd-smi version
amd-smi static
```

Run `amd-smi --help` to see the full command set supported by this build.

> [!NOTE]
> The CLI is useful for interactive inspection and automation. For integration
> into larger tools or services, prefer the AMD SMI C++ or Python APIs for
> telemetry and data collection.

## Online documentation

For the upstream AMD SMI documentation set, see:

- [AMD SMI documentation portal](https://rocm.docs.amd.com/projects/amdsmi/en/latest/)
- [AMD SMI install guide](https://rocm.docs.amd.com/projects/amdsmi/en/latest/install/install.html)
- [AMD SMI CLI reference](https://rocm.docs.amd.com/projects/amdsmi/en/latest/how-to/amdsmi-cli-tool.html)

Use those pages as the baseline reference for commands and concepts, with the
understanding that some upstream capabilities may not apply to the WSL
adaptation in this repository.

