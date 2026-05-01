# run_amdsmi_build.py

One-shot AMDSMI **build → package → install → verify** runner. Auto-detects
your distro, installs prerequisites, and exits non-zero on any failure.

## Install

```bash
sudo python3 projects/amdsmi/tests/amdsmi_build/run_amdsmi_build.py --install-cli
```

Now callable from anywhere as `run_amdsmi_build`. Auto-elevates via `sudo -E`.

## Or run directly

```bash
sudo python3 projects/amdsmi/tests/amdsmi_build/run_amdsmi_build.py
```

## Features

| Flag | What it does |
|---|---|
| *(no flags)* | Autodetect distro, build, package, install, verify |
| `--install-cli` | Symlink to `/usr/local/bin/run_amdsmi_build` |
| `--no-autodetect` | Disable `/etc/os-release` parsing |
| `--skip-build` | Reuse existing `build/` directory |
| `--skip-install` | Build + package only |
| `--build-type Release\|Debug` | CMake build type |
| `--jobs N` | Parallel build jobs |
| `--retries N` | Retry transient steps |
| `--log-dir DIR` | Per-step log destination |
| `--test-results-dir DIR` | Result-file destination |
| `--os-label LABEL` | Override label in result paths |
| `--package-manager apt\|dnf\|zypper` | Force package manager |
| `--package-format deb\|rpm` | Force package format |
| `--qa-rpaths` | RHEL 10 / AlmaLinux 8 RPM builds |
| `--debian10-sources` | Rewrite apt sources for archived Debian 10 |
| `--skip-setuptools-upgrade` | Skip pip/setuptools/wheel upgrade |
| `--install-more-itertools` | Install `more_itertools` (AzureLinux 3) |

## Supported distros

Ubuntu 20 / 22 / 24 · Debian 10 · RHEL 8 / 9 / 10 · AlmaLinux 8 ·
AzureLinux 3 · SLES 15.x

Run `run_amdsmi_build --help` for the full flag list.
