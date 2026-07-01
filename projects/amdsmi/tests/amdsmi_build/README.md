# run_amdsmi_build.py

One-shot AMDSMI **build → package → install → verify** runner. Auto-detects
your distro, installs prerequisites, and exits non-zero on any failure.

Used by `.github/workflows/amdsmi-build.yml` in CI; also runnable locally.

## Usage

Run as root (or via `sudo`) — the build/install/ldconfig stages require it:

```bash
sudo python3 projects/amdsmi/tests/amdsmi_build/run_amdsmi_build.py
```

The script auto-detects the distro from `/etc/os-release`. To force settings
(e.g. in CI), pass them explicitly:

```bash
sudo python3 projects/amdsmi/tests/amdsmi_build/run_amdsmi_build.py \
    --os-label Ubuntu22 \
    --package-manager apt \
    --package-format deb
```

## Output

| Path | Contents |
|---|---|
| `--log-dir` (default `logs/amdsmi/`) | Per-step command logs |
| `--test-results-dir` (default `/tmp/test-results-<os-label>/`) | `build_result.txt`, `install_result.txt`, `verify_wheel_result.txt` consumed by the CI summary |

## Features

| Flag | What it does |
|---|---|
| *(no flags)* | Autodetect distro, build, package, install, verify |
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
| `summarize <results-dir>` | (subcommand) Render the CI step summary |

## Supported distros

Ubuntu 20 / 22 / 24 · Debian 10 · RHEL 8 / 9 / 10 · AlmaLinux 8 ·
AzureLinux 3 · SLES 15.x

Run with `--help` for the full flag list.
