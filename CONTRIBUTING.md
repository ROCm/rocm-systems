# Contributing to AMD ROCDXG Libary

Thanks for your interest in improving librocdxg! This guide explains how to set up development environment, propose changes, and submit pull requests. 

## Quick Start
- Fork the repo and create a feature branch.
- Build and test inside WSL2 (Ubuntu 22.04/24.04).
- Keep PRs focused and well-described; link related issues.
- Ensure changes align with license terms (see Licensing).

## Issues and Discussions
- Before filing an issue, search existing issues.
- Bug reports should include:
  - Host Windows version, WSL Ubuntu version, compiler, and CMake versions
  - Windows display driver version and ROCm version
  - Exact build commands and `WIN_SDK` path used
  - Minimal repro steps and expected vs. actual behavior
  - Logs from relevant modules
- Feature requests should explain use-cases and expected API changes.

## Licensing
- Most source files are licensed under MIT as described in [LICENSE.md](LICENSE.md).
- The binary `src/thunk_proxy/libthunk_proxy.a` is under an AMD proprietary license with redistribution restrictions (see [LICENSE.md](LICENSE.md)). Do not attempt to reverse engineer or modify this binary.
- By submitting a contribution, you agree that your changes are licensed under the MIT terms applicable to this repository, unless explicitly stated otherwise by the maintainers.
