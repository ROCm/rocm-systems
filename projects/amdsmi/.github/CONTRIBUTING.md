# Contributing to AMD SMI #

We welcome contributions to AMD SMI.
Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible, including script output so
    we can collect information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

## Acceptance Criteria ##

The goal of AMD SMI project is to provide a simple CLI interface and a library
for interacting with AMD GPUs.

## Coding Style ##

We use [pre-commit](https://pre-commit.com/) hooks to enforce formatting.
Install and run with:

```bash
pip install pre-commit
pre-commit install
pre-commit run --files ./**/*
```

### C/C++ ###

Formatted with **clang-format** (Google style, 100 character line limit).
See `.clang-format` for the full configuration.

You can also format manually: `clang-format -i <path-to-source-file>`

### Python ###

Formatted and linted with **Ruff**. Configuration is in `pyproject.toml`.

You can also run manually:

```bash
ruff check --fix .   # lint and auto-fix
ruff format .        # format
```

### CMake ###

Formatted with **gersemi** (cmake-format replacement).

## Pull Request Guidelines ##

When you create a pull request, you should target the default branch. Our
current default branch is the **develop** branch, which serves as our
integration branch.

### Deliverables ###

Every AMD-owned source file must start with the two-line SPDX header, using the
file's comment leader (`//` for C/C++/Go/Rust, `#` for Python/CMake/shell):

    // Copyright Advanced Micro Devices, Inc.
    // SPDX-License-Identifier: MIT

    # Copyright Advanced Micro Devices, Inc.
    # SPDX-License-Identifier: MIT

Do not add a year, an "All rights reserved" line, or the full MIT license text
to each file. The full license text lives in `LICENSE` at the repository root.
Put the header above any file-level doc comment, and keep a shebang (and coding
line) on the first line(s) when present.

Third-party, vendored, and generated files keep their upstream headers and are
out of scope. The `amdsmi-license-headers` pre-commit hook
(`projects/amdsmi/tests/check_license_headers.py`) checks this on every commit.

### Process ###

* Reviewers are listed in the CODEOWNERS file
* All code must pass pre-commit checks before review

## References ##

1. [pre-commit](https://github.com/pre-commit/pre-commit)
1. [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
1. [Ruff](https://docs.astral.sh/ruff/)
1. [gersemi](https://github.com/BlankSpruce/gersemi)
