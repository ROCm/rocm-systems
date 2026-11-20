# RCCL Maintenance Scripts

`README.md` in this directory is a verbatim copy of NVIDIA's and describes
NVIDIA's workflow. This file covers the scripts RCCL adds, and the places where
RCCL's tooling deliberately does not match what that file says.

Run all commands below from `projects/rccl` in your ROCm/rocm-systems clone, unless otherwise stated.

## Installing pre-commit

[pre-commit](https://pre-commit.com/) can be installed through `pip` or `uv`:

```sh
pip install pre-commit
```

```sh
uv tool install pre-commit
```

### Setting up pre-commit as a pre commit git hook

```sh
pre-commit install
```

This is your best option to ensure that code is correctly formatted and linted before it ever gets committed.

## Running the checks

### Commit normally

If you've run the `pre-commit install` command above, git will automatically run the default set of checks before each commit.

### Run the default checks on staged files

```sh
pre-commit
```

### Run expensive checks, like clang-tidy on staged files

```sh
 pre-commit run --hook-stage manual
```

### Run checks on all files

Just add `--all-files` to the command lines in the sections above.
