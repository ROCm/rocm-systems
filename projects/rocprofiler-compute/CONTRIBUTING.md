## How to fork from us

To keep our development fast and conflict free, we recommend you to [fork](https://github.com/ROCm/rocprofiler-compute/fork) our repository and start your work from our `develop` branch in your private repository.

Afterwards, git clone your repository to your local machine. But that is not it! To keep track of the original develop repository, add it as another remote.

```
git remote add mainline https://github.com/ROCm/rocprofiler-compute.git
git checkout develop
```

As always in git, start a new branch with

```
git checkout -b topic-<yourFeatureName>
```

and apply your changes there. For more help reference GitHub's ['About Forking'](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project) page.

## How to contribute to ROCm Compute Profiler

### Did you find a bug?

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/ROCm/rocprofiler-compute/issues).

- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/ROCm/rocprofiler-compute/issues/new).

### Did you write a patch that fixes a bug?

- Open a new GitHub [pull request](https://github.com/ROCm/rocprofiler-compute/compare) with the patch.

- Ensure the PR description clearly describes the problem and solution. If there is an existing GitHub issue open describing this bug, please include it in the description so we can close it.

- Ensure the PR is based on the `develop` branch of the ROCm Compute Profiler GitHub repository.

> [!TIP]
> To ensure you meet all formatting requirements before publishing, we recommend you utilize our included [*pre-commit hooks*](https://pre-commit.com/#introduction). For more information on how to use pre-commit hooks please see the [section below](#using-pre-commit-hooks).

## Using pre-commit hooks

Our project supports optional [*pre-commit hooks*](https://pre-commit.com/#introduction) which developers can leverage to verify formatting before publishing their code. Once enabled, any commits you propose to the repository will be automatically checked for formatting. Initial setup is as follows:

```console
python3 -m pip install pre-commit
cd rocprofiler-compute
pre-commit install
```

Now, when you commit code to the repository you should see something like this:

![A screen capture showing terminal output from a pre-commit hook](docs/data/contributing/pre-commit-hook.png)

Please see the [pre-commit documentation](https://pre-commit.com/#quick-start) for additional information.

## Contribution Guidelines

To ensure code quality and consistency, we use **Ruff**, a fast Python linter and formatter. Before submitting a pull request, please ensure your code is formatted and linted correctly. This is the manual alternative to running ruff pre-commit hooks.

-----

## Installing and Running Ruff

Ruff is available on PyPI and can be installed using `pip`:

```bash
pip install ruff
```

Once installed, you can run Ruff from the command line. To check for linting errors and formatting issues, navigate to the project root and run:

```bash
ruff check .
ruff format --check .
```

To automatically fix most of the issues detected, you can use the `--fix` flag with the `check` command and run the `format` command without the `--check` flag:

```bash
ruff check --fix .
ruff format .
```

-----

## Disabling Formatting for Specific Sections

There may be instances where you need to disable Ruff's formatting on a specific block of code. You can do this using special comments:

  * **`# fmt: off`** and **`# fmt: on`**: These comments can be used to disable and re-enable formatting for a block of code.
  * **`# fmt: skip`**: This comment, placed at the end of a line, will prevent Ruff from formatting that specific statement.

You can also disable specific linting rules for a line by using `# noqa: <rule_code>`.

## Coding guidelines

Below are some repository specific guidelines which are followed througout the repository.
Any future contributions should adhere to these guidelines:
* Use the `pathlib` library functions instead of `os.path` for manipulating the file paths.

## Build and test documentation changes

For instructions on how to build and test documentation changes (files under docs folder), please see https://rocm.docs.amd.com/en/latest/contribute/contributing.html
