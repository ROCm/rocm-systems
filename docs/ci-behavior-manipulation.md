# CI Behavior Manipulation

TheRock CI is controlled by [`therock_configure_ci.py`](../.github/scripts/therock_configure_ci.py), where it controls push, pull request and workflow dispatch CI behavior.

## Push behavior

For `push`, TheRock CI only runs builds and tests when pushed to the `develop` branch. TheRock CI will run builds and tests for Linux `gfx94X`, `gfx950` and Windows `gfx1511`, `gfx110X`

## Pull request behavior

For `pull_request`, TheRock CI will run builds and tests for Linux `gfx94X`, `gfx950` and Windows `gfx1511`, `gfx110X`

However, if additional options are wanted, you can add a label to manipulate the behavior. The labels we provide are:

- `enable-rocm-libraries`: TheRock CI will include rocm-libraries repository at `HEAD` and will build rocm-libraries

## Workflow dispatch behavior

For `workflow_dispatch`, you are able to trigger CI in [GitHub's therock-ci.yml workflow page](https://github.com/ROCm/rocm-systems/actions/workflows/therock-ci.yml). From [`therock_matrix.py`](../.github/scripts/therock_matrix.py), please select the projects you want to run (ex: `projects/clr`)
