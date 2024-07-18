# Contributing to ROCProfiler SDK #

Contributions are welcome. Contributions at a basic level must conform to the MIT license and pass code test requirements (i.e. ctest). The author must also be able to respond to comments/questions on the PR and make any changes requested.

## Issue Discussion ##

Please use the GitHub Issues tab to let us know of any issues.

* Use your best judgment when creating issues. If your issue is already listed, please upvote the issue and
comment or post to provide additional details, such as the way to reproduce this issue.
* If you're unsure if your issue is the same, err on caution and file your issue.
  You can add a comment to include the issue number (and link) for a similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When you file an issue, please provide as much information as possible, including script output, so
we can get information about your configuration. This helps reduce the time required to
    reproduce your issue.
  * Check your issue regularly, as we may require additional information to reproduce the
issue successfully.
* You may also open an issue to ask the maintainers whether a proposed change
meets the acceptance criteria or to discuss an idea about the library.

## Acceptance Criteria ##

Github issues are recommended for any significant change to the code base that adds a feature or fixes a non-trivial issue. If the code change is large without the presence of an issue (or prior discussion with AMD), the change may not be reviewed. Small fixes that fix broken behavior or other bugs are always welcome with or without an associated issue.

## Coding Style ##

All changes must be formatted with clang-format-15/cmake-format before review/acceptance. The exact settings for these formatters must be the ones in this repository.

## Pull Request Guidelines ##

By creating a pull request, you agree to the statements made in the [code license](#code-license) section. Your pull request should target the default branch. Our current default branch is the **develop** branch, which serves as our integration branch.

## Code License ##

All code contributed to this project will be licensed under the license identified in the [License.txt](../LICENSE.txt). Your contribution will be accepted under the same license.

## Release Cadence ##

Any code contribution to this library will be released with the next version of ROCm if the contribution window for the upcoming release is still open. If the contribution window is closed but the PR contains a critical security/bug fix, an exception may be made to include the change in the next release.
