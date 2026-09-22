# Contributing to hip-tests #

We welcome contributions to the hip-tests project. Please follow these details to help ensure your contributions will be successfully accepted.
If you want to contribute to our documentation, refer to {doc}`Contribute to ROCm docs <rocm:contribute/contributing>`.

## Issue Discussion ##

Please use the [GitHub Issue](https://github.com/ROCm/hip-tests/issues) tab to notify us of issues.

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

HIP-tests is a C++ based Catch2 unit test framework that allows developers to test HIP API functionalities for AMD and NVIDIA GPUs from single source code.
Contributors wishing to make bug fixes or unit test enhancements should run on both AMD and NVIDIA platforms.### Doxygen Editing Guidelines ###

- bugs should be marked with @bugs near the code where the bug might be fixed.  The @bug message will appear in the API description and also in the
doxygen bug list.

## Code Structure ##

hip-tests contains mainly three types of application/test codes,

- `catch/unit`

  This contains catch2 unit tests for different kinds of feature functionalities in HIP, for example,
    - `atomic` is to test all types of atomic operations,
    - `compiler` contains tests to launch kernels,
    - `memory` includes applications to test and use different kinds of memory related HIP APIs.
 etc.

- `perftest`

  This directory has some perftest examples.

- `samples`

  This directory has many hip samples, including codes and instructions for building and executing different kinds of hip features and operations.

## Coding Style ##

Catch2 is a unit testing framework for C++, using cmake integration. For more information and tutorial, you can refer to [catch2 documents] (https://github.com/catchorg/Catch2/blob/devel/docs/tutorial.md).

## Pull Request Guidelines ##

By creating a pull request, you agree to the statements made in the code license section. Your pull request should target the default branch. Our current default branch is the develop branch, which serves as our integration branch.

Follow existing best practice for writing a good Git commit message.

Some tips:
    http://chris.beams.io/posts/git-commit/
    https://robots.thoughtbot.com/5-useful-tips-for-a-better-commit-message

In particular :
   - Use imperative voice, ie "Fix this bug", "Refactor the XYZ routine", "Update the doc".
     Not : "Fixing the bug", "Fixed the bug", "Bug fix", etc.
   - Subject should summarize the commit.  Do not end subject with a period.  Use a blank line
     after the subject.

### Deliverables ###

hip-tests is an open source repository. Because of this, we include the following license description at the top of every source file.
If you create new source files in the repository, please include this text in them as well (replacing "xx" with the digits for the current year):
```
// Copyright (c) 20xx Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
```

### Process ###

After you create a PR, you can take a look at a diff of the changes you made using the PR's "Files" tab.

PRs must pass through the checks and the code review described in the [Acceptance Criteria](#acceptance-criteria) section before they can be merged.

Checks may take some time to complete. You can view their progress in the table near the bottom of the pull request page. You may also be able to use the links in the table
to view logs associated with a check if it fails.

During code reviews, another developer will take a look through your proposed change. If any modifications are requested (or further discussion about anything is
needed), they may leave a comment. You can follow up and respond to the comment, and/or create comments of your own if you have questions or ideas.
When a modification request has been completed, the conversation thread about it will be marked as resolved.

To update the code in your PR (eg. in response to a code review discussion), you can simply push another commit to the branch used in your pull request.

## References ##
* For hip-test repository information, refer to [README](https://github.com/ROCm/hip-tests).
* For how to build and run hip runtime and hip catch2 unint tests from source, on the platform of AMD or NVIDIA, refer to [build HIP] (https://github.com/ROCm/HIP/blob/develop/docs/install/build.rst).
* hip samples give detail instructions on how to build and run HIP applications, you can refer to [HIP samples] (https://github.com/ROCm/hip-tests/tree/develop/samples).
