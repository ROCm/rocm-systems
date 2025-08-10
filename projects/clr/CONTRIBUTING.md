# Contributing to HIP/CLR #

We welcome contributions to the HIP project.
CLR is a part of HIP runtime for the AMD platform.Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the [GitHub Issue](https://github.com/ROCm/clr/issues) tab to notify us of issues.

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

clr Compute Language Runtime contains C++ codes for the implementation of HIP runtime APIs on the AMD platform.
Bug fixes and performance are both important goals in clr. Because of this, when a pull request is created, the owner of the repository will review, and put it in automated testing to make sure,
* The change will build on various OS platforms (Ubuntu, RHEL, etc.)
* The build package will install and run the code on different GPU architectures (MI-series, Radeon series cards, etc.),
* And the test results will achieve the goal as expected.

## Code Structure ##

clr contains three parts of codes,
- `hipamd` - contains implementation for HIP runtime on the AMD platform, which includes
   - `include/hip/amd_detail` for headers
   - `/src` for all types of functionality implementation such as hip event, memory, module and texture, etc.

- `opencl` - contains implementation of OpenCL on the AMD platform.

- `rocclr` - contains compute runtime used in HIP and OpenCL, which includes
   - `include`, header files,
   - `device`, implementation of GPU device related interfaces to the backend support,
   - `cimpiler`, implementation of interfaces with compiler,
   - `utils`, implementation of some useful utilities,
   - `os`, implementation of OS related interfaces.


## Coding Style ##

clr is a C++ runtime API implementation on the AMD platform. It allows codeing in C++ programming language, and follows styles as below,
- Code Indentation:
    - Tabs should be expanded to spaces.
    - Use 4 spaces indentation.
- Capitalization and Naming
    - Prefer camelCase for HIP interfaces and internal symbols.  Note `HIP_CLANG` uses `_` for separator.
      This guideline is not yet consistently followed in HIP code - eventual compliance is aspirational.
    - Member variables should begin with a leading "_".  This allows them to be easily distinguished from other variables or functions.

- `{}` placement
    - namespace should be on same line as `{` and separated by a space.
    - Single-line if statement should still use `{/}` pair (even though C++ does not require).
    - For functions, the opening `{` should be placed on a new line.
    - For if/else blocks, the opening `{` is placed on same line as the if/else. Use a space to separate `{` from if/else. For example,
```console
    if (foo) {
        doFoo()
    } else {
        doFooElse();
    }
```

- Miscellaneous
    - All references in function parameter lists should be const.
    - "ihip" means internal hip structures.  These should not be exposed through the HIP API.
    - Keyword TODO refers to a note that should be addressed in long-term.  Could be style issue, software architecture, or known bugs.
    - FIXME refers to a short-term bug that needs to be addressed.

- `HIP_INIT_API()` should be placed at the start of each top-level HIP API.  This function will make sure the HIP runtime is initialized, and also constructs an appropriate API string for tracing and CodeXL marker tracing. The arguments to HIP_INIT_API should match those of the parent function.
- `hipExtGetLastError()` can be called as the AMD platform specific API, to return error code from last HIP API called from the active host thread. `hipGetLastError()` and `hipPeekAtLastError()` can also return the last error that was returned by any of the HIP runtime calls in the same host thread.
- All HIP environment variables should begin with the keyword HIP_
Environment variables should be long enough to describe their purpose but short enough so they can be remembered - perhaps 10-20 characters, with 3-4 parts separated by underscores.
To see the list of current environment variables, along with their values, set HIP_PRINT_ENV and run any hip applications on ROCm platform.
HIPCC or other tools may support additional environment variables which should follow the above convention.

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

HIP is an open source library. Because of this, we include the following license description at the top of every source file.
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

