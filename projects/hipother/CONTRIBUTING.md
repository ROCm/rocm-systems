# Contributing to hipother #

We welcome contributions to the hipother project. Please follow these details to help ensure your contributions will be successfully accepted.
If you want to contribute to our documentation, refer to {doc}`Contribute to ROCm docs <rocm:contribute/contributing>`.

## Issue Discussion ##

Please use the [GitHub Issue](https://github.com/ROCm/hipother/issues) tab to notify us of issues.

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

HIPOTHER is a C++ Runtime API interface with CUDA APIs that allows developers to create portable applications for AMD and NVIDIA GPUs from single source code. Contributors wishing to submit new HIP Features (ie functions, classes, types) should also consider CUDA APIs.
Differences or limitations of HIP APIs as compared to CUDA APIs should be clearly documented and described.
Some guidelines are outlined below:

### Add a new HIP API ###

- Add a translation to the hipify-clang tool ; many examples abound.
    - For stat tracking purposes, place the API into an appropriate stat category ("dev", "mem", "stream", etc).
- Add a inlined NVIDIA implementation for the function in /hipnv/include/hip/nvidia_detail/nvidia_hip_runtime_api.h in the repository [hipother](https://github.com/ROCm/hipother).
    - These are typically headers
- Add an HIP definition and Doxygen comments for the function in /include/hip/hip_runtime_api.h, in the repository [hip](https://github.com/ROCm/hip).
    - Source implementation typically go in clr/hipamd/src/hip_*.cpp in the reposotory [clr](https://github.com/ROCm/clr). The implementation involves calls to HIP runtime (ie for hipStream_t).

### Run Unit Tests ###

For new features or bug fixes, it's mandatory to run associate [hip-tests](https://github.com/ROCm/hip-tests) on both AMD and NVIDIA platforms.
Please go to the repo and follow the steps.

For applications and benchmarks outside the hip-tests environment, developments should use a two-step development flow:
- #1. Compile, link, and install HIP. See {ref}`Building the HIP runtime` notes.
- #2. Relink the target application to include changes in HIP runtime file.

## Code Structure ##

hipother contains mainly header files with interfaces of different typs of HIP APIs to the corresponding CUDA runtime or driver APIs, for example,
- `hipnv/include/hip/nvidia_detail/nvidia_hip_runtime_api.h` - contains enumerations and HIP runtime API interfaces corresponding to CUDA enumerations and runtime APIs.

- `hipnv/include/hip/nvidia_detail/nvidia_hiprtc.h` - contains HIP runtime compiler enumerations and APIs correspond to CUDA.


## Coding Style ##
- Code Indentation:
    - Tabs should be expanded to spaces.
    - Use 4 spaces indentation.
- Capitalization and Naming
    - Prefer camelCase for HIP interfaces and internal symbols.  Note HCC uses _ for separator.
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

hipother is part of HIP open source library. Because of this, we include the following license description at the top of every source file.
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