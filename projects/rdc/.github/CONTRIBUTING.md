# Contributing to RDC #

We welcome contributions to RDC.  Please follow these details to help ensure your contributions will be successfully accepted.

## Issue Discussion ##

Please use the GitHub Issues tab to notify us of issues.

* Use your best judgement for issue creation. If your issue is already listed, upvote the issue and
  comment or post to provide additional details, such as how you reproduced this issue.
* If you're not sure if your issue is the same, err on the side of caution and file your issue.
  You can add a comment to include the issue number (and link) for the similar issue. If we evaluate
  your issue as being the same as the existing issue, we'll close the duplicate.
* If your issue doesn't exist, use the issue template to file a new issue.
  * When filing an issue, be sure to provide as much information as possible,
    including your amdgpu driver version, GPUs used, and commands ran. This
    helps reduce the time required to reproduce your issue.
  * Check your issue regularly, as we may require additional information to successfully reproduce the
    issue.
* You may also open an issue to ask questions to the maintainers about whether a proposed change
  meets the acceptance criteria, or to discuss an idea pertaining to the library.

## Acceptance Criteria ##

The goal of RDC project is to provide a remote control tool with optional
modules.

Contributors wanting to submit additional plugins must follow the guidelines
below.

* New modules/
  * Modules must be loaded at runtime with a `dlopen` call.
  * Modules must have an option to disable them in CMake.

## Code Structure ##

Modules:

    └── rdc_libs
        └── rdc_modules

Module interfaces: (see RdcRasLib.h for example)

    └── include
        └── rdc_lib
            └── impl

Client executable (rdci) source:

    └── rdci

Server executable (rdcd) source:

    └── server

Protos for protobuf and gRPC:

    └── protos
        └── rdc.proto

## Coding Style ##

Please refer to `.clang-format`. It is suggested you use `pre-commit` tool.
It mostly follows Google C++ formatting with 100 character line limit.

## Pull Request Guidelines ##

When you create a pull request, you should target the default branch. Our
current default branch is the **develop** branch, which serves as our
integration branch.

### Deliverables ###

For each new file in repository,
Please include the licensing header

    /*
    Copyright (c) 20xx - present Advanced Micro Devices, Inc. All rights reserved.

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    */

### Process ###

* Reviewers are listed in the CODEOWNERS file
* Code format guidelines

RDC uses the clang-format tool for formatting code in source files.
The formatting style is captured in .clang-format which is located at
the root of RDC. These are different options to follow:

   1. Using pre-commit and docker - `pre-commit run`
   1. Using only clang-format - `clang-format -i \<path-to-the-source-file\>`

## References ##

1. [pre-commit](https://github.com/pre-commit/pre-commit)
1. [clang-format](https://clang.llvm.org/docs/ClangFormat.html)
