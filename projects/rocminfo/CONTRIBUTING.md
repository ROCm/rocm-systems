<head>
  <meta charset="UTF-8">
  <meta name="description" content="Contributing to rocminfo">
  <meta name="keywords" content="ROCm, contributing, rocminfo">
</head>

# Contributing to `rocminfo`
We welcome contributions to `rocminfo`.  Please follow these guidelines to help ensure your contributions will be successfully accepted.

## Issue Discussion
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

## Acceptance Criteria for Contributions
The goal of `rocminfo` is to provide the user with all the system information that is known to and provided by the HSA/ROCr API. Anybody writing a ROCr application could use rocminfo to see the values the API would provide to their application. The included `rocm_agent_enumerator` prints the list of available AMD GCN ISA devices on the host. Keep these goals in mind when considering the suitability of any new changes; that is, do your changes improve the reliability or capability toward these goals.

## Coding Style
C++ code changes should conform to the [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).

## Pull Request Guidelines
When you create a pull request, you should target the `TODO ADD THIS` branch.  

By creating a pull request, you agree to the statements made in the [code license](#code-license) section. Your pull request should target the default branch. Our current default branch is the `TODO ADD THIS` branch, which serves as our integration branch.

### Deliverables
For each new file in repository, 
please include the licensing header (replace "current year" with the actual current year).
```
/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) <current year>, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */
 ```

### Process
Use the following process when creating a PR.

* Identify the issue you want to fix
* Target the `TODO ADD THIS` branch for integration
* Ensure your code builds successfully
* Verify the output of `rocminfo` and `rocm_agent_enumerator`
  * Verify your change is what was intended
  * Existing, working functionality is still intact
* Submit your PR and work with the reviewer or maintainer to get your PR approved
* Once approved, the PR is brought onto internal CI systems and may be merged into the component
  during our release cycle, as coordinated by the maintainer
* We'll inform you once your change is committed

## Code License
All code contributed to this project will be licensed under the license identified in the [License.txt](./License.txt). Your contribution will be accepted under the same license.

## References
* [License.txt](./License.txt)