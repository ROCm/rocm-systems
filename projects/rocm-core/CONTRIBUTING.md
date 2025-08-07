# ROCM-CORE Contributing Guide
To ensure the quality of the ROCM-CORE code base, the ROCM-CORE team has 
established a code review process to inform developers of the steps 
that are required to shepherd a change-set into the repository.

#### Table Of Contents

[How to get started](#How-to-get-started)

[How do I contribute?](#how-do-i-contribute)
  * [Reporting Issues](#reporting-issues)
  * [Creating a Pull Request](#Creating-a-Pull-Request)

[Responsibility of the Author](#Responsibility-of-the-Author)

[Responsibility of the Reviewer](#Responsibility-of-the-Reviewer)

[The Review](#the-review)

[References](#References)
## How to get started
rocm-core is a utility which can be used to get ROCm release version. 
It also provides the Lmod modules files for the ROCm release.
getROCmVersion function provides the ROCm version. 

## How do I contribute
### Deliverables
All contributions you make will be under the [MIT Software License](copyright).
For each new file in repository, 
Please include the licensing header
```
/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 20xx Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/
```

### Formatting the code

All the code is formatted using `clang-format`. To format a file, use:

```shell
clang-format-10 -style=file -i <path-to-source-file>
```

 
### Reporting Issues
We use [GitHub Issues](https://github.com/ROCm/rocm-core/issues) to track public **bugs** and **enhancement requests**.

#### Bugs
Please follow the template below to report any bugs found in ROCM-CORE:

1. Description: ***Please be clear and descriptive***
2. How to Reproduce the issue:
* Hardware Information:
* Docker environment or Software Version:
* Expected behavior:
* Actual behavior:
3. Any additional information:

#### Enhancement Requests
Please follow the template below to report any enhancement requests for ROCM-CORE:

1. Description: ***Please be clear and descriptive***
2. Value and Motivation:
* Feature and functionalities enabled:
* Any alternatives:
3. Any additional information:

The author must set labels (and assigns a milestone) according to his/her own understanding.

Other contributors can change these values if they disagree. That being said, 
adding a small comment explaining the motivation is highly recommended. 
In this way, we keep the process flexible while cultivating mutual understanding.

[**Note**] Most likely, the labels like "bug", "feature" or "complexity*" 
would not be changed. However, "value*" or "urgency*" might be from mutual 
understanding.
### Creating a Pull Request

No changes are allowed to be directly committed to the develop 
branch of the ROCM-CORE repository. All authors are required to 
develop their change sets on a separate branch and then create 
a pull request (PR) to merge their changes into the develop branch.

Once a PR has been created, a developer must choose two reviewers 
to review the changes made. The first reviewer should be a 
technical expert in the portion of the library that the changes 
are being made in. You can find a list of these experts in 
[CODEOWNERS](CODEOWNERS) list. 
The second reviewer should be a peer reviewer. This reviewer 
can be any other ROCM-CORE developer.

## Responsibility of the Author
The author of a PR is responsible for:
 * Writing clear, well documented code
 * Meeting expectations of code quality
 * Verifying that the changes do not break current functionality
 * Writing tests to ensure code coverage
 * Report on the impact to performance

## Responsibility of the Reviewer
Each reviewer is responsible for verifying that the changes are 
clearly written in keeping with the coding styles of the library, 
are documented in a way that future developers will be able to 
understand the intent of the added functionality, and will 
maintain or improve the overall quality of the codebase.

Reviewer's task checklist:
1. Has the PR passed?
2. Does the PR consist of a well-organized sequence of small commits, 
each of which is designed to make one specific feature or fix ?
3. Does the PR only include a reviewable amount of changes? Or it is a 
consolidation of already reviewed small batches? e.g. break it into smaller 
testable and reviewable tasks instead of a huge chunk at once.
4. Does the PR have sufficient documentation and easy to read and understand, 
feasible for test and future maintainence, related docs already in place? 
if API or functionality has changed?
5. For bugfixes and new features, new regression test created?
6. Is every PR associated with a ticket or issue number for tracking purposes?

## The Review
During the review, reviewers will look over the changes and make 
suggestions or requests for changes.

In order to assist the reviewer in prioritizing their efforts, 
authors can take the following actions:

* Set the urgency and value labels
* Set the milestone where the changes need to be delivered
* Describe the testing procedure and post the measured effect of 
  the change
* Remind reviewers via email if a PR needs attention
* If a PR needs to be reviewed as soon as possible, explain to 
  the reviewers why a review may need to take priority

## References

1. [ROCM-CORE Readme](README.md)

