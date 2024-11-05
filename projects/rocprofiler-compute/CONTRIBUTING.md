## How to fork from us

To keep our development fast and conflict free, we recommend you to [fork](https://github.com/ROCm/rocprofiler-compute/fork) our repository and start your work from our `amd-staging` branch in your private repository.

Afterwards, git clone your repository to your local machine. But that is not it! To keep track of the original develop repository, add it as another remote.

```
git remote add mainline https://github.com/ROCm/rocprofiler-compute.git
git checkout amd-staging
```

As always in git, start a new branch with

```
git checkout -b topic-<yourFeatureName>
```

and apply your changes there.

## How to contribute to ROCm Compute Profiler

### Did you find a bug?

- Ensure the bug was not already reported by searching on GitHub under [Issues](https://github.com/ROCm/rocprofiler-compute/issues).

- If you're unable to find an open issue addressing the problem, [open a new one](https://github.com/ROCm/rocprofiler-compute/issues/new).

### Did you write a patch that fixes a bug?

- Open a new GitHub [pull request](https://github.com/ROCm/rocprofiler-compute/compare) with the patch.

- Ensure the PR description clearly describes the problem and solution. If there is an existing GitHub issue open describing this bug, please include it in the description so we can close it.

- Ensure the PR is based on the `amd-staging` branch of the ROCm Compute Profiler GitHub repository.

- ROCm Compute Profiler requires new commits to include a "Signed-off-by" token in the commit message (typically enabled via the `git commit -s` option), indicating your agreement to the projects's [Developer's Certificate of Origin](https://developercertificate.org/) and compatability with the project [LICENSE](LICENSE):


> (a) The contribution was created in whole or in part by me and I
> have the right to submit it under the open source license
> indicated in the file; or
>
> (b) The contribution is based upon previous work that, to the best
> of my knowledge, is covered under an appropriate open source
> license and I have the right under that license to submit that
> work with modifications, whether created in whole or in part
> by me, under the same open source license (unless I am
> permitted to submit under a different license), as indicated
> in the file; or
>
> (c) The contribution was provided directly to me by some other
> person who certified (a), (b) or (c) and I have not modified
> it.
>
> (d) I understand and agree that this project and the contribution
> are public and that a record of the contribution (including all
> personal information I submit with it, including my sign-off) is
> maintained indefinitely and may be redistributed consistent with
> this project or the open source license(s) involved.
