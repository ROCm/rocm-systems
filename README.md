# ROCm System Quick Start Guide for Developers

Welcome to the monorepo for ROCm system software!

This quick start guide includes tips on how to work efficiently with this monorepo style.

## 📂 Sparse Checkout: Working with a Subset of the Monorepo

If you only need to work on specific projects within the monorepo, you can use **sparse checkout** to limit the files in your working directory.

### Enable Sparse Checkout:
```sh
git clone --no-checkout <monorepo-url>
cd <monorepo>
git sparse-checkout init --cone
git sparse-checkout set repo1 repo2
git checkout develop
```
This ensures that only `repo1/` and `repo2/` directories are checked out.

### Reset Sparse Checkout to Full Repo:
```sh
git sparse-checkout disable
```

📌 **Source:** [Git Sparse-Checkout Docs](https://git-scm.com/docs/git-sparse-checkout)

---

## 🔍 Viewing Old History with Subtree Split

Since the monorepo was created using **git subtree**, historical commit paths are preserved at the top-level, but per-file granularity of the commit history requires extra steps to view the history of changes before the individual repos were added to this monorepo. To view the commit history of a specific project before it was merged:

### Create a Temporary History Branch:
```sh
git subtree split --prefix=repo1 -b repo1-history
git checkout repo1-history
```

Now, you can use the typical logging commands to go through the old commit history.

📌 **Source:** [Git Subtree: A Quick Guide to Mastering Git Subtree](https://gitscripts.com/git-subtree)

---

## ⚠️ GitHub Web UI Limitations

The GitHub web interface does not correctly track file history when files are moved into a subdirectory via `git subtree`. To get the prior history of files, use the aforementioned subtree split method.

---

## ⚠️ Per-Project Submodule Limitations

Submodules that were present in the individual projects are not migrated over to the monorepo when `git subtree` is used. For the reverse direction, updates to submodules on the monorepo are also not propagated to the individual repos with `git subtree push`. If an individual project has submodules, please maintain submodules in both the monorepo and the individual repos during the migration period. Also consider why the submodule is in place, and whether a better solution should be implemented for the monorepo (e.g., shared top-level directory of submodules).

---

## 🔄 Keeping the Monorepo Updated

This monorepo will run hourly jobs to synchronize the monorepo with the individual repos using the below commands. Reminder that submodules are ignored during this process.

### Pull Updates from Individual Repo:
```sh
git subtree pull --prefix=repo1 <repo1-url> <repo1-branch>
```

### Push Changes Back to the Individual Repo:
```sh
git subtree push --prefix=repo1 <repo1-url> <repo1-branch>
```

📌 **Source:** [Git Subtree: A Quick Guide to Mastering Git Subtree](https://gitscripts.com/git-subtree)

---

This guide will be updated as needed! 🚀 Happy coding!
