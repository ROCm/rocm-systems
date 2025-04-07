#!/usr/bin/env python3

import os
import tempfile
import subprocess
import shutil


def which(cmd, require):
    v = shutil.which(cmd)
    if require and v is None:
        raise RuntimeError(f"{cmd} not found")
    return v if v is not None else ""


def run(*args, **kwargs):
    if "sensitive" not in kwargs.keys() or not kwargs["sensitive"]:
        print("\n### Executing '{}'... ###\n".format(" ".join(*args)))

    if "sensitive" in kwargs:
        del kwargs["sensitive"]

    return subprocess.run(*args, **kwargs)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="Image files to create markdown links for",
        type=str,
        required=True,
        default=[],
    )
    parser.add_argument(
        "-n",
        "--name",
        help="Name of the PR",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--repo-url",
        help="Base GitHub repo URL",
        type=str,
        default="https://github.com/AMD-ROCm-Internal/rocprofiler-sdk-internal",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        help="Output directory containing the generated markdown files and the list of these files",
        type=str,
        default=".codecov",
    )
    parser.add_argument(
        "--token",
        type=str,
        help="Personal access token or GitHub actions token",
        required=True,
    )
    parser.add_argument(
        "--bot",
        action="store_true",
        help="Configure git user.name and user.email to GitHub Actions Bot",
    )

    args = parser.parse_args()

    inidir = os.getcwd()

    if os.path.exists(args.token):
        with open(args.token, "r") as f:
            token = f.readline().strip()
    else:
        token = args.token

    repo_url = args.repo_url
    repo_url_protocol = args.repo_url.split("//")[0]
    repo_url_addr = args.repo_url.split("//", maxsplit=1)[-1]
    repo_url_secure = f"{repo_url_protocol}//oauth2:{token}@{repo_url_addr}.git"
    files = [os.path.abspath(itr) for itr in args.files]

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Working directory: {tmpdir}")

        git_cmd = which("git", require=True)
        print(f"git executable: {git_cmd}")

        run([git_cmd, "--version"])

        run(
            [
                git_cmd,
                "clone",
                repo_url_secure,
                tmpdir,
            ],
            sensitive=True,
            check=True,
        )

        os.chdir(tmpdir)

        if args.bot:
            run(
                [git_cmd, "config", "--local", "user.name", "github-actions[bot]"],
                check=True,
            )
            run(
                [
                    git_cmd,
                    "config",
                    "--local",
                    "user.email",
                    "41898282+github-actions[bot]@users.noreply.github.com",
                ],
                check=True,
            )

        _branch = f"images-{args.name}"
        _ref_branch = f"refs/{_branch}/image-ref"

        run(["pwd"])
        run([git_cmd, "switch", "--orphan", _branch], check=True)
        run([git_cmd, "commit", "--allow-empty", "-m", "Empty commit"], check=True)
        run([git_cmd, "push", "origin", f"HEAD:{_ref_branch}"], check=False)
        run([git_cmd, "fetch", "origin", _ref_branch], check=True)
        run([git_cmd, "pull", "--rebase", "origin", _ref_branch], check=True)
        run([git_cmd, "reset", "--hard", "HEAD^"], check=False)

        if not os.path.exists(args.name):
            os.makedirs(args.name)

        filenames = {}
        for itr in files:
            bname = os.path.basename(itr)
            _, ext = os.path.splitext(bname)
            oname = os.path.join(args.name, bname)
            filenames[bname] = oname
            shutil.copy2(itr, os.path.join(tmpdir, oname))

        run([git_cmd, "add", args.name])
        run([git_cmd, "status"])
        run([git_cmd, "commit", "-m", f"{args.name} code coverage files"])
        run([git_cmd, "push", "--force", "origin", f"HEAD:{_ref_branch}"], check=True)

        log = run([git_cmd, "log", "-n", "1", "--format=%H"], capture_output=True)
        hash = log.stdout.decode("utf-8").strip()
        info = {}
        for bitr, titr in filenames.items():
            info[bitr] = (
                f"![code coverage {bitr}]({repo_url}/blob/{hash}/{titr}?raw=True)"
            )

        os.chdir(inidir)

        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)

        metafname = os.path.join(args.output_dir, "gh-md-image-links.txt")
        with open(metafname, "w") as mofs:
            print(f"\n### Writing '{metafname}'... ###\n")
            for lbl, msg in info.items():
                mdfname = os.path.join(args.output_dir, f"{lbl}.md")
                with open(mdfname, "w") as ofs:
                    print(f"\n### Writing '{mdfname}'... ###\n")
                    ofs.write(f"\n{msg}\n\n")
                    mofs.write(f"{mdfname}\n")
