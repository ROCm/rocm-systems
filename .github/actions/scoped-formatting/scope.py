"""Select formatting work from a complete, pinned Git merge-base diff."""

import argparse
import os
import re
import subprocess
from pathlib import Path


def changed_paths(base: str, head: str) -> tuple[str, str, list[str]]:
    """Return the comparison SHAs and paths without fetching file contents.

    Disabling rename detection exposes both sides of a move as independent
    paths and keeps this operation confined to trees in a blobless checkout.
    NUL delimiters preserve spaces, newlines, and non-UTF-8 filename bytes.
    """
    commits = [
        subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            text=True,
        ).strip()
        for ref in (base, head)
    ]
    ancestors = subprocess.check_output(
        ["git", "merge-base", "--all", *commits], text=True
    ).splitlines()
    if len(ancestors) != 1:
        raise RuntimeError("Formatting requires one unambiguous merge base")
    base, head = ancestors[0], commits[1]
    output = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--no-ext-diff",
            "--no-textconv",
            "--no-renames",
            "-z",
            base,
            head,
            "--",
        ]
    )
    return base, head, [os.fsdecode(path) for path in output.split(b"\0") if path]


def matches_scope(paths: list[str], include: str, exclude: str) -> bool:
    """Match explicit include/exclude regexes, always covering this action."""
    included = re.compile(include)
    excluded = re.compile(exclude) if exclude else None
    return any(
        path.startswith(".github/actions/scoped-formatting/")
        or (included.search(path) and not (excluded and excluded.search(path)))
        for path in paths
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    parser.add_argument("--include", required=True)
    parser.add_argument("--exclude", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    base, head, paths = changed_paths(args.base, args.head)
    affected = matches_scope(paths, args.include, args.exclude)
    print(
        f"Inspected all {len(paths)} changed paths in {base}..{head}: "
        + ("formatting required" if affected else "no formatting inputs changed")
    )
    # Only validated Git object IDs and a boolean enter the Actions output file.
    with args.output.open("a", encoding="utf-8") as output:
        output.write(f"base={base}\nhead={head}\naffected={str(affected).lower()}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
